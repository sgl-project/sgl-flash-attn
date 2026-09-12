"""Regression for overlapping Q/V and epilogue shared memory on SM86/SM89."""

import pytest
import torch


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "head_dim,local",
    [(192, False), (256, False), (256, True)],
    ids=["head192-split-control", "head256-split", "head256-local"],
)
@pytest.mark.parametrize(
    "batch_size,seqlen_q,seqlen_k,num_splits",
    [(1, 1, 256, 2), (2, 32, 513, 4), (8, 512, 2048, 4)],
)
def test_sm8x_shared_memory(
    batch_size, seqlen_q, seqlen_k, num_splits, head_dim, local, dtype
):
    _check_sm8x_shared_memory(
        batch_size, seqlen_q, seqlen_k, 1 if local else num_splits,
        head_dim, local, dtype,
    )


def _check_sm8x_shared_memory(
    batch_size, seqlen_q, seqlen_k, num_splits, head_dim, local, dtype,
    page_size=None,
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (8, 6), (8, 9)
    ):
        pytest.skip("This shared-memory layout regression targets SM86 and SM89")

    # Read the flags from the installed build, not the current shell environment.
    # Defer extension imports until after the architecture check so this file can
    # be collected on machines that do not exercise the affected kernel.
    from flash_attn_config import CONFIG

    required_features = ["SM8x", "VARLEN", f"HDIM{head_dim}"]
    if dtype == torch.float16:
        required_features.append("FP16")
    if local:
        required_features.append("LOCAL")
    if num_splits > 1:
        required_features.append("SPLIT")
    if page_size is not None:
        required_features.append("PAGEDKV")
    disabled = [
        feature for feature in required_features
        if CONFIG["build_flags"][f"FLASHATTENTION_DISABLE_{feature}"]
    ]
    if disabled:
        pytest.skip(f"Required features were not compiled: {', '.join(disabled)}")

    from flash_attn_interface import flash_attn_with_kvcache, get_scheduler_metadata
    from test_util import attention_ref

    torch.random.manual_seed(12)
    device = "cuda"
    nheads, nheads_kv = 4, 2
    q = torch.randn(batch_size, seqlen_q, nheads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seqlen_k, nheads_kv, head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cache_seqlens = seqlen_k - (
        torch.arange(batch_size, device=device, dtype=torch.int32) % 3
    ) * 7
    key_padding_mask = (
        torch.arange(seqlen_k, device=device)[None, :] < cache_seqlens[:, None]
    )
    window_size = (32, 0) if local else (-1, -1)
    expected = attention_ref(
        q, k, v, key_padding_mask=key_padding_mask, window_size=window_size
    )[0]

    k_cache, v_cache, page_table = k, v, None
    if page_size is not None:
        assert seqlen_k % page_size == 0
        pages_per_sequence = seqlen_k // page_size
        num_pages = batch_size * pages_per_sequence
        page_table = torch.arange(
            num_pages, device=device, dtype=torch.int32
        ).roll(1).reshape(batch_size, pages_per_sequence)
        k_cache = torch.empty(
            num_pages, page_size, nheads_kv, head_dim, device=device, dtype=dtype
        )
        v_cache = torch.empty_like(k_cache)
        # The reference retains logical sequence order; the API must translate
        # absolute local-attention block coordinates through the shuffled pages.
        physical_pages = page_table.flatten().long()
        k_cache[physical_pages] = k.reshape_as(k_cache)
        v_cache[physical_pages] = v.reshape_as(v_cache)

    scheduler_metadata = get_scheduler_metadata(
        batch_size, seqlen_q, seqlen_k, nheads, nheads_kv, head_dim,
        cache_seqlens, dtype, window_size=window_size,
        num_splits=num_splits, pack_gqa=True, page_size=page_size,
    )
    if not local and batch_size == 8:
        # The prepared scheduler stores num_splits_dynamic and num_m_blocks in
        # the first two batch vectors (each padded to a multiple of four).
        # These 8-warp kernels launch one persistent CTA per SM. More work
        # tiles than SMs forces shared-storage reuse across CTA iterations.
        batch_rounded = (batch_size + 3) // 4 * 4
        work_tiles = int((
            scheduler_metadata[:batch_size]
            * scheduler_metadata[batch_rounded:batch_rounded + batch_size]
        ).sum().item()) * nheads_kv
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        assert work_tiles > sm_count, (work_tiles, sm_count)

    def run():
        return flash_attn_with_kvcache(
            q, k_cache, v_cache, cache_seqlens=cache_seqlens, window_size=window_size,
            num_splits=num_splits, pack_gqa=True,
            scheduler_metadata=scheduler_metadata, page_table=page_table,
        )

    out = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run()
    for _ in range(3):
        # Attention is linear in V. Alternating its sign also detects a replay
        # that leaves stale output instead of reading the current cache values.
        v_cache.neg_()
        expected.neg_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mode", ["local-split", "local-paged"])
def test_sm8x_local_block_coordinates(dtype, mode):
    _check_sm8x_shared_memory(
        batch_size=2, seqlen_q=32, seqlen_k=512,
        num_splits=4 if mode == "local-split" else 1,
        head_dim=256, local=True, dtype=dtype,
        page_size=256 if mode == "local-paged" else None,
    )
