"""Calls that reach the variants sgl-fa3 prunes or reroutes.

The build keeps split, non-TMA paged and softcap kernels for varlen only
(FLASHATTENTION_VARLEN_ONLY_SPLIT_PAGED_SOFTCAP): mha_fwd must turn a dense call
that needs one into a varlen call with the same result. The dense unsplit path
(what multimodal_gen uses) must keep working, and the compiled-out variants
(append KV, Q/K 64 with V 256) must fail loudly.
"""

import math

import pytest
import torch

from sgl_fa3 import flash_attn_varlen_func, flash_attn_with_kvcache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="covers the sm90 kernels",
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _attention_ref(q, k, v, causal=False, softcap=0.0):
    """q (b, sq, h, d), k/v (b, sk, hk, d*), fp32 math, bottom-right causal."""
    b, sq, h, d = q.shape
    sk, hk = k.shape[1], k.shape[2]
    k = k.repeat_interleave(h // hk, dim=2).float()
    v = v.repeat_interleave(h // hk, dim=2).float()
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), k) / math.sqrt(d)
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if causal:
        row = torch.arange(sq, device=q.device)[:, None]
        col = torch.arange(sk, device=q.device)[None, :]
        scores = scores.masked_fill(col > row + sk - sq, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", scores.softmax(-1), v)


def _qkv(b=2, sq=128, sk=512, h=8, hk=2, d=128, dv=None):
    torch.manual_seed(0)
    q = torch.randn(b, sq, h, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn(b, sk, hk, d, device=DEVICE, dtype=DTYPE)
    v = torch.randn(b, sk, hk, dv or d, device=DEVICE, dtype=DTYPE)
    return q, k, v


def _check(out, ref):
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("d", [64, 128, 256])
def test_dense_unsplit_stays_supported(causal, d):
    # multimodal_gen passes 4D tensors with no cu_seqlens, num_splits=1.
    q, k, v = _qkv(sq=512, sk=512, d=d)
    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=None, cu_seqlens_k=None,
        max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1], causal=causal,
    )
    _check(out, _attention_ref(q, k, v, causal=causal))


@pytest.mark.parametrize("causal", [False, True])
def test_dense_softcap_runs_as_varlen(causal):
    q, k, v = _qkv()
    out = flash_attn_with_kvcache(q, k, v, causal=causal, softcap=30.0, num_splits=1)
    _check(out, _attention_ref(q, k, v, causal=causal, softcap=30.0))


@pytest.mark.parametrize("num_splits", [0, 4])
def test_dense_split_runs_as_varlen(num_splits):
    q, k, v = _qkv(sq=1, sk=4096)
    out = flash_attn_with_kvcache(q, k, v, num_splits=num_splits)
    _check(out, _attention_ref(q, k, v))


@pytest.mark.parametrize("page_size", [1, 16])
def test_dense_paged_runs_as_varlen(page_size):
    q, k, v = _qkv(sq=4, sk=256)
    b, sk = k.shape[:2]
    pages_per_seq = sk // page_size
    perm = torch.randperm(b * pages_per_seq, device=DEVICE)
    page_table = perm.view(b, pages_per_seq).to(torch.int32)
    k_cache = torch.empty(b * pages_per_seq, page_size, *k.shape[2:], device=DEVICE, dtype=DTYPE)
    v_cache = torch.empty(b * pages_per_seq, page_size, *v.shape[2:], device=DEVICE, dtype=DTYPE)
    k_cache[perm] = k.reshape(b * pages_per_seq, page_size, *k.shape[2:])
    v_cache[perm] = v.reshape(b * pages_per_seq, page_size, *v.shape[2:])
    out = flash_attn_with_kvcache(q, k_cache, v_cache, page_table=page_table, causal=True)
    _check(out, _attention_ref(q, k, v, causal=True))


def test_append_kv_is_rejected():
    q, k, v = _qkv(sq=1)
    cache_seqlens = torch.full((q.shape[0],), 256, device=DEVICE, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="appending KV"):
        flash_attn_with_kvcache(
            q, k, v, k=k[:, :1].clone(), v=v[:, :1].clone(), cache_seqlens=cache_seqlens
        )


def test_qk64_v256_is_rejected():
    q, k, v = _qkv(d=64, dv=256)
    with pytest.raises(RuntimeError, match="V headdim in \\(64, 256\\]"):
        flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=None, cu_seqlens_k=None,
            max_seqlen_q=q.shape[1], max_seqlen_k=k.shape[1],
        )
