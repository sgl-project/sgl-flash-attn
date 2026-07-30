#!/usr/bin/env python
"""FA2 vs FA3 decode benchmark (hd512 by default), single script.

- Both go through flash_attn_with_kvcache + cache_seqlens (fair comparison).
- Each measurement is preceded by a correctness check against an fp32 reference,
  so a silently-wrong "fast" path can never win.
- Pick an idle GPU first (shared box!):  CUDA_VISIBLE_DEVICES=1 python bench_fa2_fa3_hd512.py

Usage examples:
    python bench_fa2_fa3_hd512.py                      # default: bf16, hd512
    python bench_fa2_fa3_hd512.py --dtype fp16 --hd 128
    python bench_fa2_fa3_hd512.py --ns 0 16 --iters 100
"""
import argparse
import math

import torch

try:
    import flash_attn as fa2
    from flash_attn import flash_attn_with_kvcache as fa2_kv
except ImportError as e:
    fa2, fa2_kv = None, None
    fa2_import_error = e
import flash_attn_interface as fa3
from flash_attn_interface import flash_attn_with_kvcache as fa3_kv


def ref_attn(q, k, v, causal):
    """fp32 reference, GQA-aware. Returns (ref, lp_ref_err_bound)."""
    b, sq, hq, d = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    kr = k.repeat_interleave(hq // hkv, dim=2)
    vr = v.repeat_interleave(hq // hkv, dim=2)
    if causal:
        i = torch.arange(sq, device=q.device).view(-1, 1)
        j = torch.arange(sk, device=q.device).view(1, -1)
        mask = j > (sk - sq) + i
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kr.float()) / math.sqrt(d)
    if causal:
        s.masked_fill_(mask, float("-inf"))
    ref = torch.einsum("bhqk,bkhd->bqhd", s.softmax(-1), vr.float())
    # same computation but with the P matrix rounded to the low precision dtype,
    # which bounds the error any correct low-precision kernel can legitimately have
    s2 = torch.einsum("bqhd,bkhd->bhqk", q, kr).float() / math.sqrt(d)
    if causal:
        s2.masked_fill_(mask, float("-inf"))
    lp = torch.einsum("bhqk,bkhd->bqhd", s2.softmax(-1).to(q.dtype).float(), vr.float())
    return ref, (lp - ref).abs().max().item()


def out_of(x):
    return x[0] if isinstance(x, tuple) else x


def time_fn(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def graph_time_fn(fn, iters, warmup=10):
    """Capture one call into a CUDA Graph and time replays (no host launch overhead)."""
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def make_l2_flush(mbytes):
    """A copy big enough to evict L2, so the timed kernel starts cold -- what a
    real model sees, since other layers stream through L2 between two visits of
    the same attention layer."""
    src = torch.empty((mbytes << 20) // 2, dtype=torch.bfloat16, device="cuda")
    dst = torch.empty_like(src)
    return lambda: dst.copy_(src)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hd", type=int, default=512, help="QK head dim")
    p.add_argument("--hdv", type=int, default=None,
                   help="V head dim (default: same as --hd). If different, only FA3 runs (FA2 requires K/V same dim)")
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--hq", type=int, default=8)
    p.add_argument("--hkv", type=int, default=1)
    p.add_argument("--sq", type=int, default=1)
    p.add_argument("--causal", type=int, choices=[0, 1], default=1,
                   help="1=causal (default), 0=non-causal; only matters when sq>1 given full cache_seqlens")
    p.add_argument("--graph", type=int, choices=[0, 1], default=0,
                   help="1 = time CUDA Graph replays (no host launch overhead); correctness still checked eagerly")
    p.add_argument("--page-size", type=int, default=0,
                   help="paged KV page size; 0 = contiguous cache (no paging). FA2 uses block_table, FA3 uses page_table")
    p.add_argument("--pool-tokens", dest="pool_tokens", type=int, default=0,
                   help="paged KV pool size in tokens (needs --page-size>0). The "
                   "request's pages are scattered over this pool, like a server's "
                   "token_to_kv_pool. 0 = tight pool (pages contiguous, L2-friendly)")
    p.add_argument("--l2-flush-mb", dest="l2_flush_mb", type=int, default=0,
                   help="MB copied inside the timed region to evict L2 before the "
                   "kernel, then subtracted out. 0 = L2 stays warm (optimistic for "
                   "decode shapes whose KV fits in the 50MB L2)")
    p.add_argument("--ns", type=int, nargs="+", default=[0], help="num_splits values (0=auto)")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--cases", type=str, default="1x32768,4x32768,32x4096",
                   help="comma-separated bxcache, e.g. 1x32768,32x4096")
    p.add_argument("--dump", type=int, default=0, metavar="N",
                   help="print first N output elements per impl side-by-side with the fp32 gold (0=off)")
    args = p.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    hdv = args.hdv if args.hdv is not None else args.hd
    run_fa2 = hdv == args.hd
    fa2_skip_reason = "hd_qk != hd_v: FA2 skipped (requires K/V same head dim), FA3 only"
    if run_fa2 and fa2_kv is None:
        run_fa2 = False
        fa2_skip_reason = f"FA2 not importable ({fa2_import_error}): FA3 only"
    if run_fa2 and args.page_size > 0 and args.page_size % 256 != 0:
        run_fa2 = False
        fa2_skip_reason = f"page_size={args.page_size} not a multiple of 256: FA2 skipped (its paged KV requires %256==0), FA3 only"
    causal = bool(args.causal)
    timer = graph_time_fn if args.graph else time_fn
    flush = make_l2_flush(args.l2_flush_mb) if args.l2_flush_mb else None

    def timed(fn, iters):
        """Kernel time with a cold L2: time(flush + kernel) - time(flush)."""
        if flush is None:
            return timer(fn, iters)
        base = timer(flush, iters)
        return timer(lambda: (flush(), fn()), iters) - base

    cases = [tuple(map(int, c.split("x"))) for c in args.cases.split(",")]
    dev = torch.cuda.current_device()
    print(f"device: {torch.cuda.get_device_name(dev)} (cuda:{dev})")
    print(f"fa2: {fa2.__file__ if fa2 is not None else 'n/a'} ({getattr(fa2, '__version__', '?')})")
    print(f"fa3: {fa3.__file__}")
    print(f"config: {args.dtype} hd_qk={args.hd} hd_v={hdv} hq{args.hq}/hkv{args.hkv} sq{args.sq} causal={causal} graph={bool(args.graph)} page_size={args.page_size} iters={args.iters}")
    print(f"pool_tokens={args.pool_tokens} l2_flush={args.l2_flush_mb}MB")
    if not run_fa2:
        print(fa2_skip_reason)
    torch.manual_seed(0)

    header = f"{'case':>16} {'ns':>4} {'FA2 ms':>9} {'FA2 GB/s':>9} {'FA3 ms':>9} {'FA3 GB/s':>9} {'FA3/FA2':>8}"
    print("\n" + header)
    print("-" * len(header))
    all_ok = True
    for b, cache in cases:
        q = torch.randn(b, args.sq, args.hq, args.hd, dtype=dtype, device="cuda") / 3
        k = torch.randn(b, cache, args.hkv, args.hd, dtype=dtype, device="cuda") / 3
        v = torch.randn(b, cache, args.hkv, hdv, dtype=dtype, device="cuda") / 3
        cs = torch.full((b,), cache, dtype=torch.int32, device="cuda")
        ref, lp_err = ref_attn(q, k, v, causal)
        if args.page_size > 0:
            page = args.page_size
            npp = (cache + page - 1) // page
            need = b * npp
            pool_pages = max(need, args.pool_tokens // page)
            k_in = torch.zeros(pool_pages, page, args.hkv, args.hd, dtype=dtype, device="cuda")
            v_in = torch.zeros(pool_pages, page, args.hkv, hdv, dtype=dtype, device="cuda")
            if pool_pages > need:
                # scattered slots inside a big pool, like a server's KV allocator
                ids = torch.randperm(pool_pages, device="cuda")[:need].to(torch.int32)
            else:
                ids = torch.arange(need, dtype=torch.int32, device="cuda")
            kp = torch.zeros(b, npp * page, args.hkv, args.hd, dtype=dtype, device="cuda")
            vp = torch.zeros(b, npp * page, args.hkv, hdv, dtype=dtype, device="cuda")
            kp[:, :cache] = k
            vp[:, :cache] = v
            k_in[ids.long()] = kp.view(need, page, args.hkv, args.hd)
            v_in[ids.long()] = vp.view(need, page, args.hkv, hdv)
            del kp, vp
            pt = ids.reshape(b, npp)
            extra = {"FA2": {"block_table": pt}, "FA3": {"page_table": pt}}
        else:
            k_in, v_in = k, v
            extra = {"FA2": {}, "FA3": {}}
        tol = max(3 * lp_err, 1e-3) + 1e-5
        # KV bytes read (K+V) once per token step
        kv_bytes = b * cache * args.hkv * (args.hd + hdv) * q.element_size()

        impls = [("FA2", fa2_kv), ("FA3", fa3_kv)] if run_fa2 else [("FA3", fa3_kv)]
        for ns in args.ns:
            times = {}
            cols = {}
            for name, fn in impls:
                out = out_of(fn(q, k_in, v_in, cache_seqlens=cs, causal=causal, num_splits=ns, **extra[name]))
                err = (out.float() - ref).abs().max().item()
                if not (torch.isfinite(out.float()).all().item() and err <= tol):
                    all_ok = False
                    print(f"FAIL correctness {name} b={b} cache={cache} ns={ns} err={err:.2e} tol={tol:.2e}")
                if args.dump > 0:
                    g = ref.flatten().float()
                    o = out.flatten().float()
                    n = min(args.dump, g.numel())
                    amax = (o - g).abs().argmax().item()
                    print(f"\n[dump] {name} b={b} cache={cache} ns={ns}  err={err:.2e} tol={tol:.2e}  (showing {n}/{g.numel()} elems)")
                    print(f"       {'idx':>8} {'gold':>14} {'out':>14} {'abs_diff':>12}")
                    for i in range(n):
                        print(f"       {i:>8} {g[i].item():>14.6f} {o[i].item():>14.6f} {abs(o[i]-g[i]).item():>12.2e}")
                    print(f"       {'^max@'+str(amax):>8} {g[amax].item():>14.6f} {o[amax].item():>14.6f} {abs(o[amax]-g[amax]).item():>12.2e}")
                ms = timed(lambda: fn(q, k_in, v_in, cache_seqlens=cs, causal=causal, num_splits=ns, **extra[name]), args.iters)
                times[name] = ms
                cols[name] = f" {ms:>9.4f} {kv_bytes / (ms * 1e-3) / 1e9:>9.0f}"
            fa2_col = cols.get("FA2", f" {'n/a':>9} {'n/a':>9}")
            ratio = f" {times['FA2'] / times['FA3']:>7.2f}x" if run_fa2 else f" {'n/a':>8}"
            print(f"{f'b={b} cache={cache}':>16} {ns:>4}{fa2_col}{cols['FA3']}{ratio}")
    print("\nall correctness checks PASS" if all_ok else "\nSOME CORRECTNESS CHECKS FAILED — timings above are not comparable!")


if __name__ == "__main__":
    main()
