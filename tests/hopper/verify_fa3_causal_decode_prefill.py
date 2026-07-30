import torch, math
from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache

def unwrap(x):
    return x[0] if isinstance(x, tuple) else x

def ref_attn(q, k, v, causal):
    # fp32 reference, bottom-right aligned causal (FA semantics)
    b, sq, hq, d = q.shape
    sk, hkv = k.shape[1], k.shape[2]
    kr = k.repeat_interleave(hq // hkv, dim=2)
    vr = v.repeat_interleave(hq // hkv, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kr.float()) / math.sqrt(d)
    if causal:
        i = torch.arange(sq, device=q.device).view(-1, 1)
        j = torch.arange(sk, device=q.device).view(1, -1)
        s.masked_fill_(j > (sk - sq) + i, float("-inf"))
    p = s.softmax(-1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, vr.float())
    # low-precision reference for tolerance scaling
    s2 = torch.einsum("bqhd,bkhd->bhqk", q, kr).float() / math.sqrt(d)
    if causal:
        s2.masked_fill_(j > (sk - sq) + i, float("-inf"))
    o2 = torch.einsum("bhqk,bkhd->bqhd", s2.softmax(-1).to(q.dtype).float(), vr.float())
    return o, o2

def check(name, out, ref, ref_lp):
    err = (out.float() - ref).abs().max().item()
    err_lp = (ref_lp - ref).abs().max().item()
    ok = torch.isfinite(out.float()).all().item() and err <= max(3 * err_lp, 1e-3) + 1e-5
    print(f"{'PASS' if ok else 'FAIL'}  {name}  err={err:.2e} (lp_ref={err_lp:.2e})")
    return ok

fails = 0
torch.manual_seed(0)
for dtype in (torch.bfloat16, torch.float16):
    tag = "bf16" if dtype == torch.bfloat16 else "fp16"
    for hd in (512, 128):
        # ---- causal decode: seqlen_q > 1, KV cache ----
        for sq in (1, 2, 5, 17, 32):
            for b, cache in ((1, 33), (2, 1000), (3, 4097)):
                hq, hkv = 8, 1
                q = torch.randn(b, sq, hq, hd, dtype=dtype, device="cuda") / 3
                k = torch.randn(b, cache + 64, hkv, hd, dtype=dtype, device="cuda") / 3
                v = torch.randn_like(k)
                cs = torch.full((b,), cache, dtype=torch.int32, device="cuda")
                ref, ref_lp = ref_attn(q, k[:, :cache], v[:, :cache], causal=True)
                for ns in (1, 2, 8, 16, 0):
                    out = unwrap(flash_attn_with_kvcache(q, k, v, cache_seqlens=cs, causal=True, num_splits=ns))
                    fails += not check(f"{tag} hd{hd} decode  sq={sq:2d} b={b} cache={cache:4d} ns={ns:2d}", out, ref, ref_lp)
        # ---- causal prefill: flash_attn_func, seqlen_q == seqlen_k ----
        for b, s in ((1, 128), (2, 517), (1, 2048)):
            for hq, hkv in ((8, 1), (8, 8)):
                q = torch.randn(b, s, hq, hd, dtype=dtype, device="cuda") / 3
                k = torch.randn(b, s, hkv, hd, dtype=dtype, device="cuda") / 3
                v = torch.randn_like(k)
                ref, ref_lp = ref_attn(q, k, v, causal=True)
                out = unwrap(flash_attn_func(q, k, v, causal=True))
                fails += not check(f"{tag} hd{hd} prefill sq=sk={s:4d} b={b} hq/hkv={hq}/{hkv}", out, ref, ref_lp)
print("=" * 60)
print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
