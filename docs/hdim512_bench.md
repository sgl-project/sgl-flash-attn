# Head dim 512 decode benchmark (SM90)

Workload: gemma4-style decode, `head_dim = 512`, `q_len = 1`, `q_head = 8`,
`k_head = 1`, `batch = 1`, bf16, causal.

## Setup

- GPU: H200 (sm90a), CUDA 13.0 (nvcc V13.0.88), torch 2.11.0+cu130
- FA3 numbers produced by:

```bash
python tests/hopper/bench_fa2_fa3_hd512.py --ns 0 --iters 100 --graph 1 \
  --cases 1x1,1x64,1x512,1x2048,1x4096,1x8192,1x16384,1x32768
```

- `sglang-triton` and `ffpa-*` columns come from external implementations benchmarked under the same shapes.

## Latency (ms, lower is better)

| cache | sglang-triton | ffpa-cute | ffpa-triton | ffpa-sm80 | FA2-16x32 | FA2-32x32 | FA2-32x16 | FA3 | FA3 vs sglang-triton |
|------:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0070 | 0.014 | 0.042 | 0.023 | 0.0064 | 0.0060 | 0.0053 | 0.0095 | 0.74x |
| 64 | 0.0072 | 0.014 | 0.046 | 0.023 | 0.0106 | 0.0100 | 0.0098 | 0.0096 | 0.75x |
| 512 | 0.0124 | 0.014 | 0.011 | 0.024 | 0.0136 | 0.0132 | 0.0159 | 0.0138 | 0.90x |
| 2048 | 0.0441 | 0.019 | 0.017 | 0.049 | 0.0234 | 0.0232 | 0.0339 | 0.0146 | 3.02x |
| 4096 | 0.0850 | 0.026 | 0.029 | 0.076 | 0.0350 | 0.0348 | 0.0367 | 0.0166 | 5.12x |
| 8192 | 0.1680 | 0.040 | 0.052 | 0.131 | 0.0393 | 0.0399 | 0.0416 | 0.0190 | 8.84x |
| 16384 | 0.3267 | 0.069 | 0.099 | 0.241 | 0.0482 | 0.0508 | 0.0518 | 0.0234 | 13.96x |
| 32768 | 0.6513 | 0.126 | 0.190 | 0.460 | 0.0667 | 0.0712 | 0.0753 | 0.0366 | 17.80x |

## Notes

- For short caches (<= 512) FA3 is at parity or slightly behind, where launch/setup overhead dominates.
- FA3 scales best with cache length: 3.0x at 2K and 17.8x at 32K over sglang-triton, and it is the fastest of all measured implementations from cache=2048 onward.
- Correctness check: `python tests/hopper/verify_fa3_causal_decode_prefill.py` must end with `ALL PASS`.
