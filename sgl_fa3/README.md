# sgl-fa3

The FlashAttention-3 forward kernels [SGLang](https://github.com/sgl-project/sglang)
serves with, packaged as a standalone wheel. Until this package existed they
were compiled into `sglang-kernel` as `sgl_kernel.flash_ops`.

The kernels come from `../hopper` in this repository; this directory holds the
torch op registration, the Python wrapper, the tests and the wheel build script.
The build image, the wheel retag script and the release index script live in
sgl-project/sglang, next to those of the other SGLang-built wheels.

## What is built

- Forward only (`FLASHATTENTION_DISABLE_BACKWARD`); sparse mask kernels off by default.
- Only the variants SGLang reaches, so the wheel stays under PyPI's file size limit:
  - no append-KV / rotary path (`FLASHATTENTION_DISABLE_APPENDKV`): SGLang writes the KV cache itself;
  - split, non-TMA paged and softcap kernels for varlen only
    (`FLASHATTENTION_VARLEN_ONLY_SPLIT_PAGED_SOFTCAP`); a dense call that needs one is run as
    varlen with a full-length `seqused_k`, which gives the same result;
  - no Q/K 64 with V 256 (`FLASHATTENTION_DISABLE_HDIMDIFF64_256`); absorbed MLA uses V 512.

  Calls that need a compiled-out variant raise instead of running something else.
- `sm_90a`, plus `sm_80`/`sm_86` (A100, A10, L40, 4090). No SM100+ targets.
- BF16, FP16 and FP8 (E4M3).
- x86_64 Linux, CUDA 13.0 (`+cu130`), one abi3 wheel for CPython 3.10+.
- Built against PyTorch 2.11's stable C++ ABI (`TORCH_TARGET_VERSION=0x020B000000000000`).
  The same binary supports PyTorch 2.11 and newer; the CUDA and GPU requirements still apply.

## Usage

```python
from sgl_fa3 import flash_attn_varlen_func, flash_attn_with_kvcache, get_scheduler_metadata
```

The ops are registered as `torch.ops.sgl_fa3.fwd` and
`torch.ops.sgl_fa3.get_scheduler_metadata`.
The Python package loads the native library through `torch.ops.load_library`;
it does not import a pybind module.

## Building

```bash
sgl_fa3/scripts/build.sh 3.10 13.0 /path/to/sglang
```

builds in a `pytorch/manylinux2_28-builder:cuda13.0` container (image from
sglang's `docker/sgl-fa3.Dockerfile`, retagged with sglang's
`scripts/rename_sgl_fa3_whl.sh`) and writes
`sgl_fa3/dist/` (`+cu130` wheel for the sgl-whl index) and
`sgl_fa3/dist-pypi/` (same wheel without the local version, for PyPI). The
build reads `../hopper`, so it needs a full checkout; an sdist is not supported.

For a local build against an installed torch:

```bash
cd sgl_fa3 && pip install . --no-build-isolation -v
```

## Releasing

Releases are cut by the `Release sgl-fa3` workflow in sgl-project/sglang
(`.github/workflows/release-whl-fa3.yml`). It checks out this branch, writes the
version into `python/sgl_fa3/_version.py`, runs `scripts/build.sh`, tests the
wheel on H200 with `tests/`, then publishes to PyPI and the
sgl-project/whl index (sglang's `scripts/update_fa3_whl_index.py`).
