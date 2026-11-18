"""Load dispatcher registrations without importing a CPython extension."""

from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import torch


def load_library():
    version = torch.__version__.split("+", 1)[0].split(".")
    if tuple(map(int, version[:2])) < (2, 11):
        raise ImportError("sgl-fa3 requires PyTorch >= 2.11")

    package_dir = Path(__file__).resolve().parent
    for suffix in EXTENSION_SUFFIXES:
        library = package_dir / f"_C{suffix}"
        if library.is_file():
            torch.ops.load_library(str(library))
            return
    raise ImportError(f"Could not find the sgl-fa3 native library in {package_dir}")
