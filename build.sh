#!/usr/bin/env bash

# 1. Exit on error, undefined variables, and pipe failures
set -euo pipefail

# 2. Configuration - Use variables for easier maintenance
BUILD_DIR="build"
TARGET_DIR="${BUILD_DIR}/sgl_fa4"
SOURCE_PKG="sgl_fa4"
PYTHON_EXE=$(which python3 || which python)

# 3. Ensure we are in the correct directory
# (Optional: check for a marker file like pyproject.toml)
if [[ ! -f "pyproject.toml" && ! -d "sgl_fa4" ]]; then
    echo "Error: Script must be run from the project root." >&2
    exit 1
fi

echo "--- Starting build process ---"

# 4. Clean up safely
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning old build directory..."
    rm -rf "$BUILD_DIR"
fi

# 5. Prepare directories and dependencies
mkdir -p "$TARGET_DIR"

echo "Installing build dependencies..."
$PYTHON_EXE -m pip install --upgrade "setuptools-git-versioning>=3.0,<4" build

# 6. Copy source code using rsync
# Added --copy-links to handle symlinks if they exist
echo "Copying source files..."
rsync -rt --exclude="pyproject.toml" flash_attn/cute "$TARGET_DIR/"
rsync -t --exclude="__init__.py" "${SOURCE_PKG}/"* "$BUILD_DIR/"
rsync -t "${SOURCE_PKG}/__init__.py" "$TARGET_DIR/"

# 7. Fix version and imports
echo "Generating version and refactoring imports..."

# Check if command exists before running
if ! command -v setuptools-git-versioning build &> /dev/null; then
    echo "Error: setuptools-git-versioning not found." >&2
    exit 1
fi

printf "%s" "$(setuptools-git-versioning "$BUILD_DIR")" > "${TARGET_DIR}/VERSION"

# Portable sed: Works on both Linux and macOS
sed -i.bak '/__version__/d' "${TARGET_DIR}/cute/__init__.py" && rm "${TARGET_DIR}/cute/__init__.py.bak"

$PYTHON_EXE "${BUILD_DIR}/rename_imports.py" \
    --target-dir "${TARGET_DIR}/cute" \
    --old-pkg "flash_attn.cute" \
    --new-pkg "sgl_fa4.cute"

# 8. Build wheel
echo "Building wheel..."
$PYTHON_EXE -m build "$BUILD_DIR"

echo "--- Build successful! ---"
