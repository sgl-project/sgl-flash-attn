if [ -d "build" ]; then
    rm -rf build/
fi

mkdir -p build/sgl_fa4 
mkdir -p dist

cp -r flash_attn/cute build/sgl_fa4/cute
cp -f sgl_fa4/pyproject.toml build/sgl_fa4/cute/pyproject.toml
cp -f sgl_fa4/rename_imports.py build/rename_imports.py

python build/rename_imports.py --target-dir build/sgl_fa4/cute --old-pkg "flash_attn.cute" --new-pkg "sgl_fa4.cute"

python -m build build/sgl_fa4/cute --wheel --no-isolation
cp build/sgl_fa4/cute/dist/*.whl ./dist

rm -rf build/
