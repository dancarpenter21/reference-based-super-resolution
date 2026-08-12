#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_path="${repo_root}/.venv"
wheel_dir="${repo_root}/.cache/rocm-wheels"
mkdir -p "${wheel_dir}"

uv venv --python 3.12 "${venv_path}"
uv pip install --python "${venv_path}/bin/python" -e "${repo_root}[dev]"

base="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"
declare -a wheels=(
  "torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
  "torchvision-0.24.0%2Brocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl"
  "triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"
)
for wheel in "${wheels[@]}"; do
  decoded="${wheel//%2B/+}"
  if [[ ! -f "${wheel_dir}/${decoded}" ]]; then
    curl --fail --location --output "${wheel_dir}/${decoded}.part" "${base}/${wheel}"
    mv "${wheel_dir}/${decoded}.part" "${wheel_dir}/${decoded}"
  fi
done
uv pip install --python "${venv_path}/bin/python" "${wheel_dir}"/*.whl

# AMD's wheel bundles a native HSA runtime; WSL must resolve the DXG-aware system runtime.
torch_lib="$(${venv_path}/bin/python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "lib")')"
rm -f "${torch_lib}"/libhsa-runtime64.so*

"${venv_path}/bin/python" -m ml_engine.cli gpu-check

