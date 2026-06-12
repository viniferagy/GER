#!/usr/bin/env bash
set -euo pipefail

# Environment installer for the GEC/GER + representation-engineering workflow.
# Target machine: 4 x A800-80G, NVIDIA driver 560.35.03, CUDA 12.6.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
GER_ROOT="${GER_ROOT:-${PROJECT_ROOT}/multilingual}"
VENV_DIR="${VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Optional heavy build. Keep this off unless the code path really imports flash_attn.
# On older Linux images, prebuilt flash-attn wheels can require a newer GLIBC than
# the host provides, so when enabled we force a local source build.
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi

uv python install "${PYTHON_VERSION}"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
DS_BUILD_OPS=0 uv sync --python "${PYTHON_VERSION}" --extra cuda126
uv pip install --python "${VENV_DIR}/bin/python" -r requirements.txt
source "${VENV_DIR}/bin/activate"
SITE_PACKAGES="$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
NVIDIA_LIB_DIRS="$(find "${SITE_PACKAGES}/nvidia" -type d -path '*/lib' 2>/dev/null | paste -sd ':' -)"
if [ -n "${NVIDIA_LIB_DIRS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}:${LD_LIBRARY_PATH:-}"
fi

if [ "${INSTALL_FLASH_ATTN}" = "1" ]; then
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "INSTALL_FLASH_ATTN=1 requires nvcc so flash-attn can be built against this host GLIBC."
        echo "Either load a CUDA devel module/toolkit first, or leave INSTALL_FLASH_ATTN=0."
        exit 1
    fi
    MAX_JOBS="${MAX_JOBS:-8}" uv pip install \
        --reinstall \
        --no-binary flash-attn \
        --no-build-isolation \
        "flash-attn==2.7.4.post1"
fi

python - <<'PY'
import torch
print("python ok")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu count:", torch.cuda.device_count())
    print("gpu 0:", torch.cuda.get_device_name(0))

import transformers, datasets, sklearn, spacy
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("sklearn:", sklearn.__version__)

from repe import repe_pipeline_registry
repe_pipeline_registry()
print("repe ok")
PY

echo
echo "Done. Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Run GEC scripts from:"
echo "  cd ${GER_ROOT}"
