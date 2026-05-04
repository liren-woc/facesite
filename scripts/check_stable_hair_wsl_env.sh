#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${STABLE_HAIR_WSL_ENV_NAME:-stablehair}"
CONDA_ROOT="${STABLE_HAIR_WSL_CONDA_ROOT:-$HOME/miniconda3}"
REPO_DIR="${1:-${PROJECT_ROOT}/third_party/Stable-Hair}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  python3 - <<PY
import json
print(json.dumps({
    "python": None,
    "modules": {},
    "cuda_visible_devices": None,
    "huggingface_token_set": False,
    "ld_library_path": None,
    "nvidia_smi_on_path": False,
    "repo_infer_exists": False,
    "config_exists": False,
    "error": f"WSL conda env not found: ${ENV_NAME}",
}))
PY
  exit 0
fi
conda activate "${ENV_NAME}"
set -u

export HF_HOME="${STABLE_HAIR_HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_DIR}"
python - <<'PY'
import importlib.util
import json
import os
import shutil
import sys

mods = ["torch", "diffusers", "transformers", "omegaconf", "cv2", "PIL", "xformers"]
payload = {
    "python": sys.executable,
    "modules": {m: importlib.util.find_spec(m) is not None for m in mods},
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "huggingface_token_set": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
    "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
    "nvidia_smi_on_path": shutil.which("nvidia-smi") is not None,
    "repo_infer_exists": os.path.exists("infer_full.py"),
    "config_exists": os.path.exists("configs/hair_transfer.yaml"),
}
print(json.dumps(payload))
PY
