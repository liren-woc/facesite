#!/usr/bin/env bash
set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${STABLE_HAIR_WSL_ENV_NAME:-stablehair}"
CONDA_ROOT="${STABLE_HAIR_WSL_CONDA_ROOT:-$HOME/miniconda3}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
set -u

export HF_HOME="${STABLE_HAIR_HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

python "${PROJECT_ROOT}/scripts/run_stable_hair_inference.py" "$@"
