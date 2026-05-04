#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="${STABLE_HAIR_ENV_NAME:-stablehair}"
CONDA_ROOT="${STABLE_HAIR_WSL_CONDA_ROOT:-$HOME/miniconda3}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${PROJECT_ROOT}/third_party/Stable-Hair"
TMP_REQ="$(mktemp)"

if [ ! -d "${REPO_DIR}" ]; then
  mkdir -p "${PROJECT_ROOT}/third_party"
  git clone https://github.com/Xiaojiu-z/Stable-Hair "${REPO_DIR}"
fi

source "${CONDA_ROOT}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip
conda run -n "${ENV_NAME}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  torch==2.2.2 torchvision==0.17.2

grep -Ev '^(torch|torchvision|xformers|bitsandbytes|triton|nvidia-)' "${REPO_DIR}/requirements.txt" > "${TMP_REQ}"
conda run -n "${ENV_NAME}" python -m pip install -r "${TMP_REQ}"
conda run -n "${ENV_NAME}" python -m pip install xformers==0.0.25.post1 bitsandbytes==0.44.1
rm -f "${TMP_REQ}"

echo "Stable-Hair environment is ready:"
echo "  conda env: ${ENV_NAME}"
echo "  python:    $(conda run -n "${ENV_NAME}" python -c 'import sys; print(sys.executable)')"
echo "  doctor:    ${PROJECT_ROOT}/scripts/check_stable_hair_wsl_env.sh"
