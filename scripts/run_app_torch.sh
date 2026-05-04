#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-D:/anaconda/envs/pytorch/python.exe}"
GENERATOR_BACKEND="${GENERATOR_BACKEND:-stable_hair}"
GENERATOR_REPO="${GENERATOR_REPO:-}"
GENERATOR_PYTHON="${GENERATOR_PYTHON:-}"
STABLE_HAIR_PYTHON="${STABLE_HAIR_PYTHON:-}"
STABLE_HAIR_PYTHON_FILE="${PROJECT_ROOT}/configs/stable_hair_python.txt"

cd "${PROJECT_ROOT}"
if [[ -z "${STABLE_HAIR_PYTHON}" && -f "${STABLE_HAIR_PYTHON_FILE}" ]]; then
  STABLE_HAIR_PYTHON="$(tr -d '\r\n' < "${STABLE_HAIR_PYTHON_FILE}")"
fi
if [[ -z "${GENERATOR_REPO}" ]]; then
  GENERATOR_REPO="third_party/Stable-Hair"
fi
if [[ -z "${GENERATOR_PYTHON}" ]]; then
  GENERATOR_PYTHON="${STABLE_HAIR_PYTHON}"
fi

CMD=(
  "${PYTHON_EXE}" -m hairstyle_tryon.app
  --generator-backend "${GENERATOR_BACKEND}"
  --generator-repo "${GENERATOR_REPO}"
  --output-dir outputs/tryon
)

if [[ -n "${GENERATOR_PYTHON}" ]]; then
  CMD+=(--generator-python "${GENERATOR_PYTHON}")
fi

PYTHONPATH=src "${CMD[@]}" "$@"
