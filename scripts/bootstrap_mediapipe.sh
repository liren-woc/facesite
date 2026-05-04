#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${PROJECT_ROOT}/models"
MODEL_PATH="${MODEL_DIR}/face_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_PATH}" ]; then
  echo "MediaPipe face landmarker already exists: ${MODEL_PATH}"
  exit 0
fi

if command -v curl >/dev/null 2>&1; then
  curl -L "${MODEL_URL}" -o "${MODEL_PATH}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${MODEL_PATH}" "${MODEL_URL}"
else
  echo "curl or wget is required to download ${MODEL_URL}"
  exit 1
fi

echo "Saved MediaPipe face landmarker to ${MODEL_PATH}"
