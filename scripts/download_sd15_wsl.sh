#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
LOCAL_DIR="${2:-$HOME/stable-hair-cache/sd15}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate stablehair

mkdir -p "$LOCAL_DIR"

python - <<'PY' "$REPO_ID" "$LOCAL_DIR"
import sys
import time
from pathlib import Path

from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
local_dir = Path(sys.argv[2]).expanduser()
local_dir.mkdir(parents=True, exist_ok=True)

required_files = [
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/pytorch_model.bin",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "vae/config.json",
    "vae/diffusion_pytorch_model.bin",
    "safety_checker/config.json",
    "safety_checker/pytorch_model.bin",
]

failures = []
for name in required_files:
    success = False
    for attempt in range(1, 6):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=name,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"OK {name} -> {path}", flush=True)
            success = True
            break
        except Exception as exc:
            print(f"RETRY {name} attempt={attempt} error={exc}", flush=True)
            time.sleep(min(8 * attempt, 30))
    if not success:
        failures.append(name)

if failures:
    raise SystemExit("Failed to download required files: " + ", ".join(failures))

print(local_dir)
PY
