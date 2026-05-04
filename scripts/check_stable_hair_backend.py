from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from hairstyle_tryon.backends.stable_hair import StableHairBackend

    parser = argparse.ArgumentParser(description="Check the Stable-Hair backend workspace.")
    parser.add_argument("--stable-hair-repo", default="third_party/Stable-Hair")
    parser.add_argument("--stable-hair-python", default=None)
    args = parser.parse_args()

    stable_hair_python = args.stable_hair_python
    config_path = project_root / "configs" / "stable_hair_python.txt"
    if stable_hair_python is None and config_path.exists():
        stable_hair_python = config_path.read_text(encoding="utf-8").lstrip("\ufeff").strip() or None

    backend = StableHairBackend(
        repo_dir=Path(args.stable_hair_repo),
        python_executable=stable_hair_python,
    )
    payload = backend.doctor()
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
