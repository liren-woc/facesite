from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))

    from hairstyle_tryon.backends.factory import create_generation_backend

    parser = argparse.ArgumentParser(description="Run one backend generation request in a clean subprocess.")
    parser.add_argument("--backend", required=True, choices=["stable_hair"])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--face-path", required=True)
    parser.add_argument("--shape-path", required=True)
    parser.add_argument("--color-path", default=None)
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args()

    backend = create_generation_backend(
        args.backend,
        repo_dir=Path(args.repo),
        python_executable=args.python_executable,
    )
    result_path = backend.run(
        face_path=args.face_path,
        shape_path=args.shape_path,
        color_path=args.color_path,
        result_path=args.result_path,
    )
    print(Path(result_path).resolve())


if __name__ == "__main__":
    main()
