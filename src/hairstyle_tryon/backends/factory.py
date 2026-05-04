from __future__ import annotations

from pathlib import Path

from .stable_hair import StableHairBackend


def create_generation_backend(
    backend_name: str,
    *,
    repo_dir: str | Path,
    python_executable: str | None = None,
):
    normalized = backend_name.strip().lower()
    if normalized == "stable_hair":
        return StableHairBackend(repo_dir=Path(repo_dir), python_executable=python_executable)
    raise ValueError(f"Unsupported generation backend: {backend_name}")
