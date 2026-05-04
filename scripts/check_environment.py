from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path


MODULES = [
    "torch",
    "torchvision",
    "numpy",
    "cv2",
    "mediapipe",
    "PIL",
    "gradio",
    "yaml",
    "pandas",
    "sklearn",
]


def module_status(name: str) -> dict:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return {"installed": False, "version": None}

    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
    except Exception as exc:
        return {"installed": True, "version": None, "import_error": str(exc)}

    return {"installed": True, "version": version}


def torch_status() -> dict:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False}

    import torch

    version = getattr(torch, "__version__", None)
    cuda_module = getattr(torch, "cuda", None)
    torch_version_module = getattr(torch, "version", None)
    if cuda_module is None or not hasattr(cuda_module, "is_available"):
        return {
            "installed": True,
            "version": version,
            "import_warning": "Imported module named 'torch', but it does not expose the expected PyTorch API.",
            "module_repr": repr(torch),
        }

    cuda_available = torch.cuda.is_available()
    payload = {
        "installed": True,
        "version": version,
        "cuda_available": cuda_available,
        "torch_cuda_version": getattr(torch_version_module, "cuda", None),
        "device": None,
    }
    if cuda_available:
        payload["device"] = torch.cuda.get_device_name(0)
    return payload


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cuda_home = os.environ.get("CUDA_HOME")
    payload = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "project_root": str(project_root),
        "modules": {name: module_status(name) for name in MODULES},
        "torch": torch_status(),
        "stable_hair_runtime_toolchain": {
            "ninja_on_path": shutil.which("ninja") is not None,
            "cl_on_path": shutil.which("cl") is not None,
            "nvcc_on_path": shutil.which("nvcc") is not None,
            "cuda_home": cuda_home,
            "cuda_home_exists": bool(cuda_home and Path(cuda_home).exists()),
        },
        "mediapipe_models": {
            "face_landmarker_task_exists": (
                project_root / "models" / "face_landmarker.task"
            ).exists(),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
