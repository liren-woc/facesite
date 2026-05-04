from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class StableHairNotReady(RuntimeError):
    """Raised when the Stable-Hair backend has not been bootstrapped yet."""


@dataclass(frozen=True)
class StableHairBackend:
    repo_dir: Path
    python_executable: str | None = None

    def _project_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def _read_text_config(self, path: Path) -> str | None:
        if not path.exists():
            return None
        value = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
        return value or None

    def resolve_python_executable(self) -> str:
        if self.python_executable:
            return self.python_executable
        env_python = os.environ.get("STABLE_HAIR_PYTHON")
        if env_python:
            return env_python
        config_python = self._read_text_config(
            self._project_root() / "configs" / "stable_hair_python.txt"
        )
        if config_python:
            return config_python
        return sys.executable

    def resolve_pretrained_model_path(self) -> str | None:
        env_value = os.environ.get("STABLE_HAIR_PRETRAINED_MODEL_PATH")
        if env_value:
            return env_value
        config_value = self._read_text_config(
            self._project_root() / "configs" / "stable_hair_sd15_path.txt"
        )
        if config_value:
            return config_value
        return None

    @staticmethod
    def _is_wsl_target(target: str) -> bool:
        return target.startswith("wsl://")

    @staticmethod
    def _windows_to_wsl_path(path: str | Path) -> str:
        resolved = str(Path(path).resolve())
        drive, remainder = resolved[0], resolved[2:].replace("\\", "/")
        return f"/mnt/{drive.lower()}{remainder}"

    def _resolve_wsl_distro(self) -> str:
        target = self.resolve_python_executable()
        if not self._is_wsl_target(target):
            raise StableHairNotReady("WSL backend requested, but target is not a wsl:// URI.")
        distro = target.removeprefix("wsl://").strip()
        return distro or "Ubuntu"

    def _build_wsl_command(self, stable_hair_args: list[str]) -> list[str]:
        project_root = self._project_root()
        runner = self._windows_to_wsl_path(project_root / "scripts" / "run_stable_hair_wsl.sh")
        distro = self._resolve_wsl_distro()
        joined_args = " ".join(shlex.quote(arg) for arg in stable_hair_args)
        shell_command = f"{shlex.quote(runner)} {joined_args}"
        return ["wsl", "-d", distro, "bash", "-lc", shell_command]

    def _build_wsl_doctor_command(self) -> list[str]:
        project_root = self._project_root()
        runner = self._windows_to_wsl_path(project_root / "scripts" / "check_stable_hair_wsl_env.sh")
        repo_dir = self._windows_to_wsl_path(self.repo_dir)
        distro = self._resolve_wsl_distro()
        shell_command = f"{shlex.quote(runner)} {shlex.quote(repo_dir)}"
        return ["wsl", "-d", distro, "bash", "-lc", shell_command]

    @staticmethod
    def _run_process(cmd: list[str], *, cwd: Path | None = None, label: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or str(exc)).strip()
            if len(message) > 3000:
                message = message[-3000:]
            raise RuntimeError(f"{label} failed: {message or exc}") from exc

    def ensure_ready(self) -> None:
        repo_dir = self.repo_dir.resolve()
        stage1_dir = repo_dir / "models" / "stage1"
        stage2_dir = repo_dir / "models" / "stage2"
        stage1_model = stage1_dir / "pytorch_model.bin"
        stage2_model = stage2_dir / "pytorch_model.bin"
        stage2_adapter = stage2_dir / "pytorch_model_1.bin"
        stage2_controlnet = stage2_dir / "pytorch_model_2.bin"
        runner_path = self._project_root() / "scripts" / "run_stable_hair_inference.py"

        if not repo_dir.exists():
            raise StableHairNotReady(f"Stable-Hair repo not found: {repo_dir}")
        if not (repo_dir / "infer_full.py").exists():
            raise StableHairNotReady(f"Stable-Hair inference entry not found under: {repo_dir}")
        if not (repo_dir / "configs" / "hair_transfer.yaml").exists():
            raise StableHairNotReady(f"Stable-Hair config not found under: {repo_dir}")
        if not stage1_dir.exists():
            raise StableHairNotReady(
                f"Stable-Hair stage1 weights are missing under: {stage1_dir}"
            )
        if not stage2_dir.exists():
            raise StableHairNotReady(
                f"Stable-Hair stage2 weights are missing under: {stage2_dir}"
            )
        if not stage1_model.exists():
            raise StableHairNotReady(
                f"Stable-Hair stage1 main checkpoint is missing: {stage1_model}"
            )
        if not stage2_model.exists():
            raise StableHairNotReady(
                f"Stable-Hair stage2 main checkpoint is missing: {stage2_model}"
            )
        if not stage2_adapter.exists():
            raise StableHairNotReady(
                f"Stable-Hair stage2 adapter checkpoint is missing: {stage2_adapter}"
            )
        if not stage2_controlnet.exists():
            raise StableHairNotReady(
                f"Stable-Hair stage2 controlnet checkpoint is missing: {stage2_controlnet}"
            )
        if not runner_path.exists():
            raise StableHairNotReady(
                f"Stable-Hair local runner script is missing: {runner_path}"
            )

        python_target = self.resolve_python_executable()
        if self._is_wsl_target(python_target):
            return

        python_executable = Path(python_target)
        if not python_executable.exists():
            raise StableHairNotReady(
                f"Stable-Hair python not found: {python_executable}. "
                "Pass --generator-python or set STABLE_HAIR_PYTHON."
            )

    def doctor(self) -> dict[str, object]:
        self.ensure_ready()
        repo_dir = self.repo_dir.resolve()
        stage1_dir = repo_dir / "models" / "stage1"
        stage2_dir = repo_dir / "models" / "stage2"
        stage1_model = stage1_dir / "pytorch_model.bin"
        stage2_model = stage2_dir / "pytorch_model.bin"
        stage2_adapter = stage2_dir / "pytorch_model_1.bin"
        stage2_controlnet = stage2_dir / "pytorch_model_2.bin"
        infer_script = repo_dir / "infer_full.py"
        config_path = repo_dir / "configs" / "hair_transfer.yaml"
        local_runner = self._project_root() / "scripts" / "run_stable_hair_inference.py"
        pretrained_model_path = self.resolve_pretrained_model_path()
        python_executable = self.resolve_python_executable()
        pretrained_model_local = False
        if pretrained_model_path:
            pretrained_model_local = Path(pretrained_model_path).exists() or (
                self._is_wsl_target(python_executable) and pretrained_model_path.startswith("/")
            )

        stage1_files = sorted(path.name for path in stage1_dir.glob("*")) if stage1_dir.exists() else []
        stage2_files = sorted(path.name for path in stage2_dir.glob("*")) if stage2_dir.exists() else []

        try:
            if self._is_wsl_target(python_executable):
                result = subprocess.run(
                    self._build_wsl_doctor_command(),
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
            else:
                probe = (
                    "import importlib.util, json, os, sys;"
                    "mods=['torch','diffusers','transformers','omegaconf','cv2','PIL','xformers'];"
                    "payload={"
                    "'python': sys.executable,"
                    "'modules': {m: (importlib.util.find_spec(m) is not None) for m in mods},"
                    "'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),"
                    "'huggingface_token_set': bool(os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')),"
                    "'repo_infer_exists': os.path.exists('infer_full.py'),"
                    "'config_exists': os.path.exists('configs/hair_transfer.yaml')"
                    "};"
                    "print(json.dumps(payload))"
                )
                result = subprocess.run(
                    [python_executable, "-c", probe],
                    cwd=repo_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="ignore",
                )
            runtime = json.loads(result.stdout)
        except subprocess.CalledProcessError as exc:
            runtime = {
                "python": python_executable,
                "modules": {},
                "error": exc.stderr.strip() or exc.stdout.strip() or str(exc),
            }

        modules = runtime.get("modules", {})
        runtime_ready = bool(modules) and all(bool(value) for value in modules.values()) and not runtime.get("error")
        message = (
            "Stable-Hair repository, checkpoints, and launcher are present."
            if runtime_ready
            else "Stable-Hair files are present, but the runtime environment is not ready yet."
        )
        return {
            "repo_dir": str(repo_dir),
            "python": python_executable,
            "pretrained_model_path": pretrained_model_path or "stable-diffusion-v1-5/stable-diffusion-v1-5",
            "pretrained_model_is_local_dir": pretrained_model_local,
            "infer_script_exists": infer_script.exists(),
            "config_exists": config_path.exists(),
            "local_runner_exists": local_runner.exists(),
            "stage1_dir_exists": stage1_dir.exists(),
            "stage2_dir_exists": stage2_dir.exists(),
            "stage1_model_exists": stage1_model.exists(),
            "stage2_model_exists": stage2_model.exists(),
            "stage2_adapter_exists": stage2_adapter.exists(),
            "stage2_controlnet_exists": stage2_controlnet.exists(),
            "stage1_files": stage1_files,
            "stage2_files": stage2_files,
            "ready": runtime_ready,
            "message": message,
            "note": (
                "The public Stable-Hair release still expects a source face and a reference hairstyle image. "
                "A vetted internal hairstyle reference library is still required for no-upload-reference product use."
            ),
            "runtime": runtime,
        }

    def run(
        self,
        face_path: str | Path,
        shape_path: str | Path,
        color_path: str | Path | None = None,
        result_path: str | Path = "outputs/tryon/result.png",
    ) -> Path:
        self.ensure_ready()

        repo_dir = self.repo_dir.resolve()
        face_path = Path(face_path).resolve()
        shape_path = Path(shape_path).resolve()
        result_path = Path(result_path).resolve()
        result_path.parent.mkdir(parents=True, exist_ok=True)

        python_target = self.resolve_python_executable()
        runner_path = self._project_root() / "scripts" / "run_stable_hair_inference.py"
        pretrained_model_path = self.resolve_pretrained_model_path()
        run_args = [
            "--repo",
            self._windows_to_wsl_path(repo_dir) if self._is_wsl_target(python_target) else str(repo_dir),
            "--face-path",
            self._windows_to_wsl_path(face_path) if self._is_wsl_target(python_target) else str(face_path),
            "--shape-path",
            self._windows_to_wsl_path(shape_path) if self._is_wsl_target(python_target) else str(shape_path),
            "--result-path",
            self._windows_to_wsl_path(result_path) if self._is_wsl_target(python_target) else str(result_path),
        ]
        low_vram_size = os.environ.get("STABLE_HAIR_SIZE", "384")
        low_vram_steps = os.environ.get("STABLE_HAIR_STEPS", "20")
        run_args.extend(["--size", str(low_vram_size), "--steps", str(low_vram_steps)])
        if pretrained_model_path:
            if self._is_wsl_target(python_target) and Path(pretrained_model_path).exists():
                run_args.extend(["--pretrained-model-path", self._windows_to_wsl_path(pretrained_model_path)])
            else:
                run_args.extend(["--pretrained-model-path", pretrained_model_path])

        if self._is_wsl_target(python_target):
            self._run_process(self._build_wsl_command(run_args), label="Stable-Hair WSL inference")
        else:
            cmd = [python_target, str(runner_path), *run_args]
            self._run_process(cmd, label="Stable-Hair inference")

        if not result_path.exists():
            raise RuntimeError(f"Stable-Hair did not produce the expected result file: {result_path}")
        return result_path
