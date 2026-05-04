from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    label: str
    session_dir: Path


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", value.strip(), flags=re.UNICODE)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-_")
    return cleaned[:48] or "session"


def create_session_dir(base_dir: str | Path, label: str | None = None) -> SessionInfo:
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    normalized_label = _slugify(label or "session")
    session_id = f"{timestamp}_{normalized_label}"
    session_dir = (base_path / session_id).resolve()
    suffix = 1
    while session_dir.exists():
        session_id = f"{timestamp}_{normalized_label}_{suffix}"
        session_dir = (base_path / session_id).resolve()
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "uploads").mkdir(parents=True, exist_ok=True)
    return SessionInfo(session_id=session_id, label=normalized_label, session_dir=session_dir)


def snapshot_inputs(
    session_dir: str | Path,
    *,
    face_inputs: dict[str, str | None],
    extra_inputs: dict[str, str | None] | None = None,
) -> dict[str, str]:
    target_root = Path(session_dir) / "uploads"
    target_root.mkdir(parents=True, exist_ok=True)
    stored: dict[str, str] = {}

    all_inputs: dict[str, str | None] = dict(face_inputs)
    if extra_inputs:
        all_inputs.update(extra_inputs)

    for view, source in all_inputs.items():
        if not source:
            continue
        source_path = Path(source)
        if not source_path.exists():
            continue
        suffix = source_path.suffix.lower() or ".jpg"
        target_path = target_root / f"{view}{suffix}"
        shutil.copy2(source_path, target_path)
        stored[view] = str(target_path)

    return stored


def write_session_profile(session_dir: str | Path, payload: dict[str, Any]) -> Path:
    session_dir = Path(session_dir)
    profile_path = session_dir / "session_profile.json"
    profile_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return profile_path


def append_session_index(base_dir: str | Path, record: dict[str, Any]) -> Path:
    index_path = Path(base_dir) / "session_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return index_path
