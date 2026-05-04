from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CSV_COLUMNS = [
    "face_shape_hint",
    "hairline_height_hint",
    "face_ratio_h_w",
    "forehead_to_face_ratio",
    "presentation_preference",
    "maintenance_preference",
    "forehead_goal",
    "preferred_style_tag",
    "age_group",
    "style_id",
    "label",
]


def extract_feedback_row(payload: dict[str, Any], *, style_id: str, label: int) -> dict[str, Any]:
    analysis = payload.get("analysis", {})
    metrics = analysis.get("session_metrics", {})
    return {
        "face_shape_hint": str(analysis.get("dominant_face_shape_hint", "unknown")),
        "hairline_height_hint": str(analysis.get("dominant_hairline_height_hint", "unknown")),
        "face_ratio_h_w": metrics.get("mean_face_ratio_h_w"),
        "forehead_to_face_ratio": metrics.get("mean_forehead_to_face_ratio"),
        "presentation_preference": str(payload.get("presentation_preference", "any")),
        "maintenance_preference": str(payload.get("maintenance_preference", "any")),
        "forehead_goal": str(payload.get("forehead_goal", "auto")),
        "preferred_style_tag": str(payload.get("preferred_style_tag", "any")),
        "age_group": str(payload.get("age_group", "any")),
        "style_id": style_id,
        "label": int(label),
    }


def append_feedback_csv(csv_path: str | Path, row: dict[str, Any]) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key) for key in CSV_COLUMNS})
    return csv_path


def append_feedback_event(jsonl_path: str | Path, event: dict[str, Any]) -> Path:
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return jsonl_path
