from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hairstyle_tryon.analysis import analyze_photo
from hairstyle_tryon.generation_prep import prepare_generation_face
from hairstyle_tryon.quality import check_image_quality
from hairstyle_tryon.recommend import load_catalog


ALLOWED_VIEW_HINTS = {"front", "left_oblique", "right_oblique"}


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _load_pool_entries(manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    return {str(entry["slug"]): entry for entry in entries}


def _validate_reference(prepared_path: Path, *, min_blur_score: float) -> dict[str, Any]:
    quality = check_image_quality(prepared_path)
    analysis = analyze_photo(prepared_path)
    view_hint = str(analysis.pose["view_hint"])
    issues: list[str] = []
    if quality.is_monochrome:
        issues.append("reference looks monochrome")
    if quality.blur_score < min_blur_score:
        issues.append(f"reference is too blurry (< {min_blur_score})")
    if quality.width < 768 or quality.height < 768:
        issues.append("reference resolution is too low")
    if view_hint not in ALLOWED_VIEW_HINTS:
        issues.append(f"view hint is not acceptable for front try-on: {view_hint}")
    return {
        "quality": asdict(quality),
        "analysis": asdict(analysis),
        "ok": len(issues) == 0,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote reference-pool images into the active generation library.")
    parser.add_argument("--config", default="configs/reference_pool_activation.json")
    parser.add_argument("--catalog", default="data/hairstyles/catalog.example.json")
    parser.add_argument("--library-manifest", default="data/hairstyles/reference_library.cn.json")
    parser.add_argument("--report", default="outputs/reference_library/sync_report.json")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    catalog_path = _resolve(args.catalog)
    library_manifest_path = _resolve(args.library_manifest)
    report_path = _resolve(args.report)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    activations = config.get("activations", [])
    catalog = load_catalog(catalog_path)
    catalog_by_id = {str(item["id"]): item for item in catalog}

    pool_cache: dict[Path, dict[str, dict[str, Any]]] = {}
    report_entries: list[dict[str, Any]] = []
    library_entries: list[dict[str, Any]] = []

    for activation in activations:
        source_manifest_path = _resolve(activation["source_manifest"])
        if source_manifest_path not in pool_cache:
            pool_cache[source_manifest_path] = _load_pool_entries(source_manifest_path)
        pool_entries = pool_cache[source_manifest_path]

        source_slug = str(activation["source_slug"])
        style_id = str(activation["style_id"])
        display_name = str(activation.get("display_name", style_id))
        min_blur_score = float(activation.get("min_blur_score", 45.0))

        catalog_item = catalog_by_id.get(style_id)
        if catalog_item is None:
            report_entries.append(
                {
                    "style_id": style_id,
                    "source_slug": source_slug,
                    "ok": False,
                    "issues": [f"catalog style_id not found: {style_id}"],
                }
            )
            continue

        source_entry = pool_entries.get(source_slug)
        if source_entry is None:
            report_entries.append(
                {
                    "style_id": style_id,
                    "source_slug": source_slug,
                    "ok": False,
                    "issues": [f"source slug not found in pool manifest: {source_slug}"],
                }
            )
            continue

        source_path = _resolve(source_entry["local_path"])
        style_dir = _resolve(Path("data/hairstyles/references") / style_id)
        style_dir.mkdir(parents=True, exist_ok=True)
        source_suffix = source_path.suffix.lower() or ".jpg"
        raw_target_path = style_dir / f"front{source_suffix}"
        prepared_path = style_dir / "prepared.jpg"

        shutil.copy2(source_path, raw_target_path)
        prepare_generation_face(raw_target_path, prepared_path, output_size=1024)
        validation = _validate_reference(prepared_path, min_blur_score=min_blur_score)

        ok = bool(validation["ok"])
        if ok:
            catalog_item["reference_image"] = str(prepared_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            catalog_item["status"] = "vetted_reference"
            library_entries.append(
                {
                    "style_id": style_id,
                    "name": display_name,
                    "status": "vetted_v1",
                    "source_manifest": str(Path(activation["source_manifest"])).replace("\\", "/"),
                    "source_slug": source_slug,
                    "source_path": str(raw_target_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "prepared_path": str(prepared_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "review_notes": [
                        "Promoted from reference_pool into generation-ready library.",
                        f"Source credit: {source_entry.get('source_credit', '')}".strip(),
                    ],
                }
            )
        else:
            catalog_item["reference_image"] = ""
            catalog_item["status"] = "needs_vetted_reference"

        report_entries.append(
            {
                "style_id": style_id,
                "display_name": display_name,
                "source_slug": source_slug,
                "source_path": str(raw_target_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "prepared_path": str(prepared_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "min_blur_score": min_blur_score,
                **validation,
            }
        )

    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    library_manifest_payload = {
        "library_id": config.get("library_id", "china_first_reference_v2"),
        "title": config.get("title", "中国审美优先参考库"),
        "description": config.get("description", ""),
        "source_policy": config.get("source_policy", {}),
        "entries": library_entries,
    }
    library_manifest_path.write_text(
        json.dumps(library_manifest_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_payload = {
        "config": str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "entries_checked": len(report_entries),
        "entries_passed": sum(1 for entry in report_entries if entry.get("ok")),
        "entries": report_entries,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
