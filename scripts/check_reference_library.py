from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hairstyle_tryon.analysis import analyze_photo
from hairstyle_tryon.quality import check_image_quality


ALLOWED_VIEW_HINTS = {"front", "left_oblique", "right_oblique"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the China-first hairstyle reference library.")
    parser.add_argument("--manifest", default="data/hairstyles/reference_library.cn.json")
    parser.add_argument("--report", default="outputs/reference_library/validation_report.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    passed = 0

    for entry in manifest.get("entries", []):
        prepared_path = Path(str(entry["prepared_path"]))
        quality = check_image_quality(prepared_path)
        analysis = analyze_photo(prepared_path)
        view_hint = str(analysis.pose["view_hint"])
        issues: list[str] = []
        if quality.is_monochrome:
            issues.append("reference looks monochrome")
        if quality.blur_score < 45:
            issues.append("reference is too blurry")
        if quality.width < 768 or quality.height < 768:
            issues.append("reference resolution is too low")
        if view_hint not in ALLOWED_VIEW_HINTS:
            issues.append(f"view hint is not acceptable for front try-on: {view_hint}")

        ok = len(issues) == 0
        if ok:
            passed += 1

        results.append(
            {
                "style_id": entry["style_id"],
                "prepared_path": str(prepared_path),
                "quality": asdict(quality),
                "analysis": asdict(analysis),
                "ok": ok,
                "issues": issues,
            }
        )

    payload = {
        "library_id": manifest.get("library_id"),
        "entries_checked": len(results),
        "entries_passed": passed,
        "entries": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
