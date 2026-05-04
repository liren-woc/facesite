from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_points(image_path: Path) -> tuple[list[tuple[float, float]], int, int]:
    import cv2
    import mediapipe as mp

    from hairstyle_tryon.analysis import _extract_points_with_legacy_api, _extract_points_with_tasks_api

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if hasattr(mp, "solutions"):
        points = _extract_points_with_legacy_api(image_rgb)
    else:
        points = _extract_points_with_tasks_api(image_rgb, image_path)
    if points is None:
        raise RuntimeError(f"No face detected in: {image_path}")
    height, width = image_rgb.shape[:2]
    return points, width, height


def _compute_square_crop(points: list[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    from hairstyle_tryon.analysis import LANDMARKS, euclidean

    top = points[LANDMARKS["top"]]
    chin = points[LANDMARKS["chin"]]
    left_face = points[LANDMARKS["left_face"]]
    right_face = points[LANDMARKS["right_face"]]

    face_height = euclidean(top, chin)
    face_width = euclidean(left_face, right_face)
    face_center_x = (left_face[0] + right_face[0]) / 2.0
    crop_size = max(face_height * 2.35, face_width * 2.15)
    crop_top = top[1] - face_height * 0.9
    crop_left = face_center_x - crop_size / 2.0
    crop_right = crop_left + crop_size
    crop_bottom = crop_top + crop_size

    if crop_left < 0:
        crop_right -= crop_left
        crop_left = 0
    if crop_top < 0:
        crop_bottom -= crop_top
        crop_top = 0
    if crop_right > width:
        shift = crop_right - width
        crop_left = max(0, crop_left - shift)
        crop_right = width
    if crop_bottom > height:
        shift = crop_bottom - height
        crop_top = max(0, crop_top - shift)
        crop_bottom = height

    return (
        int(round(crop_left)),
        int(round(crop_top)),
        int(round(crop_right)),
        int(round(crop_bottom)),
    )


def prepare_entry(source_path: Path, prepared_path: Path, output_size: int) -> dict[str, object]:
    from PIL import Image

    points, width, height = _load_points(source_path)
    left, top, right, bottom = _compute_square_crop(points, width, height)

    with Image.open(source_path) as image:
        prepared = image.convert("RGB").crop((left, top, right, bottom)).resize((output_size, output_size))
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.save(prepared_path, quality=95)

    return {
        "source_path": str(source_path),
        "prepared_path": str(prepared_path),
        "crop_box": {"left": left, "top": top, "right": right, "bottom": bottom},
        "output_size": output_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a curated hairstyle reference library.")
    parser.add_argument(
        "--manifest",
        default="data/hairstyles/reference_library.cn.json",
        help="Reference manifest JSON path.",
    )
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument(
        "--report",
        default="outputs/reference_library/prepare_report.json",
        help="JSON report output path.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    results: list[dict[str, object]] = []
    for entry in entries:
        source_path = Path(str(entry["source_path"]))
        prepared_path = Path(str(entry["prepared_path"]))
        result = prepare_entry(source_path, prepared_path, args.output_size)
        result["style_id"] = str(entry["style_id"])
        results.append(result)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"entries": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"entries": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
