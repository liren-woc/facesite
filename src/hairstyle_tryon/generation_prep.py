from __future__ import annotations

from pathlib import Path


def _load_points(image_path: Path) -> tuple[list[tuple[float, float]], int, int]:
    import cv2
    import mediapipe as mp

    from .analysis import _extract_points_with_legacy_api, _extract_points_with_tasks_api

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


def _compute_generation_crop(points: list[tuple[float, float]], width: int, height: int) -> tuple[int, int, int, int]:
    from .analysis import LANDMARKS, euclidean

    top = points[LANDMARKS["top"]]
    chin = points[LANDMARKS["chin"]]
    left_face = points[LANDMARKS["left_face"]]
    right_face = points[LANDMARKS["right_face"]]

    face_height = euclidean(top, chin)
    face_width = euclidean(left_face, right_face)
    face_center_x = (left_face[0] + right_face[0]) / 2.0

    crop_size = max(face_height * 2.18, face_width * 2.02)
    crop_top = top[1] - face_height * 0.84
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


def prepare_generation_face(
    source_path: str | Path,
    prepared_path: str | Path,
    *,
    output_size: int = 1024,
) -> Path:
    from PIL import Image

    source_path = Path(source_path)
    prepared_path = Path(prepared_path)
    points, width, height = _load_points(source_path)
    left, top, right, bottom = _compute_generation_crop(points, width, height)

    with Image.open(source_path) as image:
        prepared = image.convert("RGB").crop((left, top, right, bottom)).resize((output_size, output_size))
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        prepared.save(prepared_path, quality=95)

    return prepared_path
