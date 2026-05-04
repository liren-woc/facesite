from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")


LANDMARKS = {
    "top": 10,
    "chin": 152,
    "left_face": 234,
    "right_face": 454,
    "nose_tip": 1,
    "upper_lip": 13,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_forehead": 70,
    "right_forehead": 300,
    "left_cheek": 93,
    "right_cheek": 323,
    "left_jaw": 172,
    "right_jaw": 397,
    "left_chin": 149,
    "right_chin": 378,
}

EYEBROW_LANDMARKS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]


@dataclass(frozen=True)
class FaceGeometry:
    face_height: float
    face_width: float
    forehead_width: float
    cheekbone_width: float
    jaw_width: float
    chin_width: float
    upper_third_height: float
    middle_third_height: float
    lower_third_height: float
    face_ratio_h_w: float
    forehead_jaw_ratio: float
    cheek_jaw_ratio: float
    jaw_face_ratio: float
    upper_to_middle_ratio: float
    middle_to_lower_ratio: float
    face_shape_hint: str


@dataclass(frozen=True)
class HairlineEstimate:
    forehead_height: float
    forehead_to_face_ratio: float
    hairline_height_hint: str
    hairline_pattern_hint: str
    recession_risk_hint: str
    center_skin_exposure: float
    temple_skin_exposure: float
    confidence: str
    notes: list[str]


@dataclass(frozen=True)
class FaceAnalysisResult:
    image_path: str
    geometry: FaceGeometry
    hairline: HairlineEstimate
    pose: dict[str, float | str]
    identity_signature: dict[str, float]
    notes: list[str]


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def safe_div(a: float, b: float, eps: float = 1e-8) -> float:
    return float(a) / (float(b) + eps)


def _silence_mediapipe_logs() -> None:
    try:
        from absl import logging as absl_logging

        absl_logging.set_verbosity(absl_logging.ERROR)
        absl_logging.set_stderrthreshold("error")
    except Exception:
        return


def compute_geometry(points: list[tuple[float, float]]) -> FaceGeometry:
    required = max(LANDMARKS.values())
    if len(points) <= required:
        raise ValueError(f"Need at least {required + 1} face landmarks, got {len(points)}.")

    p = {name: points[index] for name, index in LANDMARKS.items()}

    face_height = euclidean(p["top"], p["chin"])
    face_width = euclidean(p["left_face"], p["right_face"])
    forehead_width = euclidean(p["left_forehead"], p["right_forehead"])
    cheekbone_width = euclidean(p["left_cheek"], p["right_cheek"])
    jaw_width = euclidean(p["left_jaw"], p["right_jaw"])
    chin_width = euclidean(p["left_chin"], p["right_chin"])
    upper_third_height = euclidean(p["top"], p["nose_tip"])
    middle_third_height = euclidean(p["nose_tip"], p["upper_lip"])
    lower_third_height = euclidean(p["upper_lip"], p["chin"])

    face_ratio_h_w = safe_div(face_height, face_width)
    forehead_jaw_ratio = safe_div(forehead_width, jaw_width)
    cheek_jaw_ratio = safe_div(cheekbone_width, jaw_width)
    jaw_face_ratio = safe_div(jaw_width, face_width)

    return FaceGeometry(
        face_height=face_height,
        face_width=face_width,
        forehead_width=forehead_width,
        cheekbone_width=cheekbone_width,
        jaw_width=jaw_width,
        chin_width=chin_width,
        upper_third_height=upper_third_height,
        middle_third_height=middle_third_height,
        lower_third_height=lower_third_height,
        face_ratio_h_w=face_ratio_h_w,
        forehead_jaw_ratio=forehead_jaw_ratio,
        cheek_jaw_ratio=cheek_jaw_ratio,
        jaw_face_ratio=jaw_face_ratio,
        upper_to_middle_ratio=safe_div(upper_third_height, middle_third_height),
        middle_to_lower_ratio=safe_div(middle_third_height, lower_third_height),
        face_shape_hint=infer_face_shape(
            face_ratio_h_w=face_ratio_h_w,
            forehead_jaw_ratio=forehead_jaw_ratio,
            cheek_jaw_ratio=cheek_jaw_ratio,
            jaw_face_ratio=jaw_face_ratio,
        ),
    )


def estimate_pose(points: list[tuple[float, float]]) -> dict[str, float | str]:
    required = max(LANDMARKS.values())
    if len(points) <= required:
        raise ValueError(f"Need at least {required + 1} face landmarks, got {len(points)}.")

    left_face = points[LANDMARKS["left_face"]]
    right_face = points[LANDMARKS["right_face"]]
    nose_tip = points[LANDMARKS["nose_tip"]]
    left_eye = points[LANDMARKS["left_eye_outer"]]
    right_eye = points[LANDMARKS["right_eye_outer"]]

    face_width = euclidean(left_face, right_face)
    face_center_x = (left_face[0] + right_face[0]) / 2.0
    yaw_score = safe_div(nose_tip[0] - face_center_x, face_width / 2.0)
    eye_line_roll = safe_div(right_eye[1] - left_eye[1], euclidean(left_eye, right_eye))

    abs_yaw = abs(yaw_score)
    if abs_yaw <= 0.08:
        view = "front"
    elif yaw_score < -0.28:
        view = "left_profile"
    elif yaw_score < -0.08:
        view = "left_oblique"
    elif yaw_score > 0.28:
        view = "right_profile"
    else:
        view = "right_oblique"

    return {
        "yaw_score": float(yaw_score),
        "roll_score": float(eye_line_roll),
        "view_hint": view,
    }


def build_identity_signature(geometry: FaceGeometry, hairline: HairlineEstimate, pose: dict[str, float | str]) -> dict[str, float]:
    return {
        "face_ratio_h_w": geometry.face_ratio_h_w,
        "forehead_jaw_ratio": geometry.forehead_jaw_ratio,
        "cheek_jaw_ratio": geometry.cheek_jaw_ratio,
        "jaw_face_ratio": geometry.jaw_face_ratio,
        "upper_to_middle_ratio": geometry.upper_to_middle_ratio,
        "middle_to_lower_ratio": geometry.middle_to_lower_ratio,
        "forehead_to_face_ratio": hairline.forehead_to_face_ratio,
        "center_skin_exposure": hairline.center_skin_exposure,
        "temple_skin_exposure": hairline.temple_skin_exposure,
        "yaw_score": float(pose["yaw_score"]),
    }


def _skin_ratio(image_rgb, region: tuple[int, int, int, int]) -> float:
    import cv2
    import numpy as np

    x1, y1, x2, y2 = region
    patch = image_rgb[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if patch.size == 0:
        return 0.0
    ycrcb = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_RGB2YCrCb)
    mask = (
        (ycrcb[:, :, 1] >= 133)
        & (ycrcb[:, :, 1] <= 173)
        & (ycrcb[:, :, 2] >= 77)
        & (ycrcb[:, :, 2] <= 127)
    )
    return float(mask.mean())


def _infer_hairline_pattern(
    points: list[tuple[float, float]],
    image_rgb,
    geometry: FaceGeometry,
) -> tuple[str, str, float, float]:
    left_face = points[LANDMARKS["left_face"]]
    right_face = points[LANDMARKS["right_face"]]
    left_forehead = points[LANDMARKS["left_forehead"]]
    right_forehead = points[LANDMARKS["right_forehead"]]
    top = points[LANDMARKS["top"]]

    face_center_x = (left_face[0] + right_face[0]) / 2.0
    face_width = geometry.face_width
    face_height = geometry.face_height
    eyebrow_y = mean(points[index][1] for index in EYEBROW_LANDMARKS)

    center_region = (
        int(face_center_x - face_width * 0.10),
        int(top[1] - face_height * 0.06),
        int(face_center_x + face_width * 0.10),
        int(top[1] + face_height * 0.14),
    )
    left_region = (
        int(left_forehead[0] - face_width * 0.12),
        int(top[1] - face_height * 0.03),
        int(left_forehead[0] + face_width * 0.02),
        int(top[1] + face_height * 0.18),
    )
    right_region = (
        int(right_forehead[0] - face_width * 0.02),
        int(top[1] - face_height * 0.03),
        int(right_forehead[0] + face_width * 0.12),
        int(top[1] + face_height * 0.18),
    )

    center_skin = _skin_ratio(image_rgb, center_region)
    left_skin = _skin_ratio(image_rgb, left_region)
    right_skin = _skin_ratio(image_rgb, right_region)
    temple_skin = (left_skin + right_skin) / 2.0

    if center_skin >= 0.92 and temple_skin >= 0.50:
        return "open_forehead", "high", center_skin, temple_skin
    if temple_skin >= 0.78 and temple_skin >= center_skin - 0.08:
        return "temple_recession", "high", center_skin, temple_skin
    if temple_skin >= 0.62:
        return "temple_recession", "medium", center_skin, temple_skin
    if center_skin >= 0.88 and eyebrow_y - top[1] >= face_height * 0.18:
        return "open_forehead", "medium", center_skin, temple_skin
    return "balanced", "low", center_skin, temple_skin


def compute_hairline_estimate(points: list[tuple[float, float]], geometry: FaceGeometry, image_rgb) -> HairlineEstimate:
    """Estimate visible forehead / hairline height from landmarks.

    MediaPipe face landmarks do not trace the real hair boundary. This proxy uses
    the top forehead landmark and eyebrow landmarks, so it is useful for ranking
    hairstyles but should be replaced by face parsing for precise hairline work.
    """
    required = max(max(LANDMARKS.values()), max(EYEBROW_LANDMARKS))
    if len(points) <= required:
        raise ValueError(f"Need at least {required + 1} face landmarks, got {len(points)}.")

    top_y = points[LANDMARKS["top"]][1]
    eyebrow_y = mean(points[index][1] for index in EYEBROW_LANDMARKS)
    forehead_height = max(0.0, eyebrow_y - top_y)
    ratio = safe_div(forehead_height, geometry.face_height)

    if ratio >= 0.30:
        hint = "high"
    elif ratio <= 0.18:
        hint = "low"
    else:
        hint = "balanced"

    pattern_hint, recession_risk, center_skin, temple_skin = _infer_hairline_pattern(points, image_rgb, geometry)

    if hint == "balanced" and pattern_hint in {"open_forehead", "temple_recession"}:
        hint = "high"

    return HairlineEstimate(
        forehead_height=forehead_height,
        forehead_to_face_ratio=ratio,
        hairline_height_hint=hint,
        hairline_pattern_hint=pattern_hint,
        recession_risk_hint=recession_risk,
        center_skin_exposure=center_skin,
        temple_skin_exposure=temple_skin,
        confidence="low_to_medium",
        notes=[
            "This is a 2D landmark proxy, not a real hair boundary measurement.",
            "Temple/center exposure is a lightweight image heuristic for open forehead or recession risk.",
            "Use a face parsing hair mask for a precise hairline estimate.",
        ],
    )


def infer_face_shape(
    *,
    face_ratio_h_w: float,
    forehead_jaw_ratio: float,
    cheek_jaw_ratio: float,
    jaw_face_ratio: float,
) -> str:
    """A heuristic hint, not a medical or beauty judgment."""
    if face_ratio_h_w >= 1.35:
        return "oblong"
    if face_ratio_h_w <= 1.12 and jaw_face_ratio >= 0.82:
        return "round"
    if jaw_face_ratio >= 0.86 and abs(forehead_jaw_ratio - 1.0) < 0.12:
        return "square"
    if forehead_jaw_ratio >= 1.12 and cheek_jaw_ratio >= 1.12:
        return "heart"
    return "oval"


def analyze_photo(image_path: str | Path) -> FaceAnalysisResult:
    """Analyze one photo with MediaPipe Face Mesh.

    The heavy dependencies are imported inside the function so the project can
    still be inspected on machines without the ML environment installed.
    """
    _silence_mediapipe_logs()
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "Missing analysis dependencies. Install requirements-core.txt in a Python 3.10 environment."
        ) from exc

    image_path = Path(image_path)
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

    geometry = compute_geometry(points)
    hairline = compute_hairline_estimate(points, geometry, image_rgb)
    pose = estimate_pose(points)
    identity_signature = build_identity_signature(geometry, hairline, pose)

    notes = [
        "face_shape_hint is a heuristic from 2D landmarks.",
        "hairline_height_hint is a forehead proxy; precise hairline analysis requires face parsing.",
        "hairline_pattern_hint is a lightweight exposure heuristic for open forehead / temple recession risk.",
        "view_hint is a rough 2D yaw estimate from landmarks.",
        "identity_signature is only a lightweight geometric consistency signature, not face recognition.",
    ]
    return FaceAnalysisResult(str(image_path), geometry, hairline, pose, identity_signature, notes)


def _default_task_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "face_landmarker.task"


def _extract_points_with_tasks_api(image_rgb, image_path: Path) -> list[tuple[float, float]] | None:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = Path(os.environ.get("FACE_LANDMARKER_TASK", _default_task_model_path()))
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe task model not found: {model_path}. "
            "Run scripts/bootstrap_mediapipe.sh or set FACE_LANDMARKER_TASK."
        )

    h, w = image_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    return [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]


def _extract_points_with_legacy_api(image_rgb) -> list[tuple[float, float]] | None:
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        return None

    h, w = image_rgb.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(image_rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    return [(lm.x * w, lm.y * h) for lm in landmarks]


def analyze_session(image_paths: Iterable[str | Path]) -> dict:
    results = [analyze_photo(path) for path in image_paths]
    if not results:
        raise ValueError("At least one image is required.")

    face_shapes = [item.geometry.face_shape_hint for item in results]
    dominant_shape = max(set(face_shapes), key=face_shapes.count)
    ratios = [item.geometry.face_ratio_h_w for item in results]
    hairline_hints = [item.hairline.hairline_height_hint for item in results]
    dominant_hairline = max(set(hairline_hints), key=hairline_hints.count)
    view_hints = [str(item.pose["view_hint"]) for item in results]

    return {
        "dominant_face_shape_hint": dominant_shape,
        "dominant_hairline_height_hint": dominant_hairline,
        "mean_face_ratio_h_w": mean(ratios),
        "mean_forehead_to_face_ratio": mean(item.hairline.forehead_to_face_ratio for item in results),
        "detected_view_hints": view_hints,
        "images": [asdict(item) for item in results],
        "notes": [
            "For production use, add face parsing for hairline and DECA/3DDFA for 3D head geometry.",
        ],
    }


def save_json(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze face geometry from one or more photos.")
    parser.add_argument("images", nargs="+", help="Input face photos.")
    parser.add_argument("--output", default="outputs/tryon/analysis.json", help="Output JSON path.")
    args = parser.parse_args()

    payload = analyze_session(args.images)
    save_json(payload, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
