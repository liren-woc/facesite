from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

from .analysis import FaceAnalysisResult, analyze_photo
from .backends.factory import create_generation_backend
from .backends.stable_hair import StableHairNotReady
from .feedback_store import append_feedback_event, append_feedback_csv, extract_feedback_row
from .quality import check_image_quality
from .recommend import recommend_styles
from .session_store import (
    append_session_index,
    create_session_dir,
    snapshot_inputs,
    write_session_profile,
)


REQUIRED_VIEWS = ("front", "left", "right", "hairline")
OPTIONAL_VIEWS = ("crown",)
ALL_VIEWS = REQUIRED_VIEWS + OPTIONAL_VIEWS
GENERATION_VIEW_MAP = {
    "front": "front.png",
    "left": "left.png",
    "right": "right.png",
}


def _normalize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return str(Path(path))


def _write_result_payload(output_dir: Path, payload: dict[str, Any]) -> Path:
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path


def _finalize_output(output_root: Path, output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    session = payload.setdefault("session", {})
    result_path = _write_result_payload(output_dir, payload)
    session["result_json"] = str(result_path)
    _write_result_payload(output_dir, payload)

    if session.get("session_id"):
        write_session_profile(output_dir, payload)
        append_session_index(
            output_root,
            {
                "session_id": session.get("session_id"),
                "session_label": session.get("session_label"),
                "session_dir": session.get("session_dir"),
                "result_json": session.get("result_json"),
                "presentation_preference": payload.get("presentation_preference"),
                "maintenance_preference": payload.get("maintenance_preference"),
                "forehead_goal": payload.get("forehead_goal"),
                "preferred_style_tag": payload.get("preferred_style_tag"),
                "generation_status": payload.get("generation", {}).get("status"),
                "selected_style_id": payload.get("generation", {}).get("selected_style_id"),
            },
        )
    return payload


def _empty_analysis_summary() -> dict[str, Any]:
    return {
        "dominant_face_shape_hint": "unknown",
        "dominant_hairline_height_hint": "unknown",
        "dominant_hairline_pattern_hint": "unknown",
        "dominant_recession_risk_hint": "unknown",
        "view_coverage": {view: False for view in ALL_VIEWS},
        "missing_required_views": list(REQUIRED_VIEWS),
        "recommended_next_uploads": list(REQUIRED_VIEWS),
        "same_person_verification": {
            "status": "insufficient_data",
            "score": None,
            "threshold": 0.82,
            "message": "Need at least two analyzed face images.",
        },
        "session_metrics": {},
        "images": [],
        "notes": [],
    }


def _verify_same_person(results: list[FaceAnalysisResult]) -> dict[str, Any]:
    if len(results) < 2:
        return {
            "status": "insufficient_data",
            "score": None,
            "threshold": 0.82,
            "message": "Need at least two analyzed face images.",
        }

    feature_names = [
        "face_ratio_h_w",
        "forehead_jaw_ratio",
        "cheek_jaw_ratio",
        "jaw_face_ratio",
        "upper_to_middle_ratio",
        "middle_to_lower_ratio",
        "forehead_to_face_ratio",
    ]
    signatures = [item.identity_signature for item in results]
    pair_scores: list[float] = []
    for index, left in enumerate(signatures):
        for right in signatures[index + 1 :]:
            distance = mean(abs(float(left[name]) - float(right[name])) for name in feature_names)
            yaw_penalty = min(abs(float(left["yaw_score"]) - float(right["yaw_score"])), 1.0) * 0.1
            score = max(0.0, 1.0 - distance - yaw_penalty)
            pair_scores.append(score)

    final_score = min(pair_scores) if pair_scores else 0.0
    same_person = final_score >= 0.82
    return {
        "status": "pass" if same_person else "review_needed",
        "score": round(final_score, 4),
        "threshold": 0.82,
        "message": (
            "Geometry across views looks consistent enough for the same person."
            if same_person
            else "Geometry differs noticeably across uploads. Ask the user to confirm or re-upload."
        ),
        "note": "This is a geometric consistency check, not a face-recognition model.",
    }


def _aggregate_session(analyses_by_view: dict[str, FaceAnalysisResult]) -> dict[str, Any]:
    summary = _empty_analysis_summary()
    if not analyses_by_view:
        summary["notes"].append("No images passed face analysis.")
        return summary

    results = list(analyses_by_view.values())
    face_shapes = [item.geometry.face_shape_hint for item in results]
    hairline_hints = [item.hairline.hairline_height_hint for item in results]
    hairline_patterns = [item.hairline.hairline_pattern_hint for item in results]
    recession_risks = [item.hairline.recession_risk_hint for item in results]
    summary["dominant_face_shape_hint"] = max(set(face_shapes), key=face_shapes.count)
    summary["dominant_hairline_height_hint"] = max(set(hairline_hints), key=hairline_hints.count)
    summary["dominant_hairline_pattern_hint"] = max(set(hairline_patterns), key=hairline_patterns.count)
    summary["dominant_recession_risk_hint"] = max(set(recession_risks), key=recession_risks.count)
    summary["view_coverage"] = {view: view in analyses_by_view for view in ALL_VIEWS}
    summary["missing_required_views"] = [view for view in REQUIRED_VIEWS if view not in analyses_by_view]
    summary["recommended_next_uploads"] = list(summary["missing_required_views"])
    summary["same_person_verification"] = _verify_same_person(results)
    summary["session_metrics"] = {
        "mean_face_ratio_h_w": round(mean(item.geometry.face_ratio_h_w for item in results), 4),
        "mean_forehead_to_face_ratio": round(mean(item.hairline.forehead_to_face_ratio for item in results), 4),
        "mean_center_skin_exposure": round(mean(item.hairline.center_skin_exposure for item in results), 4),
        "mean_temple_skin_exposure": round(mean(item.hairline.temple_skin_exposure for item in results), 4),
        "mean_forehead_jaw_ratio": round(mean(item.geometry.forehead_jaw_ratio for item in results), 4),
        "mean_cheek_jaw_ratio": round(mean(item.geometry.cheek_jaw_ratio for item in results), 4),
        "mean_jaw_face_ratio": round(mean(item.geometry.jaw_face_ratio for item in results), 4),
        "mean_upper_to_middle_ratio": round(mean(item.geometry.upper_to_middle_ratio for item in results), 4),
        "mean_middle_to_lower_ratio": round(mean(item.geometry.middle_to_lower_ratio for item in results), 4),
    }
    summary["images"] = [
        {
            "view": view,
            **asdict(result),
        }
        for view, result in analyses_by_view.items()
    ]
    summary["notes"] = [
        "Front/left/right/hairline inputs form the minimum recommended upload set.",
        "Hairline and same-person checks are still heuristic and should be upgraded with face parsing and identity embeddings.",
    ]
    return summary


def _build_recommendation_context(summary: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(summary.get("dominant_face_shape_hint", "unknown")),
        str(summary.get("dominant_hairline_height_hint", "unknown")),
        str(summary.get("dominant_hairline_pattern_hint", "unknown")),
        str(summary.get("dominant_recession_risk_hint", "unknown")),
    )


def _derive_effective_preferences(
    summary: dict[str, Any],
    *,
    presentation_preference: str,
    maintenance_preference: str,
    forehead_goal: str,
    preferred_style_tag: str,
    age_group: str,
) -> tuple[str, str, str, str, str]:
    hairline_pattern_hint = str(summary.get("dominant_hairline_pattern_hint", "unknown"))
    recession_risk_hint = str(summary.get("dominant_recession_risk_hint", "unknown"))

    effective_forehead_goal = forehead_goal
    if forehead_goal == "auto":
        if hairline_pattern_hint in {"open_forehead", "temple_recession"} or recession_risk_hint == "high":
            effective_forehead_goal = "cover"
        else:
            effective_forehead_goal = "balance"

    return (
        presentation_preference,
        maintenance_preference,
        effective_forehead_goal,
        preferred_style_tag,
        age_group,
    )


def _build_personal_profile(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("session_metrics", {})
    face_ratio = float(metrics.get("mean_face_ratio_h_w", 0.0) or 0.0)
    forehead_ratio = float(metrics.get("mean_forehead_to_face_ratio", 0.0) or 0.0)
    jaw_ratio = float(metrics.get("mean_jaw_face_ratio", 0.0) or 0.0)
    cheek_ratio = float(metrics.get("mean_cheek_jaw_ratio", 0.0) or 0.0)
    upper_to_middle = float(metrics.get("mean_upper_to_middle_ratio", 0.0) or 0.0)
    middle_to_lower = float(metrics.get("mean_middle_to_lower_ratio", 0.0) or 0.0)
    temple_skin = float(metrics.get("mean_temple_skin_exposure", 0.0) or 0.0)
    center_skin = float(metrics.get("mean_center_skin_exposure", 0.0) or 0.0)

    if face_ratio >= 1.24:
        face_length_profile = "long"
    elif face_ratio <= 1.11:
        face_length_profile = "short"
    else:
        face_length_profile = "balanced"

    if jaw_ratio >= 0.84:
        jaw_profile = "strong"
    elif jaw_ratio <= 0.76:
        jaw_profile = "narrow"
    else:
        jaw_profile = "balanced"

    if cheek_ratio >= 1.22:
        cheek_profile = "prominent"
    elif cheek_ratio <= 1.12:
        cheek_profile = "soft"
    else:
        cheek_profile = "balanced"

    if forehead_ratio >= 0.22 or center_skin >= 0.90:
        forehead_profile = "open"
    elif forehead_ratio <= 0.17:
        forehead_profile = "compact"
    else:
        forehead_profile = "balanced"

    if temple_skin >= 0.72:
        recession_profile = "strong"
    elif temple_skin >= 0.56:
        recession_profile = "medium"
    else:
        recession_profile = "low"

    if upper_to_middle >= 3.3:
        upper_third_profile = "tall"
    else:
        upper_third_profile = "balanced"

    if middle_to_lower <= 0.55:
        lower_third_profile = "long"
    else:
        lower_third_profile = "balanced"

    return {
        "face_ratio_h_w": face_ratio,
        "forehead_to_face_ratio": forehead_ratio,
        "jaw_face_ratio": jaw_ratio,
        "cheek_jaw_ratio": cheek_ratio,
        "upper_to_middle_ratio": upper_to_middle,
        "middle_to_lower_ratio": middle_to_lower,
        "temple_skin_exposure": temple_skin,
        "center_skin_exposure": center_skin,
        "face_length_profile": face_length_profile,
        "jaw_profile": jaw_profile,
        "cheek_profile": cheek_profile,
        "forehead_profile": forehead_profile,
        "recession_profile": recession_profile,
        "upper_third_profile": upper_third_profile,
        "lower_third_profile": lower_third_profile,
    }


def _resolve_catalog_reference(reference_path: str | Path | None) -> Path | None:
    if not reference_path:
        return None
    path = Path(reference_path)
    if path.exists():
        return path.resolve()
    project_root = Path(__file__).resolve().parents[2]
    candidate = (project_root / path).resolve()
    if candidate.exists():
        return candidate
    return None


def _resolve_generation_candidates(
    recommendations: list[dict[str, Any]],
    explicit_shape_path: str | Path | None,
    explicit_color_path: str | Path | None,
) -> list[dict[str, Any]]:
    shape_candidate = _resolve_catalog_reference(explicit_shape_path)
    color_candidate = _resolve_catalog_reference(explicit_color_path)

    if shape_candidate is not None:
        return [{
            "shape_path": shape_candidate,
            "color_path": color_candidate or shape_candidate,
            "style": None,
        }]

    candidates: list[dict[str, Any]] = []
    for item in recommendations:
        candidate = _resolve_catalog_reference(item.get("reference_image"))
        if candidate is None:
            continue
        candidates.append({
            "shape_path": candidate,
            "color_path": color_candidate or candidate,
            "style": item,
        })
    return candidates


def _validate_generated_front(
    source_analysis: FaceAnalysisResult | None,
    generated_path: str | Path,
) -> dict[str, Any]:
    if source_analysis is None:
        return {"status": "skipped", "message": "Source front analysis is unavailable."}

    try:
        generated_analysis = analyze_photo(generated_path)
    except Exception as exc:
        return {"status": "failed", "message": f"Generated image could not be analyzed: {exc}"}

    source_geometry = source_analysis.geometry
    generated_geometry = generated_analysis.geometry
    deltas = {
        "face_ratio_h_w": abs(generated_geometry.face_ratio_h_w - source_geometry.face_ratio_h_w),
        "jaw_face_ratio": abs(generated_geometry.jaw_face_ratio - source_geometry.jaw_face_ratio),
        "cheek_jaw_ratio": abs(generated_geometry.cheek_jaw_ratio - source_geometry.cheek_jaw_ratio),
        "forehead_jaw_ratio": abs(generated_geometry.forehead_jaw_ratio - source_geometry.forehead_jaw_ratio),
    }
    limits = {
        "face_ratio_h_w": 0.10,
        "jaw_face_ratio": 0.09,
        "cheek_jaw_ratio": 0.12,
        "forehead_jaw_ratio": 0.14,
    }
    violations = [
        f"{name} drifted by {value:.3f}"
        for name, value in deltas.items()
        if value > limits[name]
    ]
    if violations:
        return {
            "status": "failed",
            "message": "Generated silhouette drifted too far from the source face: " + "; ".join(violations),
            "deltas": deltas,
        }

    return {"status": "ok", "message": "Geometry stayed within the allowed range.", "deltas": deltas}


def _run_generation_candidates(
    *,
    backend,
    face_inputs: dict[str, str | None],
    output_dir: Path,
    candidates: list[dict[str, Any]],
    front_analysis: FaceAnalysisResult | None,
    generate_side_views: bool,
) -> dict[str, Any]:
    view_order = ("front", "left", "right") if generate_side_views else ("front",)
    attempt_logs: list[dict[str, Any]] = []

    for attempt_index, candidate in enumerate(candidates, start=1):
        style = candidate.get("style") or {}
        style_id = str(style.get("style_id") or style.get("id") or f"candidate_{attempt_index}")
        style_name = str(style.get("name") or style_id)
        generated_views: dict[str, str] = {}
        validation: dict[str, Any] | None = None
        try:
            for view in view_order:
                filename = GENERATION_VIEW_MAP[view]
                source_path = face_inputs.get(view)
                if not source_path:
                    continue
                final_result_path = output_dir / filename
                attempt_result_path = (
                    final_result_path
                    if len(candidates) == 1
                    else output_dir / f"{Path(filename).stem}__{style_id}.png"
                )
                generated = backend.run(
                    face_path=source_path,
                    shape_path=candidate["shape_path"],
                    color_path=candidate["color_path"],
                    result_path=attempt_result_path,
                )
                generated_views[view] = str(generated)

            if "front" in generated_views:
                validation = _validate_generated_front(front_analysis, generated_views["front"])
                if validation["status"] == "failed":
                    raise RuntimeError(validation["message"])

            finalized_views: dict[str, str] = {}
            for view, generated_path in generated_views.items():
                generated_path_obj = Path(generated_path).resolve()
                final_result_path = (output_dir / GENERATION_VIEW_MAP[view]).resolve()
                if generated_path_obj != final_result_path:
                    shutil.copy2(generated_path_obj, final_result_path)
                    generated_path_obj.unlink(missing_ok=True)
                finalized_views[view] = str(final_result_path)

            return {
                "selected_style_id": style.get("style_id") or style.get("id"),
                "selected_style_name": style.get("name"),
                "selected_reference_image": str(candidate["shape_path"]),
                "generated_views": finalized_views,
                "attempt_logs": attempt_logs,
                "validation": validation,
            }
        except Exception as exc:
            for generated_path in generated_views.values():
                Path(generated_path).unlink(missing_ok=True)
            attempt_logs.append(
                {
                    "attempt": attempt_index,
                    "style_id": style_id,
                    "style_name": style_name,
                    "reference_image": str(candidate["shape_path"]),
                    "error": str(exc),
                    "validation": validation,
                }
            )

    raise RuntimeError(
        "All vetted hairstyle candidates failed to generate a stable front result. "
        f"Attempts: {json.dumps(attempt_logs, ensure_ascii=False)}"
    )


def _analyze_image(view: str, image_path: str | Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "view": view,
        "image_path": str(image_path),
        "quality": None,
        "analysis": None,
    }
    try:
        payload["quality"] = asdict(check_image_quality(image_path))
    except Exception as exc:
        payload["quality"] = {"error": str(exc)}

    try:
        analysis = analyze_photo(image_path)
        payload["analysis"] = asdict(analysis)
        payload["detected_view_hint"] = analysis.pose["view_hint"]
        payload["_analysis_result"] = analysis
        return payload
    except Exception as exc:
        payload["analysis"] = {
            "error": str(exc),
            "notes": [
                "Analysis failed for this upload. The session can continue with the remaining images.",
            ],
        }
        return payload


def _front_generation_blockers(payload: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    front_upload = next((item for item in payload.get("uploads", []) if item.get("view") == "front"), None)
    if not front_upload:
        blockers.append("Front face image is required for generation.")
        return blockers

    quality = front_upload.get("quality") or {}
    if quality.get("is_monochrome"):
        blockers.append("Front image is monochrome. Use a natural color photo for realistic hair rendering.")

    width = quality.get("width")
    height = quality.get("height")
    if isinstance(width, (int, float)) and isinstance(height, (int, float)):
        if width < 640 or height < 640:
            blockers.append("Front image resolution is too low for realistic try-on. Use at least 640x640.")

    blur_score = quality.get("blur_score")
    if isinstance(blur_score, (int, float)) and blur_score < 45:
        blockers.append("Front image is too blurry for a believable hairstyle render.")

    return blockers


def run_pipeline(
    *,
    front_path: str | Path,
    left_path: str | Path | None = None,
    right_path: str | Path | None = None,
    hairline_path: str | Path | None = None,
    crown_path: str | Path | None = None,
    shape_path: str | Path | None = None,
    color_path: str | Path | None = None,
    catalog_path: str | Path = "data/hairstyles/catalog.example.json",
    generator_backend: str = "stable_hair",
    generator_repo: str | Path | None = None,
    generator_python: str | None = None,
    output_dir: str | Path = "outputs/tryon",
    skip_generation: bool = False,
    generate_side_views: bool = False,
    top_k: int = 5,
    presentation_preference: str = "any",
    maintenance_preference: str = "any",
    forehead_goal: str = "auto",
    preferred_style_tag: str = "any",
    age_group: str = "any",
    session_label: str | None = None,
    session_notes: str | None = None,
    enable_fallback: bool = True,
) -> dict[str, Any]:
    output_root = Path(output_dir)
    if session_label:
        session_info = create_session_dir(output_root, session_label)
        output_dir = session_info.session_dir
    else:
        session_info = None
        output_dir = output_root
        output_dir.mkdir(parents=True, exist_ok=True)

    face_inputs = {
        "front": _normalize_path(front_path),
        "left": _normalize_path(left_path),
        "right": _normalize_path(right_path),
        "hairline": _normalize_path(hairline_path),
        "crown": _normalize_path(crown_path),
    }
    extra_inputs = {
        "shape_reference": _normalize_path(shape_path),
        "color_reference": _normalize_path(color_path),
    }
    payload: dict[str, Any] = {
        "face_inputs": face_inputs,
        "presentation_preference": presentation_preference,
        "maintenance_preference": maintenance_preference,
        "forehead_goal": forehead_goal,
        "preferred_style_tag": preferred_style_tag,
        "age_group": age_group,
        "session": {
            "session_id": session_info.session_id if session_info else None,
            "session_label": session_info.label if session_info else session_label,
            "session_dir": str(output_dir.resolve()),
            "notes": session_notes or "",
        },
        "shape_path": _normalize_path(shape_path),
        "color_path": _normalize_path(color_path),
        "uploads": [],
        "analysis": _empty_analysis_summary(),
        "recommendations": [],
        "generation": {
            "backend": generator_backend,
            "status": "skipped" if skip_generation else "not_started",
            "generated_views": {},
            "message": None,
        },
    }
    payload["session"]["stored_inputs"] = snapshot_inputs(
        output_dir,
        face_inputs=face_inputs,
        extra_inputs=extra_inputs,
    )

    analyses_by_view: dict[str, FaceAnalysisResult] = {}
    for view, path in face_inputs.items():
        if not path:
            continue
        image_payload = _analyze_image(view, path)
        payload["uploads"].append(image_payload)
        analysis_result = image_payload.pop("_analysis_result", None)
        if analysis_result is not None:
            analyses_by_view[view] = analysis_result

    payload["analysis"] = _aggregate_session(analyses_by_view)

    for ref_name, ref_path in (("shape", shape_path), ("color", color_path)):
        if ref_path is None:
            continue
        try:
            payload["uploads"].append({
                "view": ref_name,
                "image_path": str(ref_path),
                "quality": asdict(check_image_quality(ref_path)),
                "analysis": None,
            })
        except Exception as exc:
            payload["uploads"].append({
                "view": ref_name,
                "image_path": str(ref_path),
                "quality": {"error": str(exc)},
                "analysis": None,
            })

    (
        face_shape_hint,
        hairline_hint,
        hairline_pattern_hint,
        recession_risk_hint,
    ) = _build_recommendation_context(payload["analysis"])
    (
        effective_presentation_preference,
        effective_maintenance_preference,
        effective_forehead_goal,
        effective_style_tag,
        effective_age_group,
    ) = _derive_effective_preferences(
        payload["analysis"],
        presentation_preference=presentation_preference,
        maintenance_preference=maintenance_preference,
        forehead_goal=forehead_goal,
        preferred_style_tag=preferred_style_tag,
        age_group=age_group,
    )
    payload["personal_profile"] = _build_personal_profile(payload["analysis"])
    payload["effective_preferences"] = {
        "presentation_preference": effective_presentation_preference,
        "maintenance_preference": effective_maintenance_preference,
        "forehead_goal": effective_forehead_goal,
        "preferred_style_tag": effective_style_tag,
        "age_group": effective_age_group,
    }
    try:
        recommendations = recommend_styles(
            face_shape_hint,
            catalog_path,
            hairline_height_hint=hairline_hint,
            hairline_pattern_hint=hairline_pattern_hint,
            recession_risk_hint=recession_risk_hint,
            personal_profile=payload["personal_profile"],
            top_k=top_k,
            age_group=effective_age_group,
            presentation_preference=effective_presentation_preference,
            maintenance_preference=effective_maintenance_preference,
            forehead_goal=effective_forehead_goal,
            preferred_style_tag=effective_style_tag,
        )
        payload["recommendations"] = [asdict(item) for item in recommendations]
    except Exception as exc:
        payload["recommendations_error"] = str(exc)

    if skip_generation:
        return _finalize_output(output_root, output_dir, payload)

    if generator_backend == "disabled":
        payload["generation"] = {
            "backend": generator_backend,
            "status": "skipped",
            "generated_views": {},
            "message": "Generation backend is disabled. Analysis and recommendation only.",
        }
        return _finalize_output(output_root, output_dir, payload)

    generation_candidates = _resolve_generation_candidates(
        payload.get("recommendations", []),
        shape_path,
        color_path,
    )
    if generation_candidates:
        first_style = generation_candidates[0].get("style")
        if first_style is not None:
            payload["generation"]["selected_style_id"] = first_style.get("style_id")
            payload["generation"]["selected_style_name"] = first_style.get("name")
            payload["generation"]["selected_reference_image"] = str(generation_candidates[0]["shape_path"])

    if not generation_candidates:
        payload["generation"] = {
            "backend": generator_backend,
            "status": "skipped",
            "generated_views": {},
            "message": "No vetted hairstyle reference is currently available. Generation is paused until the reference library is rebuilt.",
            "selected_style_id": payload["generation"].get("selected_style_id"),
            "selected_style_name": payload["generation"].get("selected_style_name"),
            "selected_reference_image": payload["generation"].get("selected_reference_image"),
        }
        return _finalize_output(output_root, output_dir, payload)

    front_input = face_inputs["front"]
    if not front_input:
        payload["generation"] = {
            "backend": generator_backend,
            "status": "failed",
            "generated_views": {},
            "message": "Front face image is required for generation.",
        }
        return _finalize_output(output_root, output_dir, payload)

    blockers = _front_generation_blockers(payload)
    if blockers:
        payload["generation"] = {
            "backend": generator_backend,
            "status": "skipped",
            "generated_views": {},
            "message": " ".join(blockers),
            "selected_style_id": payload["generation"].get("selected_style_id"),
            "selected_style_name": payload["generation"].get("selected_style_name"),
            "selected_reference_image": payload["generation"].get("selected_reference_image"),
        }
        return _finalize_output(output_root, output_dir, payload)

    try:
        normalized_backend = generator_backend.strip().lower()
        backend_repo = generator_repo or "third_party/Stable-Hair"
        backend_python = generator_python
        backend = create_generation_backend(
            generator_backend,
            repo_dir=Path(backend_repo),
            python_executable=backend_python,
        )
        generation_result = _run_generation_candidates(
            backend=backend,
            face_inputs=face_inputs,
            output_dir=output_dir,
            candidates=generation_candidates,
            front_analysis=analyses_by_view.get("front"),
            generate_side_views=generate_side_views,
        )

        payload["generation"] = {
            "backend": generator_backend,
            "status": "ok",
            "generated_views": generation_result["generated_views"],
            "message": (
                "Generation finished for the front portrait."
                if not generate_side_views
                else "Generation finished for all available views."
            ),
            "selected_style_id": generation_result.get("selected_style_id") or payload["generation"].get("selected_style_id"),
            "selected_style_name": generation_result.get("selected_style_name") or payload["generation"].get("selected_style_name"),
            "selected_reference_image": generation_result.get("selected_reference_image") or payload["generation"].get("selected_reference_image"),
            "attempt_logs": generation_result.get("attempt_logs", []),
            "validation": generation_result.get("validation"),
        }
    except Exception as exc:
        primary_status = "not_ready" if isinstance(exc, StableHairNotReady) else "failed"
        payload["generation"] = {
            "backend": generator_backend,
            "status": primary_status,
            "generated_views": {},
            "message": str(exc),
            "selected_style_id": payload["generation"].get("selected_style_id"),
            "selected_style_name": payload["generation"].get("selected_style_name"),
            "selected_reference_image": payload["generation"].get("selected_reference_image"),
        }

    return _finalize_output(output_root, output_dir, payload)


def save_feedback(
    *,
    payload: dict[str, Any],
    style_id: str,
    label: int,
    note: str = "",
    feedback_csv_path: str | Path = "outputs/recommender/user_feedback.csv",
    feedback_events_path: str | Path = "outputs/recommender/user_feedback_events.jsonl",
) -> dict[str, Any]:
    row = extract_feedback_row(payload, style_id=style_id, label=label)
    csv_path = append_feedback_csv(feedback_csv_path, row)
    event = {
        "session_id": payload.get("session", {}).get("session_id"),
        "session_dir": payload.get("session", {}).get("session_dir"),
        "style_id": style_id,
        "label": int(label),
        "note": note.strip(),
        "row": row,
    }
    event_path = append_feedback_event(feedback_events_path, event)
    return {
        "feedback_csv": str(csv_path),
        "feedback_events": str(event_path),
        "saved_style_id": style_id,
        "saved_label": int(label),
        "note": note.strip(),
    }


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hairstyle try-on pipeline.")
    parser.add_argument("--front", required=True, help="Front face photo.")
    parser.add_argument("--left", help="Left-side face photo.")
    parser.add_argument("--right", help="Right-side face photo.")
    parser.add_argument("--hairline", help="Hairline close-up photo.")
    parser.add_argument("--crown", help="Optional crown / back-top photo.")
    parser.add_argument("--shape", help="Hairstyle shape reference image.")
    parser.add_argument("--color", help="Optional hair color reference image.")
    parser.add_argument("--catalog", default="data/hairstyles/catalog.example.json")
    parser.add_argument("--generator-backend", default="stable_hair", choices=["stable_hair", "disabled"])
    parser.add_argument("--generator-repo", default=None)
    parser.add_argument("--generator-python", default=None)
    parser.add_argument("--output-dir", default="outputs/tryon")
    parser.add_argument("--output-json", default="outputs/tryon/result.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--presentation-preference", default="any", choices=["masculine", "feminine", "any"])
    parser.add_argument("--maintenance-preference", default="any", choices=["low", "medium", "high", "any"])
    parser.add_argument("--forehead-goal", default="auto", choices=["auto", "cover", "balance", "open"])
    parser.add_argument("--preferred-style-tag", default="any")
    parser.add_argument("--age-group", default="any", choices=["teen", "young_adult", "adult", "middle_aged", "senior", "any"])
    parser.add_argument("--session-label", default=None)
    parser.add_argument("--session-notes", default=None)
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    payload = run_pipeline(
        front_path=args.front,
        left_path=args.left,
        right_path=args.right,
        hairline_path=args.hairline,
        crown_path=args.crown,
        shape_path=args.shape,
        color_path=args.color,
        catalog_path=args.catalog,
        generator_backend=args.generator_backend,
        generator_repo=args.generator_repo,
        generator_python=args.generator_python,
        output_dir=args.output_dir,
        skip_generation=args.skip_generation,
        generate_side_views=False,
        top_k=args.top_k,
        presentation_preference=args.presentation_preference,
        maintenance_preference=args.maintenance_preference,
        forehead_goal=args.forehead_goal,
        preferred_style_tag=args.preferred_style_tag,
        age_group=args.age_group,
        session_label=args.session_label,
        session_notes=args.session_notes,
        enable_fallback=not args.disable_fallback,
    )
    output_json_path = args.output_json
    session_result_json = payload.get("session", {}).get("result_json")
    if args.output_json == "outputs/tryon/result.json" and session_result_json:
        output_json_path = session_result_json
    save_json(payload, output_json_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
