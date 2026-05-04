from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_REGIONS = {"cn", "kr"}
ALLOWED_PRESENTATIONS = {"masculine", "feminine", "any"}
ALLOWED_MAINTENANCE = {"low", "medium", "high", "any"}
ALLOWED_FOREHEAD_GOALS = {"auto", "cover", "balance", "open"}
ALLOWED_AGE_GROUPS = {"teen", "young_adult", "adult", "middle_aged", "senior", "any"}
COMMON_STYLE_TAGS = {
    "clean",
    "natural",
    "workplace",
    "stable",
    "rejuvenating",
    "soft",
    "student",
    "fresh",
    "mature",
    "cute",
    "mainstream",
    "korean",
}
MATURE_STYLE_TAGS = {"clean", "natural", "workplace", "stable", "mature"}
YOUTHFUL_STYLE_TAGS = {"student", "cute", "rejuvenating", "fresh"}


@dataclass(frozen=True)
class StyleScore:
    style_id: str
    name: str
    score: float
    reference_image: str
    presentation: str
    maintenance_level: str
    style_tags: list[str]
    reasons: list[str]


def load_catalog(catalog_path: str | Path) -> list[dict[str, Any]]:
    with Path(catalog_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Catalog must be a JSON list.")
    return data


def _as_set(style: dict[str, Any], key: str) -> set[str]:
    values = style.get(key, [])
    if isinstance(values, list):
        return {str(item) for item in values}
    if values is None:
        return set()
    return {str(values)}


def _is_allowed_region(style: dict[str, Any]) -> bool:
    regions = _as_set(style, "aesthetic_regions")
    return not regions or bool(regions & ALLOWED_REGIONS)


def _normalize_presentation(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in ALLOWED_PRESENTATIONS else "any"


def _normalize_maintenance(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in ALLOWED_MAINTENANCE else "any"


def _normalize_forehead_goal(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in ALLOWED_FOREHEAD_GOALS else "auto"


def _normalize_age_group(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in ALLOWED_AGE_GROUPS else "any"


def _normalize_style_tag(value: str) -> str:
    normalized = str(value).strip().lower()
    return normalized if normalized in COMMON_STYLE_TAGS else "any"


def _matches_presentation(style: dict[str, Any], presentation_preference: str) -> bool:
    presentation = str(style.get("presentation", "any")).strip().lower()
    if presentation_preference == "any":
        return True
    return presentation in {presentation_preference, "any", "unisex"}


def _score_target_metric(
    *,
    score: float,
    reasons: list[str],
    current: float | None,
    target: float | None,
    tolerance: float | None,
    weight: float,
    label: str,
) -> float:
    if current is None or target is None or tolerance is None or tolerance <= 0:
        return score
    distance = abs(current - target)
    if distance <= tolerance:
        closeness = max(0.0, 1.0 - distance / tolerance)
        score += weight * closeness
        reasons.append(f"{label} stays close to this person's measured profile")
    else:
        penalty = min(distance / tolerance - 1.0, 1.4) * (weight * 0.65)
        score -= penalty
        reasons.append(f"{label} drifts away from this person's measured profile")
    return score


def _score_personal_profile(
    style: dict[str, Any],
    personal_profile: dict[str, Any] | None,
    score: float,
    reasons: list[str],
) -> float:
    if not personal_profile:
        return score

    ideal = style.get("ideal_profile")
    if not isinstance(ideal, dict):
        return score

    score = _score_target_metric(
        score=score,
        reasons=reasons,
        current=personal_profile.get("face_ratio_h_w"),
        target=ideal.get("face_ratio_h_w"),
        tolerance=ideal.get("face_ratio_tolerance"),
        weight=1.35,
        label="face length-width balance",
    )
    score = _score_target_metric(
        score=score,
        reasons=reasons,
        current=personal_profile.get("forehead_to_face_ratio"),
        target=ideal.get("forehead_to_face_ratio"),
        tolerance=ideal.get("forehead_ratio_tolerance"),
        weight=1.15,
        label="forehead openness",
    )
    score = _score_target_metric(
        score=score,
        reasons=reasons,
        current=personal_profile.get("jaw_face_ratio"),
        target=ideal.get("jaw_face_ratio"),
        tolerance=ideal.get("jaw_face_ratio_tolerance"),
        weight=1.05,
        label="jaw-to-face proportion",
    )
    score = _score_target_metric(
        score=score,
        reasons=reasons,
        current=personal_profile.get("cheek_jaw_ratio"),
        target=ideal.get("cheek_jaw_ratio"),
        tolerance=ideal.get("cheek_jaw_ratio_tolerance"),
        weight=0.95,
        label="cheekbone-to-jaw balance",
    )
    score = _score_target_metric(
        score=score,
        reasons=reasons,
        current=personal_profile.get("temple_skin_exposure"),
        target=ideal.get("temple_skin_exposure"),
        tolerance=ideal.get("temple_skin_tolerance"),
        weight=1.4,
        label="temple recession handling",
    )

    preferred_profiles = style.get("preferred_personal_profiles", {})
    if isinstance(preferred_profiles, dict):
        for key in (
            "face_length_profile",
            "jaw_profile",
            "cheek_profile",
            "forehead_profile",
            "recession_profile",
            "upper_third_profile",
            "lower_third_profile",
        ):
            expected = preferred_profiles.get(key)
            current = personal_profile.get(key)
            if not expected or current is None:
                continue
            values = {str(item) for item in expected} if isinstance(expected, list) else {str(expected)}
            if str(current) in values:
                score += 0.55
                reasons.append(f"{key} matches the person's scanned profile")
            else:
                score -= 0.35
                reasons.append(f"{key} is less aligned with the person's scanned profile")

    return score


def _score_age_group(
    style: dict[str, Any],
    age_group: str,
    score: float,
    reasons: list[str],
) -> float:
    age_group = _normalize_age_group(age_group)
    if age_group == "any":
        return score

    style_tags = _as_set(style, "style_tags")
    preferred_age_groups = _as_set(style, "preferred_age_groups")
    avoid_age_groups = _as_set(style, "avoid_age_groups")

    if preferred_age_groups:
        if age_group in preferred_age_groups or "any" in preferred_age_groups:
            score += 1.75
            reasons.append(f"catalog marks it as a better fit for the {age_group} life stage")
        else:
            score -= 1.15
            reasons.append(f"catalog does not list it as a primary fit for the {age_group} life stage")

    if age_group in avoid_age_groups:
        score -= 4.2
        reasons.append(f"catalog flags it as a weak fit for the {age_group} life stage")

    if age_group == "teen":
        if style_tags & {"student", "fresh", "rejuvenating", "mainstream", "korean"}:
            score += 0.95
            reasons.append("keeps a younger, more current school-age vibe")
        if style_tags & {"mature", "stable"} and "natural" not in style_tags:
            score -= 0.45
            reasons.append("reads slightly older than a teen-focused direction")
    elif age_group == "young_adult":
        if style_tags & {"mainstream", "rejuvenating", "fresh", "korean"}:
            score += 0.85
            reasons.append("stays closer to a current young-adult trend direction")
        if style_tags & {"student", "cute"}:
            score += 0.25
            reasons.append("still carries a younger feel that can work at this stage")
    elif age_group == "adult":
        if style_tags & {"clean", "natural", "mainstream", "workplace"}:
            score += 0.7
            reasons.append("balances trend with an adult daily-wear impression")
        if style_tags & {"student", "cute"}:
            score -= 0.5
            reasons.append("leans slightly too juvenile for an adult profile")
    elif age_group == "middle_aged":
        if style_tags & MATURE_STYLE_TAGS:
            score += 1.25
            reasons.append("keeps the silhouette steadier for a middle-aged profile")
        if style_tags & {"student", "cute"}:
            score -= 2.2
            reasons.append("looks too juvenile for a middle-aged profile")
        if "rejuvenating" in style_tags and not (style_tags & {"clean", "natural", "workplace"}):
            score -= 0.9
            reasons.append("tries too hard to look younger instead of staying natural")
    elif age_group == "senior":
        if style_tags & MATURE_STYLE_TAGS:
            score += 1.55
            reasons.append("keeps a calmer and more age-appropriate balance for a senior profile")
        if style_tags & {"student", "cute"}:
            score -= 3.0
            reasons.append("reads far too young for a senior profile")
        if style_tags & {"rejuvenating", "korean"} and not (style_tags & {"clean", "natural", "mature", "stable", "workplace"}):
            score -= 1.9
            reasons.append("leans too youthful instead of staying grounded for a senior profile")

    return score


def score_style(
    style: dict[str, Any],
    face_shape_hint: str,
    hairline_height_hint: str = "unknown",
    hairline_pattern_hint: str = "unknown",
    recession_risk_hint: str = "unknown",
    personal_profile: dict[str, Any] | None = None,
    age_group: str = "any",
    presentation_preference: str = "any",
    maintenance_preference: str = "any",
    forehead_goal: str = "auto",
    preferred_style_tag: str = "any",
) -> StyleScore:
    score = 0.0
    reasons: list[str] = []
    presentation = str(style.get("presentation", "any"))
    presentation_preference = _normalize_presentation(presentation_preference)
    maintenance_preference = _normalize_maintenance(maintenance_preference)
    forehead_goal = _normalize_forehead_goal(forehead_goal)
    age_group = _normalize_age_group(age_group)
    preferred_style_tag = _normalize_style_tag(preferred_style_tag)

    if not _is_allowed_region(style):
        return StyleScore(
            style_id=str(style.get("id", "")),
            name=str(style.get("name", style.get("id", "unnamed"))),
            score=-999.0,
            reference_image=str(style.get("reference_image", "")),
            presentation=presentation,
            maintenance_level=str(style.get("maintenance_level", "medium")),
            style_tags=sorted(_as_set(style, "style_tags")),
            reasons=["not in the China/Korea-targeted aesthetic pool"],
        )

    if not _matches_presentation(style, presentation_preference):
        return StyleScore(
            style_id=str(style.get("id", "")),
            name=str(style.get("name", style.get("id", "unnamed"))),
            score=-998.0,
            reference_image=str(style.get("reference_image", "")),
            presentation=presentation,
            maintenance_level=str(style.get("maintenance_level", "medium")),
            style_tags=sorted(_as_set(style, "style_tags")),
            reasons=[f"not in the requested {presentation_preference} presentation pool"],
        )

    suited = _as_set(style, "suited_face_shapes")
    preferred_hairline = _as_set(style, "preferred_hairline")
    avoid_hairline = _as_set(style, "avoid_hairline")
    style_tags = _as_set(style, "style_tags")
    exposure = str(style.get("forehead_exposure", "medium"))
    coverage = str(style.get("hairline_coverage", "medium"))
    bangs = str(style.get("bangs", "none"))
    volume = str(style.get("volume", "medium"))
    face_length_effect = str(style.get("face_length_effect", "neutral"))
    jaw_effect = str(style.get("jaw_effect", "neutral"))
    side_weight = str(style.get("side_weight", "medium"))
    neck_balance = str(style.get("neck_balance", "neutral"))
    temple_coverage = str(style.get("temple_coverage", "medium"))
    silhouette_balance = str(style.get("silhouette_balance", "medium"))
    trend_score = float(style.get("trend_score", 0.55))
    origin = str(style.get("origin_label", "asia_reference"))
    status = str(style.get("status", "unknown"))
    reference_image = str(style.get("reference_image", ""))
    maintenance_level = str(style.get("maintenance_level", "medium")).strip().lower()

    if face_shape_hint in suited:
        score += 3.2
        reasons.append(f"catalog marks it as suitable for {face_shape_hint}")
    else:
        score -= 0.6
        reasons.append(f"not a primary fit for {face_shape_hint}")

    if hairline_height_hint in preferred_hairline:
        score += 2.4
        reasons.append(f"works better for {hairline_height_hint} hairline")
    if hairline_height_hint in avoid_hairline:
        score -= 2.8
        reasons.append(f"should avoid {hairline_height_hint} hairline")

    if face_shape_hint == "oval":
        if "clean" in style_tags or "natural" in style_tags:
            score += 0.8
            reasons.append("clean/natural styling is stable for oval faces")
        if exposure == "high" and bangs in {"none", "slicked_back"}:
            score -= 0.9
            reasons.append("too much forehead exposure can look severe")
    elif face_shape_hint == "round":
        if volume in {"top", "medium"}:
            score += 1.1
            reasons.append("adds vertical structure for a round face")
        if side_weight == "heavy":
            score -= 0.8
            reasons.append("heavy side weight can widen a round face")
    elif face_shape_hint == "square":
        if jaw_effect == "softens":
            score += 1.3
            reasons.append("softens a stronger jaw line")
        if bangs in {"curtain", "textured_fringe", "side"}:
            score += 0.7
            reasons.append("fringe helps soften facial edges")
    elif face_shape_hint == "oblong":
        if face_length_effect == "shortens":
            score += 1.5
            reasons.append("helps shorten a visually long face")
        if exposure == "high":
            score -= 1.5
            reasons.append("high forehead exposure can lengthen the face further")
    elif face_shape_hint == "heart":
        if side_weight in {"medium", "heavy"}:
            score += 0.9
            reasons.append("adds support around the lower face")
        if jaw_effect == "softens":
            score += 0.5
            reasons.append("keeps the lower face more balanced")

    if hairline_height_hint == "high":
        if coverage == "high":
            score += 1.6
            reasons.append("high hairline coverage is safer")
        if bangs in {"curtain", "side", "textured_fringe", "full", "short_fringe"}:
            score += 1.2
            reasons.append("fringe helps reduce a tall-forehead impression")
        if exposure == "high":
            score -= 2.2
            reasons.append("too much forehead exposure for a high hairline")
        if bangs in {"none", "slicked_back"}:
            score -= 1.7
            reasons.append("exposes the hairline too directly")
    elif hairline_height_hint == "balanced":
        if exposure == "medium":
            score += 0.6
            reasons.append("balanced forehead exposure works well here")
        if "workplace" in style_tags or "clean" in style_tags:
            score += 0.4
            reasons.append("stable mainstream styling")
    elif hairline_height_hint == "low":
        if coverage == "high" and bangs == "full":
            score -= 1.1
            reasons.append("full heavy bangs may compress a lower hairline too much")
        if exposure in {"medium", "high"}:
            score += 0.5
            reasons.append("some forehead opening can improve balance")

    if hairline_pattern_hint in {"open_forehead", "temple_recession"} or recession_risk_hint == "high":
        if temple_coverage in {"medium", "high"}:
            score += 1.6
            reasons.append("temple coverage is safer for recession or open-forehead cases")
        if bangs in {"curtain", "textured_fringe", "short_fringe"}:
            score += 1.5
            reasons.append("textured fringe helps soften temple recession and forehead exposure")
        if bangs == "side":
            score -= 2.8
            reasons.append("side-part exposure tends to exaggerate temple recession")
        if exposure in {"medium", "high"}:
            score -= 1.4
            reasons.append("too much open-forehead exposure for the current hairline pattern")
        if silhouette_balance == "high":
            score += 0.8
            reasons.append("frontal silhouette stays more balanced around the head shape")
    elif hairline_pattern_hint == "balanced":
        if bangs == "side" and preferred_style_tag in {"stable", "workplace"}:
            score += 0.35
            reasons.append("side part can still work for a balanced mainstream presentation")

    if maintenance_preference != "any":
        if maintenance_level == maintenance_preference:
            score += 0.9
            reasons.append(f"matches the requested {maintenance_preference}-maintenance routine")
        elif maintenance_preference == "low" and maintenance_level == "high":
            score -= 1.5
            reasons.append("too demanding for a low-maintenance routine")
        elif maintenance_preference == "high" and maintenance_level == "low":
            score -= 0.2
            reasons.append("may be too plain for a higher-maintenance styling goal")
        else:
            score -= 0.4
            reasons.append("maintenance intensity is slightly off target")

    if forehead_goal == "cover":
        if coverage == "high" or bangs in {"curtain", "textured_fringe", "full", "short_fringe", "side"}:
            score += 1.7
            reasons.append("better for covering the forehead and softening the hairline")
        if exposure == "high":
            score -= 1.9
            reasons.append("shows too much forehead for a cover-focused goal")
    elif forehead_goal == "balance":
        if exposure == "medium" and coverage in {"medium", "high"}:
            score += 1.0
            reasons.append("keeps forehead exposure in a balanced range")
        if bangs == "full" and hairline_height_hint == "low":
            score -= 0.7
            reasons.append("may over-compress the forehead for a balance-focused goal")
    elif forehead_goal == "open":
        if exposure in {"medium", "high"} and bangs not in {"full", "textured_fringe"}:
            score += 1.2
            reasons.append("supports a cleaner open-forehead presentation")
        if coverage == "high":
            score -= 1.1
            reasons.append("covers too much of the forehead for an open look")

    if preferred_style_tag != "any":
        if preferred_style_tag in style_tags:
            score += 1.1
            reasons.append(f"matches the requested {preferred_style_tag} style direction")
        else:
            score -= 0.35
            reasons.append(f"does not strongly express the requested {preferred_style_tag} direction")

    score = _score_personal_profile(style, personal_profile, score, reasons)
    score = _score_age_group(style, age_group, score, reasons)

    score += (trend_score - 0.5) * 2.4
    if trend_score >= 0.8:
        reasons.append("closer to the current China-first trend direction")
    elif trend_score <= 0.45:
        score -= 0.45
        reasons.append("less aligned with the current mainstream trend direction")

    if origin == "cn_mainstream":
        score += 0.9
        reasons.append("prioritizes Chinese mainstream aesthetic")
    elif origin == "kr_support":
        score += 0.35
        reasons.append("uses Korean styling as a secondary reference")

    if status == "vetted_reference" and reference_image:
        score += 0.55
        reasons.append("has a vetted local reference ready for generation")
    elif status == "needs_vetted_reference":
        score -= 0.15
        reasons.append("reference is still pending manual review")

    if presentation_preference != "any":
        score += 0.5
        reasons.append(f"matches the requested {presentation_preference} presentation")

    if "workplace" in style_tags:
        score += 0.35
    if "rejuvenating" in style_tags and bangs in {"curtain", "textured_fringe", "side"}:
        score += 0.3
    if neck_balance == "slims":
        score += 0.4
        reasons.append("less likely to create a heavy neck/jaw impression")
    elif neck_balance == "bulky":
        score -= 1.0
        reasons.append("may create a heavy lower-face or neck impression")

    if not reasons:
        reasons.append("neutral match")

    return StyleScore(
        style_id=str(style.get("id", "")),
        name=str(style.get("name", style.get("id", "unnamed"))),
        score=round(score, 4),
        reference_image=reference_image,
        presentation=presentation,
        maintenance_level=maintenance_level,
        style_tags=sorted(style_tags),
        reasons=reasons,
    )


def recommend_styles(
    face_shape_hint: str,
    catalog_path: str | Path,
    hairline_height_hint: str = "unknown",
    hairline_pattern_hint: str = "unknown",
    recession_risk_hint: str = "unknown",
    personal_profile: dict[str, Any] | None = None,
    top_k: int = 5,
    age_group: str = "any",
    presentation_preference: str = "any",
    maintenance_preference: str = "any",
    forehead_goal: str = "auto",
    preferred_style_tag: str = "any",
) -> list[StyleScore]:
    catalog = load_catalog(catalog_path)
    scored = [
        score_style(
            style,
            face_shape_hint,
            hairline_height_hint,
            hairline_pattern_hint,
            recession_risk_hint,
            personal_profile=personal_profile,
            age_group=age_group,
            presentation_preference=presentation_preference,
            maintenance_preference=maintenance_preference,
            forehead_goal=forehead_goal,
            preferred_style_tag=preferred_style_tag,
        )
        for style in catalog
    ]
    scored = [item for item in scored if item.score > -100]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend hairstyle references from a local catalog.")
    parser.add_argument("--face-shape", required=True, help="Face shape hint, e.g. oval, round, square.")
    parser.add_argument("--hairline", default="unknown", help="Hairline hint: low, balanced, high, unknown.")
    parser.add_argument(
        "--presentation",
        default="any",
        choices=["masculine", "feminine", "any"],
        help="Preferred presentation pool.",
    )
    parser.add_argument("--maintenance", default="any", choices=["low", "medium", "high", "any"])
    parser.add_argument("--forehead-goal", default="auto", choices=["auto", "cover", "balance", "open"])
    parser.add_argument("--age-group", default="any", choices=["teen", "young_adult", "adult", "middle_aged", "senior", "any"])
    parser.add_argument("--style-tag", default="any")
    parser.add_argument(
        "--catalog",
        default="data/hairstyles/catalog.example.json",
        help="Hairstyle catalog JSON path.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    result = recommend_styles(
        args.face_shape,
        args.catalog,
        hairline_height_hint=args.hairline,
        hairline_pattern_hint="unknown",
        recession_risk_hint="unknown",
        personal_profile=None,
        top_k=args.top_k,
        age_group=args.age_group,
        presentation_preference=args.presentation,
        maintenance_preference=args.maintenance,
        forehead_goal=args.forehead_goal,
        preferred_style_tag=args.style_tag,
    )
    print(json.dumps([item.__dict__ for item in result], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
