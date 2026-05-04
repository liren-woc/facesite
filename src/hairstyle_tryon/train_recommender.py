from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from hairstyle_tryon.recommend import load_catalog
else:
    from .recommend import load_catalog


COMMON_STYLE_TAGS = [
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
]

USER_FEATURES = [
    "face_shape_hint",
    "hairline_height_hint",
    "face_ratio_h_w",
    "forehead_to_face_ratio",
    "presentation_preference",
    "maintenance_preference",
    "forehead_goal",
    "preferred_style_tag",
    "age_group",
]

STYLE_FEATURES = [
    "presentation",
    "length",
    "bangs",
    "curl",
    "volume",
    "covers_forehead",
    "forehead_exposure",
    "hairline_coverage",
    "face_length_effect",
    "jaw_effect",
    "maintenance_level",
    *[f"tag_{tag}" for tag in COMMON_STYLE_TAGS],
]

NUMERIC_FEATURES = ["face_ratio_h_w", "forehead_to_face_ratio", *[f"tag_{tag}" for tag in COMMON_STYLE_TAGS]]
CATEGORICAL_FEATURES = [
    "face_shape_hint",
    "hairline_height_hint",
    "presentation_preference",
    "maintenance_preference",
    "forehead_goal",
    "preferred_style_tag",
    "age_group",
    "presentation",
    "length",
    "bangs",
    "curl",
    "volume",
    "covers_forehead",
    "forehead_exposure",
    "hairline_coverage",
    "face_length_effect",
    "jaw_effect",
    "maintenance_level",
]

OPTIONAL_USER_DEFAULTS = {
    "presentation_preference": "any",
    "maintenance_preference": "any",
    "forehead_goal": "auto",
    "preferred_style_tag": "any",
    "age_group": "any",
}


def _style_to_row(style: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "style_id": style["id"],
        "presentation": style.get("presentation", "any"),
        "length": style.get("length"),
        "bangs": style.get("bangs"),
        "curl": style.get("curl"),
        "volume": style.get("volume"),
        "covers_forehead": style.get("covers_forehead"),
        "forehead_exposure": style.get("forehead_exposure"),
        "hairline_coverage": style.get("hairline_coverage"),
        "face_length_effect": style.get("face_length_effect"),
        "jaw_effect": style.get("jaw_effect"),
        "maintenance_level": style.get("maintenance_level", "medium"),
    }
    tags = {str(item) for item in style.get("style_tags", [])}
    for tag in COMMON_STYLE_TAGS:
        row[f"tag_{tag}"] = int(tag in tags)
    return row


def build_training_frame(feedback_csv: str | Path, catalog_path: str | Path) -> pd.DataFrame:
    feedback = pd.read_csv(feedback_csv)
    required = {"face_shape_hint", "hairline_height_hint", "face_ratio_h_w", "forehead_to_face_ratio", "style_id", "label"}
    missing = sorted(required - set(feedback.columns))
    if missing:
        raise ValueError(f"Feedback CSV is missing required columns: {missing}")

    for column, default in OPTIONAL_USER_DEFAULTS.items():
        if column not in feedback.columns:
            feedback[column] = default
        else:
            feedback[column] = feedback[column].fillna(default)

    catalog = load_catalog(catalog_path)
    styles = pd.DataFrame([_style_to_row(style) for style in catalog])
    df = feedback.merge(styles, on="style_id", how="left", validate="many_to_one")
    style_descriptor_columns = [column for column in STYLE_FEATURES if not column.startswith("tag_")]
    if df[style_descriptor_columns].isna().all(axis=1).any():
        missing_styles = sorted(df.loc[df["presentation"].isna(), "style_id"].unique())
        raise ValueError(f"Feedback references style IDs not found in catalog: {missing_styles}")

    df["label"] = df["label"].astype(int)
    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("unknown").astype(str)
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def build_model() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train(
    *,
    feedback_csv: str | Path,
    catalog_path: str | Path,
    output_model: str | Path,
    output_metrics: str | Path,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    df = build_training_frame(feedback_csv, catalog_path)
    if len(df) < 8:
        raise ValueError("Need at least 8 feedback rows to train a baseline recommender.")
    if df["label"].nunique() < 2:
        raise ValueError("Need both positive and negative feedback labels.")

    X = df[USER_FEATURES + STYLE_FEATURES]
    y = df["label"]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "feedback_csv": str(feedback_csv),
        "catalog_path": str(catalog_path),
        "num_rows": int(len(df)),
        "num_train_rows": int(len(X_train)),
        "num_test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "features": USER_FEATURES + STYLE_FEATURES,
        "note": "Use real user feedback before treating this as a meaningful model.",
    }

    output_model = Path(output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_model)

    output_metrics = Path(output_metrics)
    output_metrics.parent.mkdir(parents=True, exist_ok=True)
    with output_metrics.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small hairstyle recommendation ranker.")
    parser.add_argument("--feedback-csv", required=True)
    parser.add_argument("--catalog", default="data/hairstyles/catalog.example.json")
    parser.add_argument("--output-model", default="outputs/recommender/style_ranker.joblib")
    parser.add_argument("--output-metrics", default="outputs/recommender/metrics.json")
    args = parser.parse_args()

    metrics = train(
        feedback_csv=args.feedback_csv,
        catalog_path=args.catalog,
        output_model=args.output_model,
        output_metrics=args.output_metrics,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
