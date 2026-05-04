from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageQuality:
    image_path: str
    width: int
    height: int
    mean_brightness: float
    blur_score: float
    colorfulness: float
    is_monochrome: bool
    warnings: list[str]


def check_image_quality(image_path: str | Path) -> ImageQuality:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Missing OpenCV. Install requirements-core.txt.") from exc

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    b_channel, g_channel, r_channel = cv2.split(image.astype("float32"))
    rg = cv2.absdiff(r_channel, g_channel)
    yb = cv2.absdiff(0.5 * (r_channel + g_channel), b_channel)
    std_rg, mean_rg = rg.std(), rg.mean()
    std_yb, mean_yb = yb.std(), yb.mean()
    colorfulness = float((std_rg**2 + std_yb**2) ** 0.5 + 0.3 * ((mean_rg**2 + mean_yb**2) ** 0.5))
    is_monochrome = colorfulness < 12.0

    warnings: list[str] = []
    if width < 512 or height < 512:
        warnings.append("image resolution is low; use at least 512x512 for better hair transfer")
    if mean_brightness < 55:
        warnings.append("image is dark; use brighter lighting")
    if mean_brightness > 205:
        warnings.append("image is very bright; reduce overexposure")
    if blur_score < 60:
        warnings.append("image may be blurry; use a sharper photo")
    if is_monochrome:
        warnings.append("image looks monochrome; realistic hair rendering needs a natural color photo")

    return ImageQuality(
        image_path=str(image_path),
        width=width,
        height=height,
        mean_brightness=mean_brightness,
        blur_score=blur_score,
        colorfulness=colorfulness,
        is_monochrome=is_monochrome,
        warnings=warnings,
    )
