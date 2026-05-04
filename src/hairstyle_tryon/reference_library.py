from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .recommend import load_catalog


@dataclass(frozen=True)
class ReferenceLibraryStatus:
    total_styles: int
    vetted_styles: int
    pending_styles: int
    vetted_masculine: int
    vetted_feminine: int


def summarize_reference_library(catalog_path: str | Path) -> ReferenceLibraryStatus:
    catalog = load_catalog(catalog_path)
    total_styles = len(catalog)
    vetted_styles = 0
    vetted_masculine = 0
    vetted_feminine = 0

    for item in catalog:
        status = str(item.get("status", ""))
        reference_image = str(item.get("reference_image", ""))
        if status != "vetted_reference" or not reference_image:
            continue
        resolved = Path(reference_image)
        if not resolved.exists():
            resolved = (Path(__file__).resolve().parents[2] / reference_image).resolve()
        if not resolved.exists():
            continue
        vetted_styles += 1
        presentation = str(item.get("presentation", "any")).strip().lower()
        if presentation == "masculine":
            vetted_masculine += 1
        elif presentation == "feminine":
            vetted_feminine += 1

    return ReferenceLibraryStatus(
        total_styles=total_styles,
        vetted_styles=vetted_styles,
        pending_styles=max(total_styles - vetted_styles, 0),
        vetted_masculine=vetted_masculine,
        vetted_feminine=vetted_feminine,
    )
