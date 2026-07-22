"""Deterministic page-text linearization strategies.

Column-major linearization exists because the Gorkhapatra silver references
sort lines strict y-then-x across multi-column layouts
(``tools/build_limbu_gorkhapatra_all_corpus.py``), and a few pixels of
detector-vs-PDF y-jitter flip that interleaving constantly: on the 16-page
Limbu pack the same documents score page cp CER 0.4134 under strict-y
comparison but 0.0451 when both sides are linearized column-major
(2026-07-05, ``outputs/line-merge-v1-2026-07-05/``). The function below is
the single shared definition — page-eval v2 applies it to the reference and
the prediction, and the assembly stage emits it — so scores stay comparable
only within one linearization definition.
"""

from __future__ import annotations

import statistics
from typing import Sequence

# Fraction of the median box width within which two x-centers belong to the
# same column band. Validated on the 16-page Limbu pack (6 columns, ~580 px
# wide, ~40 px gutters): 0.6 keeps gutters separating and jitter joining.
COLUMN_CENTER_BAND_FRACTION = 0.6

XYWH = tuple[float, float, float, float]


def column_major_indices(
    boxes: Sequence[XYWH],
    *,
    band_fraction: float = COLUMN_CENTER_BAND_FRACTION,
) -> list[int]:
    """Order box indices column-major: x-center bands left-to-right, then y.

    ``boxes`` are ``(x, y, w, h)`` in any consistent pixel space. Boxes are
    clustered into column bands by x-center proximity (within
    ``band_fraction`` x the median box width of the running band mean);
    bands are emitted left-to-right, boxes within a band top-to-bottom.
    Deterministic: ties resolve by (y, x, input index).
    """
    if band_fraction <= 0:
        raise ValueError(f"band_fraction must be positive, got {band_fraction}")
    if not boxes:
        return []
    for index, box in enumerate(boxes):
        if len(box) != 4:
            raise ValueError(f"box {index} must be (x, y, w, h), got {box!r}")
    median_width = statistics.median(box[2] for box in boxes)
    if median_width <= 0:
        # Degenerate geometry (all zero-width boxes): fall back to y-then-x.
        return sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0], i))

    ordered = sorted(range(len(boxes)), key=lambda i: (boxes[i][0] + boxes[i][2] / 2.0, i))
    bands: list[dict] = []
    for index in ordered:
        x, _, w, _ = boxes[index]
        center_x = x + w / 2.0
        placed = False
        for band in bands:
            if abs(center_x - band["center_x"]) < band_fraction * median_width:
                band["members"].append(index)
                band["center_sum"] += center_x
                band["center_x"] = band["center_sum"] / len(band["members"])
                placed = True
                break
        if not placed:
            bands.append({"center_x": center_x, "center_sum": center_x, "members": [index]})

    bands.sort(key=lambda band: band["center_x"])
    result: list[int] = []
    for band in bands:
        result.extend(
            sorted(band["members"], key=lambda i: (boxes[i][1], boxes[i][0], i))
        )
    return result


def linearize_texts_column_major(
    items: Sequence[tuple[XYWH, str]],
    *,
    band_fraction: float = COLUMN_CENTER_BAND_FRACTION,
) -> str:
    """Join item texts with newlines in column-major order."""
    order = column_major_indices([item[0] for item in items], band_fraction=band_fraction)
    return "\n".join(items[i][1] for i in order)
