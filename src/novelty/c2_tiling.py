"""C2: relevance-proportional tiling of retrieved pages for a single-image VLM.

VisRAG Eq. (3) concatenates the top-k pages horizontally, so each page gets
an equal share of the VLM's pixel budget. C2 gives page i the area weight

    w_i = p_i^alpha / sum_j p_j^alpha,    p = softmax(s / tau),

so it receives B_i = B * w_i pixels. The expected share of pixels on the
relevant page is sum_i p_i w_i. With alpha = 1 this is sum_i p_i^2, which is
at least 1/k by Cauchy-Schwarz. For any alpha >= 0, p and w are sorted the
same way, so Chebyshev's sum inequality gives the same 1/k bound.

Cells come from a squarified treemap (Bruls, Huizing, van Wijk 2000).
Weights are laid out in descending order, so the top page gets the largest
cell, in the top-left corner, with sides on the order of sqrt(w_i). Each page
is scaled to fit inside its cell with the aspect ratio preserved and is never
stretched.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PIL import Image

from .c1_marginalization import retrieval_log_posterior

BACKGROUND = (255, 255, 255)

__all__ = [
    "baseline_horizontal_concat",
    "compute_tiling_layout",
    "relevance_proportional_tiling",
    "relevance_tiling_weights",
]


def baseline_horizontal_concat(
    images: list[Image.Image],
    total_budget_px: int = 1344 * 1344,
) -> Image.Image:
    """VisRAG Eq. (3): stitch k pages left to right in columns of equal width.

    Every page is resized to a common column width with its aspect ratio
    kept, and pages are top-aligned on a white canvas. The width is the
    largest that keeps ``canvas_width * canvas_height <= total_budget_px``,
    capped at the widest input page so small pages are not blown up.
    """
    pages = _validate_images(images)
    budget = _positive_int(total_budget_px, "total_budget_px")
    k = len(pages)
    aspects = [page.height / page.width for page in pages]
    max_aspect = max(aspects)

    cell_width = max(1, min(max(page.width for page in pages), int(math.sqrt(budget / (k * max_aspect)))))
    while True:
        heights = [max(1, round(cell_width * aspect)) for aspect in aspects]
        if cell_width == 1 or k * cell_width * max(heights) <= budget:
            break
        cell_width -= 1

    canvas = Image.new("RGB", (k * cell_width, max(heights)), BACKGROUND)
    for index, (page, height) in enumerate(zip(pages, heights)):
        canvas.paste(page.resize((cell_width, height), Image.Resampling.LANCZOS), (index * cell_width, 0))
    return canvas


def relevance_tiling_weights(
    cos_scores: np.ndarray,
    tau: float = 0.02,
    alpha: float = 1.0,
    total_budget_px: int = 1344 * 1344,
) -> dict:
    """Posterior, area weights, pixel budgets, and expected relevant-page share."""
    log_p = retrieval_log_posterior(cos_scores, tau)
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"alpha must be a non-negative finite scalar, got {alpha}")
    budget = _positive_int(total_budget_px, "total_budget_px")

    p = np.exp(log_p)
    log_w = alpha * log_p
    w = np.exp(log_w - log_w.max())
    w /= w.sum()
    k = p.shape[0]
    expected = float(np.dot(p, w))
    sum_p_squared = float(np.dot(p, p))
    return {
        "posterior": p,
        "weights": w,
        "budgets_px": budget * w,
        "expected_true_page_ratio": expected,
        "uniform_ratio": 1.0 / k,
        "gain_over_uniform": expected * k,
        "sum_p_squared": sum_p_squared,
        "cauchy_schwarz_holds": sum_p_squared >= 1.0 / k - 1e-12,
    }


def compute_tiling_layout(
    weights: np.ndarray,
    canvas_size: tuple[int, int] = (1344, 1344),
) -> list[tuple[int, int, int, int]]:
    """Integer ``(x0, y0, x1, y1)`` cells tiling the canvas, one per weight, in input order.

    Cell areas follow ``weights`` up to pixel rounding, and the cells cover
    the canvas with no gaps or overlaps. A zero weight gets an empty cell.
    """
    width, height = _canvas_dims(canvas_size)
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0:
        raise ValueError(f"weights must be a non-empty 1-D array, got shape {w.shape}")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("weights must be finite, non-negative, and not all zero")

    order = [int(i) for i in np.argsort(-w, kind="mergesort") if w[i] > 0]
    areas = [float(w[i] / w.sum()) * width * height for i in order]
    float_rects = _squarify(areas, 0.0, 0.0, float(width), float(height))

    rects: list[tuple[int, int, int, int]] = [(0, 0, 0, 0)] * w.shape[0]
    for index, (x, y, dx, dy) in zip(order, float_rects):
        rects[index] = (
            _snap(x, width),
            _snap(y, height),
            _snap(x + dx, width),
            _snap(y + dy, height),
        )
    return rects


def relevance_proportional_tiling(
    images: list[Image.Image],
    cos_scores: np.ndarray,
    tau: float = 0.02,
    alpha: float = 1.0,
    canvas_size: tuple[int, int] = (1344, 1344),
) -> tuple[Image.Image, list[float]]:
    """C2: tile pages on one canvas with cell area proportional to ``p_i^alpha``.

    Returns the composite and each page's cell area as a fraction of the
    canvas, in input order. Pages are scaled to fit their cells and centred,
    with letterboxing instead of stretching. Use ``relevance_tiling_weights``
    for the posterior, budgets, and expected relevant-page share.
    """
    pages = _validate_images(images)
    width, height = _canvas_dims(canvas_size)
    weights = relevance_tiling_weights(cos_scores, tau=tau, alpha=alpha, total_budget_px=width * height)
    if len(pages) != weights["weights"].shape[0]:
        raise ValueError(
            f"images and cos_scores must have the same length, got {len(pages)} and {weights['weights'].shape[0]}"
        )

    rects = compute_tiling_layout(weights["weights"], (width, height))
    canvas = Image.new("RGB", (width, height), BACKGROUND)
    fractions = []
    for page, (x0, y0, x1, y1) in zip(pages, rects):
        cell_w, cell_h = x1 - x0, y1 - y0
        fractions.append(cell_w * cell_h / (width * height))
        if cell_w <= 0 or cell_h <= 0:
            continue
        scale = min(cell_w / page.width, cell_h / page.height)
        new_w = min(cell_w, max(1, round(page.width * scale)))
        new_h = min(cell_h, max(1, round(page.height * scale)))
        resized = page.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas.paste(resized, (x0 + (cell_w - new_w) // 2, y0 + (cell_h - new_h) // 2))
    return canvas, fractions


def _squarify(
    areas: Sequence[float],
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[tuple[float, float, float, float]]:
    """Squarified treemap of descending ``areas`` into the given rectangle."""
    rects: list[tuple[float, float, float, float]] = []
    remaining = list(areas)
    while remaining:
        if len(remaining) == 1:
            rects.append((x, y, width, height))
            break
        short_side = min(width, height)
        row = [remaining[0]]
        cursor = 1
        while cursor < len(remaining) and _worst_ratio(row + [remaining[cursor]], short_side) <= _worst_ratio(
            row, short_side
        ):
            row.append(remaining[cursor])
            cursor += 1
        remaining = remaining[cursor:]
        row_area = sum(row)
        # The last row and the last cell of each row absorb float drift so the canvas is covered exactly.
        if width >= height:
            thickness = min(width, row_area / height) if remaining else width
            offset = y
            for i, area in enumerate(row):
                length = (y + height - offset) if i == len(row) - 1 else area / thickness
                rects.append((x, offset, thickness, length))
                offset += length
            x += thickness
            width -= thickness
        else:
            thickness = min(height, row_area / width) if remaining else height
            offset = x
            for i, area in enumerate(row):
                length = (x + width - offset) if i == len(row) - 1 else area / thickness
                rects.append((offset, y, length, thickness))
                offset += length
            y += thickness
            height -= thickness
    return rects


def _worst_ratio(row: Sequence[float], short_side: float) -> float:
    total = sum(row)
    if total <= 0 or short_side <= 0:
        return math.inf
    side_sq = short_side * short_side
    return max(max(side_sq * area / (total * total), (total * total) / (side_sq * area)) for area in row)


def _snap(value: float, limit: int) -> int:
    return int(min(limit, max(0, math.floor(value + 0.5))))


def _validate_images(images: Sequence[Image.Image]) -> list[Image.Image]:
    pages = list(images)
    if not pages:
        raise ValueError("images must not be empty")
    for page in pages:
        if not isinstance(page, Image.Image):
            raise TypeError(f"expected PIL.Image.Image, got {type(page).__name__}")
        if page.width <= 0 or page.height <= 0:
            raise ValueError("images must have positive width and height")
    return [page.convert("RGB") for page in pages]


def _canvas_dims(canvas_size: tuple[int, int]) -> tuple[int, int]:
    if len(canvas_size) != 2:
        raise ValueError(f"canvas_size must be (width, height), got {canvas_size!r}")
    return _positive_int(canvas_size[0], "canvas width"), _positive_int(canvas_size[1], "canvas height")


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)
