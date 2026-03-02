#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

def robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

@dataclass
class BoxResult:
    x0: int
    y0: int
    x1: int
    y1: int
    cx: float
    cy: float
    sum_pos: float

def _box_centroid(patch: np.ndarray) -> Tuple[float, float]:
    """Flux-weighted centroid in patch coords (0..w-1, 0..h-1), using positive flux."""
    wgt = np.clip(patch, 0, None)
    s = float(np.sum(wgt))
    if s <= 0:
        h, w = patch.shape
        return (w - 1) / 2.0, (h - 1) / 2.0
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    cx = float(np.sum(wgt * xx) / (s + 1e-12))
    cy = float(np.sum(wgt * yy) / (s + 1e-12))
    return cx, cy

def find_zeroth_order(
    img_detect: np.ndarray,
    box_w: int = 100,
    box_h: int = 100,
    step: int = 4,
    score_mode: str = "compact_flux",  # integrated | contrast | compact_flux
) -> BoxResult:
    """
    Zeroth order:
      - scans the WHOLE image
      - selects the box with highest local integrated flux (positive)
      - returns centroid (flux-weighted) INSIDE that box
    """
    img = np.asarray(img_detect, dtype=float)
    h, w = img.shape
    box_w = int(max(3, box_w))
    box_h = int(max(3, box_h))
    x_max = w - box_w
    y_max = h - box_h

    best: Optional[Tuple[float, BoxResult]] = None

    for yy in range(0, y_max + 1, step):
        for xx in range(0, x_max + 1, step):
            patch = img[yy:yy + box_h, xx:xx + box_w]
            pos = np.clip(patch, 0, None)
            s = float(np.sum(pos))

            if score_mode == "integrated":
                score = s
            elif score_mode == "compact_flux":
                cx_p, cy_p = _box_centroid(patch)
                yy_i, xx_i = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
                wgt = pos
                m0 = float(np.sum(wgt)) + 1e-12
                var = float(np.sum(wgt * ((xx_i - cx_p) ** 2 + (yy_i - cy_p) ** 2)) / m0)
                compact = float(np.sqrt(max(var, 0.0)))
                score = s / (1.0 + compact)
            else:
                mad = robust_mad(patch)
                score = s / (mad + 1e-6)

            if best is None or score > best[0]:
                cx_p, cy_p = _box_centroid(patch)
                br = BoxResult(
                    x0=xx, y0=yy,
                    x1=xx + box_w - 1, y1=yy + box_h - 1,
                    cx=float(xx + cx_p),
                    cy=float(yy + cy_p),
                    sum_pos=float(s),
                )
                best = (float(score), br)

    if best is None:
        y0, x0 = np.unravel_index(np.argmax(img), img.shape)
        return BoxResult(x0=int(x0), y0=int(y0), x1=int(x0), y1=int(y0),
                         cx=float(x0), cy=float(y0), sum_pos=float(max(img[y0, x0], 0.0)))
    return best[1]
