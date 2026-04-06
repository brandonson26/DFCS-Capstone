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


def _integral(a: np.ndarray) -> np.ndarray:
    sat = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=float)
    sat[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    return sat


def _rect_sum(sat: np.ndarray, ys: np.ndarray, xs: np.ndarray, h: int, w: int) -> np.ndarray:
    y0 = ys[:, None]
    x0 = xs[None, :]
    y1 = y0 + int(h)
    x1 = x0 + int(w)
    return sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]


def _box_centroid(patch: np.ndarray) -> Tuple[float, float]:
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
    step = int(max(1, step))
    x_max = w - box_w
    y_max = h - box_h
    ys = np.arange(0, y_max + 1, step, dtype=int)
    xs = np.arange(0, x_max + 1, step, dtype=int)

    if ys.size == 0 or xs.size == 0:
        y0, x0 = np.unravel_index(np.argmax(img), img.shape)
        return BoxResult(
            x0=int(x0),
            y0=int(y0),
            x1=int(x0),
            y1=int(y0),
            cx=float(x0),
            cy=float(y0),
            sum_pos=float(max(img[y0, x0], 0.0)),
        )

    # Fast exact path for integrated + compact_flux.
    if score_mode in {"integrated", "compact_flux"}:
        pos = np.clip(img, 0, None)
        Y, X = np.indices(pos.shape, dtype=float)
        sat_w = _integral(pos)
        sat_x = _integral(pos * X)
        sat_y = _integral(pos * Y)
        sat_x2y2 = _integral(pos * (X * X + Y * Y))

        sum_w = _rect_sum(sat_w, ys, xs, box_h, box_w)

        if score_mode == "integrated":
            i_flat = int(np.argmax(sum_w))
            iy, ix = np.unravel_index(i_flat, sum_w.shape)
            yy = int(ys[iy])
            xx = int(xs[ix])
            s = float(sum_w[iy, ix])
            patch = img[yy:yy + box_h, xx:xx + box_w]
            cx_p, cy_p = _box_centroid(patch)
            return BoxResult(
                x0=xx,
                y0=yy,
                x1=xx + box_w - 1,
                y1=yy + box_h - 1,
                cx=float(xx + cx_p),
                cy=float(yy + cy_p),
                sum_pos=float(s),
            )

        sum_x = _rect_sum(sat_x, ys, xs, box_h, box_w)
        sum_y = _rect_sum(sat_y, ys, xs, box_h, box_w)
        sum_x2y2 = _rect_sum(sat_x2y2, ys, xs, box_h, box_w)
        m0 = sum_w + 1e-12
        cx_abs = sum_x / m0
        cy_abs = sum_y / m0
        var = np.maximum(sum_x2y2 / m0 - cx_abs * cx_abs - cy_abs * cy_abs, 0.0)
        compact = np.sqrt(var)
        score = sum_w / (1.0 + compact)

        i_flat = int(np.argmax(score))
        iy, ix = np.unravel_index(i_flat, score.shape)
        yy = int(ys[iy])
        xx = int(xs[ix])
        s = float(sum_w[iy, ix])
        cx = float(cx_abs[iy, ix]) if s > 0 else xx + (box_w - 1) / 2.0
        cy = float(cy_abs[iy, ix]) if s > 0 else yy + (box_h - 1) / 2.0
        return BoxResult(
            x0=xx,
            y0=yy,
            x1=xx + box_w - 1,
            y1=yy + box_h - 1,
            cx=float(cx),
            cy=float(cy),
            sum_pos=float(s),
        )

    # Legacy contrast mode retained.
    best: Optional[Tuple[float, BoxResult]] = None
    for yy in range(0, y_max + 1, step):
        for xx in range(0, x_max + 1, step):
            patch = img[yy:yy + box_h, xx:xx + box_w]
            pos = np.clip(patch, 0, None)
            s = float(np.sum(pos))
            mad = robust_mad(patch)
            score = s / (mad + 1e-6)

            if best is None or score > best[0]:
                cx_p, cy_p = _box_centroid(patch)
                br = BoxResult(
                    x0=xx,
                    y0=yy,
                    x1=xx + box_w - 1,
                    y1=yy + box_h - 1,
                    cx=float(xx + cx_p),
                    cy=float(yy + cy_p),
                    sum_pos=float(s),
                )
                best = (float(score), br)

    if best is None:
        y0, x0 = np.unravel_index(np.argmax(img), img.shape)
        return BoxResult(
            x0=int(x0),
            y0=int(y0),
            x1=int(x0),
            y1=int(y0),
            cx=float(x0),
            cy=float(y0),
            sum_pos=float(max(img[y0, x0], 0.0)),
        )
    return best[1]
