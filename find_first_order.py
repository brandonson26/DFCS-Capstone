#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class BoxResult:
    x0: int
    y0: int
    x1: int
    y1: int
    cx: float
    cy: float
    sum_pos: float


@dataclass
class FirstOrderResult:
    direction: str
    mean_above: float
    mean_below: float
    search_box: Tuple[int, int, int, int]
    first_point: BoxResult
    above_point: Optional[BoxResult] = None
    below_point: Optional[BoxResult] = None


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


def _mean_pos(img: np.ndarray, box: Tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = box
    if x1 < x0 or y1 < y0:
        return float("-inf")
    patch = img[y0:y1 + 1, x0:x1 + 1]
    if patch.size == 0:
        return float("-inf")
    return float(np.mean(np.clip(patch, 0, None)))


def _find_compact_in_bounds(
    img_detect: np.ndarray,
    bounds: Tuple[int, int, int, int],
    win_w: int = 21,
    win_h: int = 21,
    step: int = 3,
) -> BoxResult:
    img = np.asarray(img_detect, dtype=float)
    H, W = img.shape
    x0, y0, x1, y1 = bounds

    win_w = int(max(3, win_w))
    win_h = int(max(3, win_h))
    step = int(max(1, step))

    x_min = int(np.clip(x0, 0, W - win_w))
    y_min = int(np.clip(y0, 0, H - win_h))
    x_max = int(np.clip(x1 - win_w + 1, 0, W - win_w))
    y_max = int(np.clip(y1 - win_h + 1, 0, H - win_h))

    ys = np.arange(y_min, y_max + 1, step, dtype=int)
    xs = np.arange(x_min, x_max + 1, step, dtype=int)

    if ys.size > 0 and xs.size > 0:
        pos = np.clip(img, 0, None)
        Y, X = np.indices(pos.shape, dtype=float)
        sat_w = _integral(pos)
        sat_x = _integral(pos * X)
        sat_y = _integral(pos * Y)
        sat_x2y2 = _integral(pos * (X * X + Y * Y))

        sum_w = _rect_sum(sat_w, ys, xs, win_h, win_w)
        sum_x = _rect_sum(sat_x, ys, xs, win_h, win_w)
        sum_y = _rect_sum(sat_y, ys, xs, win_h, win_w)
        sum_x2y2 = _rect_sum(sat_x2y2, ys, xs, win_h, win_w)

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
        cx = float(cx_abs[iy, ix]) if s > 0 else xx + (win_w - 1) / 2.0
        cy = float(cy_abs[iy, ix]) if s > 0 else yy + (win_h - 1) / 2.0
        return BoxResult(
            x0=xx,
            y0=yy,
            x1=xx + win_w - 1,
            y1=yy + win_h - 1,
            cx=float(cx),
            cy=float(cy),
            sum_pos=float(s),
        )

    patch = img[y0:y1 + 1, x0:x1 + 1]
    if patch.size == 0:
        return BoxResult(x0=x0, y0=y0, x1=x0, y1=y0, cx=float(x0), cy=float(y0), sum_pos=0.0)
    yy_rel, xx_rel = np.unravel_index(np.argmax(patch), patch.shape)
    yy_abs = y0 + yy_rel
    xx_abs = x0 + xx_rel
    v = float(max(img[yy_abs, xx_abs], 0.0))
    return BoxResult(x0=xx_abs, y0=yy_abs, x1=xx_abs, y1=yy_abs, cx=float(xx_abs), cy=float(yy_abs), sum_pos=v)


def find_first_order(
    img_bgsub: np.ndarray,
    img_detect: np.ndarray,
    zeroth: BoxResult,
    fixed_w: int = 400,
    fixed_h: int = 1000,
    pad: int = 5,
    inner_win_w: int = 21,
    inner_win_h: int = 21,
) -> FirstOrderResult:
    H, W = img_bgsub.shape
    fixed_w = int(max(10, min(fixed_w, W)))
    fixed_h = int(max(10, min(fixed_h, H)))

    cx = float(zeroth.cx)
    x0 = int(round(cx - fixed_w / 2))
    x0 = max(0, min(x0, W - fixed_w))
    x1 = x0 + fixed_w - 1

    above_y0 = int(zeroth.y1 + pad)
    if above_y0 > H - 1:
        above_y0, above_y1 = 0, -1
    else:
        above_y1 = H - 1

    below_y1 = int(zeroth.y0 - pad)
    if below_y1 < 0:
        below_y0, below_y1 = 0, -1
    else:
        below_y0 = 0

    above_box = (x0, above_y0, x1, above_y1)
    below_box = (x0, below_y0, x1, below_y1)

    mean_above = _mean_pos(img_bgsub, above_box)
    mean_below = _mean_pos(img_bgsub, below_box)

    above_first: Optional[BoxResult] = None
    below_first: Optional[BoxResult] = None
    if above_box[3] >= above_box[1]:
        above_first = _find_compact_in_bounds(
            img_detect=img_detect,
            bounds=above_box,
            win_w=inner_win_w,
            win_h=inner_win_h,
            step=3,
        )
    if below_box[3] >= below_box[1]:
        below_first = _find_compact_in_bounds(
            img_detect=img_detect,
            bounds=below_box,
            win_w=inner_win_w,
            win_h=inner_win_h,
            step=3,
        )

    if mean_above >= mean_below:
        direction = "above"
        chosen = above_box
        first = above_first if above_first is not None else below_first
    else:
        direction = "below"
        chosen = below_box
        first = below_first if below_first is not None else above_first

    if first is None:
        if chosen[3] < chosen[1]:
            direction = "below" if direction == "above" else "above"
            chosen = below_box if direction == "below" else above_box
        first = _find_compact_in_bounds(
            img_detect=img_detect,
            bounds=chosen,
            win_w=inner_win_w,
            win_h=inner_win_h,
            step=3,
        )

    return FirstOrderResult(
        direction=direction,
        mean_above=float(mean_above),
        mean_below=float(mean_below),
        search_box=chosen,
        first_point=first,
        above_point=above_first,
        below_point=below_first,
    )
