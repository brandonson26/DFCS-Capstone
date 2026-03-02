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
    direction: str              # "above" or "below"
    mean_above: float
    mean_below: float
    search_box: Tuple[int,int,int,int]     # chosen box used to find first order
    first_point: BoxResult                  # compact point in first order

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

def _mean_pos(img: np.ndarray, box: Tuple[int,int,int,int]) -> float:
    x0,y0,x1,y1 = box
    if x1 < x0 or y1 < y0:
        return float("-inf")
    patch = img[y0:y1+1, x0:x1+1]
    if patch.size == 0:
        return float("-inf")
    return float(np.mean(np.clip(patch, 0, None)))

def _find_compact_in_bounds(
    img_detect: np.ndarray,
    bounds: Tuple[int,int,int,int],   # x0,y0,x1,y1 inclusive box bounds
    win_w: int = 21,
    win_h: int = 21,
    step: int = 3,
) -> BoxResult:
    """
    Find compact max integrated positive-flux window inside bounds.
    Returns window box + centroid.
    """
    img = np.asarray(img_detect, dtype=float)
    H, W = img.shape
    x0, y0, x1, y1 = bounds

    win_w = int(max(3, win_w))
    win_h = int(max(3, win_h))

    # top-left ranges
    x_min = int(np.clip(x0, 0, W - win_w))
    y_min = int(np.clip(y0, 0, H - win_h))
    x_max = int(np.clip(x1 - win_w + 1, 0, W - win_w))
    y_max = int(np.clip(y1 - win_h + 1, 0, H - win_h))

    best: Optional[Tuple[float, BoxResult]] = None

    for yy in range(y_min, y_max + 1, step):
        for xx in range(x_min, x_max + 1, step):
            patch = img[yy:yy+win_h, xx:xx+win_w]
            pos = np.clip(patch, 0, None)
            s = float(np.sum(pos))
            # compactness preference
            cx_p, cy_p = _box_centroid(patch)
            yy_i, xx_i = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            m0 = float(np.sum(pos)) + 1e-12
            var = float(np.sum(pos * ((xx_i - cx_p)**2 + (yy_i - cy_p)**2)) / m0)
            compact = float(np.sqrt(max(var, 0.0)))
            score = s / (1.0 + compact)

            if best is None or score > best[0]:
                br = BoxResult(
                    x0=xx, y0=yy,
                    x1=xx+win_w-1, y1=yy+win_h-1,
                    cx=float(xx+cx_p), cy=float(yy+cy_p),
                    sum_pos=float(s),
                )
                best = (float(score), br)

    if best is None:
        # fallback: brightest pixel inside bounds
        patch = img[y0:y1+1, x0:x1+1]
        if patch.size == 0:
            return BoxResult(x0=x0, y0=y0, x1=x0, y1=y0, cx=float(x0), cy=float(y0), sum_pos=0.0)
        yy_rel, xx_rel = np.unravel_index(np.argmax(patch), patch.shape)
        yy_abs = y0 + yy_rel
        xx_abs = x0 + xx_rel
        v = float(max(img[yy_abs, xx_abs], 0.0))
        return BoxResult(x0=xx_abs, y0=yy_abs, x1=xx_abs, y1=yy_abs, cx=float(xx_abs), cy=float(yy_abs), sum_pos=v)

    return best[1]

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
    """
    First order logic:
      - define two fixed-width boxes above and below zeroth (no overlap),
        each extending to the image edge in height
      - compute average positive flux in each on img_bgsub -> choose direction
      - inside chosen box, find compact flux point (inner window) on img_detect -> centroid
    """
    H, W = img_bgsub.shape
    fixed_w = int(max(10, min(fixed_w, W)))
    fixed_h = int(max(10, min(fixed_h, H)))

    cx = float(zeroth.cx)
    x0 = int(round(cx - fixed_w / 2))
    x0 = max(0, min(x0, W - fixed_w))
    x1 = x0 + fixed_w - 1

    # ABOVE box: directly above zeroth box, extend to image top
    above_y0 = int(zeroth.y1 + pad)
    if above_y0 > H - 1:
        above_y0, above_y1 = 0, -1  # empty
    else:
        above_y1 = H - 1

    # BELOW box: directly below zeroth box, extend to image bottom
    below_y1 = int(zeroth.y0 - pad)
    if below_y1 < 0:
        below_y0, below_y1 = 0, -1  # empty
    else:
        below_y0 = 0

    above_box = (x0, above_y0, x1, above_y1)
    below_box = (x0, below_y0, x1, below_y1)

    mean_above = _mean_pos(img_bgsub, above_box)
    mean_below = _mean_pos(img_bgsub, below_box)

    if mean_above >= mean_below:
        direction = "above"
        chosen = above_box
    else:
        direction = "below"
        chosen = below_box

    # If chosen box is empty (edge cases), fall back to the other
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
    )
