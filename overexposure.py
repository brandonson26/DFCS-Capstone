"""
overexposure.py

Purpose:
--------
Lightweight overexposure / saturation flag for spectral images.

What it does:
-------------
Compares two small image patches:
  1) A patch around the zeroth-order source
  2) A patch around the first-order source

It flags the frame as "overexposed" if EITHER:
  - The first-order patch reaches near the detector saturation level, OR
  - The first-order patch is brighter than the zeroth-order patch
    (often a sign of saturation, blooming, or mis-detection)

This function:
  - Does NOT modify the image
  - Does NOT correct exposure
  - Simply returns a boolean flag + diagnostic info

Typical use:
------------
from overexposure import detect_overexposure_first

is_over, info = detect_overexposure_first(
    data=img_raw,
    i0=zeroth_y, j0=zeroth_x,
    i1=first_y,  j1=first_x,
    h=100, w=100
)
"""

import numpy as np


def detect_overexposure_first(data, i0, j0, i1, j1, h, w,
                              sat=65535.0, margin=1000.0):
    """
    Parameters
    ----------
    data : 2D numpy array
        Image data (raw or background-subtracted).
    i0, j0 : int
        Top-left corner of zeroth-order patch (row, col).
    i1, j1 : int
        Top-left corner of first-order patch (row, col).
    h, w : int
        Height and width of both patches (pixels).
    sat : float
        Detector saturation level.
    margin : float
        Distance below saturation considered "near saturation".

    Returns
    -------
    is_over : bool
        True if frame is likely overexposed.
    info : dict
        Diagnostic values used in the decision.
    """

    H, W = data.shape

    # Safe patch extraction with boundary clipping
    def patch(i, j):
        i = int(np.clip(i, 0, H - 1))
        j = int(np.clip(j, 0, W - 1))
        return data[i:min(i + h, H), j:min(j + w, W)]

    z = patch(i0, j0)  # zeroth-order patch
    f = patch(i1, j1)  # first-order patch

    max0 = float(np.max(z)) if z.size else float("nan")
    max1 = float(np.max(f)) if f.size else float("nan")

    thr = sat - margin
    near_sat = np.isfinite(max1) and (max1 >= thr)
    brighter = np.isfinite(max0) and np.isfinite(max1) and (max1 > max0)

    return (
        bool(near_sat or brighter),
        {
            "max_zeroth": max0,
            "max_first": max1,
            "thr": thr,
            "first_near_sat": bool(near_sat),
            "first_brighter_than_zeroth": bool(brighter),
        },
    )
