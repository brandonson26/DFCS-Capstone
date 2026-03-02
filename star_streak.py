#!/usr/bin/env python3
"""
star_streak.py

Minimal star streak detector module.

This file is meant to be IMPORTED by capstone.py:
    from star_streak import detect_star_streak

No console printing, no subprocess calls, no CLI, no file I/O.
capstone.py handles reading FITS, extracting spectrum, saving outputs, and routing folders.
"""

import numpy as np
from scipy.signal import savgol_filter, find_peaks


def detect_star_streak(s, profile):
    """
    Parameters
    ----------
    s : 1D array
        Distance along extraction path (pixels).
    profile : 1D array
        Extracted 1D flux/counts array.

    Returns
    -------
    has_streak : bool
    peaks : np.ndarray (indices into profile)
    smoothed : np.ndarray
    props : dict (from scipy.signal.find_peaks)
    valid_mask : list[bool]
        Per-peak mask: True means peak is considered "real/valid" under the rules.
    """
    s = np.asarray(s, dtype=float)
    profile = np.asarray(profile, dtype=float)

    if profile.size < 7:
        return False, np.array([], dtype=int), profile.copy(), {}, []

    # Ensure an odd window <= len(profile)
    # Your original logic: min(101, len(profile) - (len(profile) % 2 == 0))
    n = int(profile.size)
    window = min(101, n)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        window = 5 if n >= 5 else (n | 1)

    smoothed = savgol_filter(profile, window, 3, mode="interp")
    median_back = float(np.median(smoothed))

    peaks, props = find_peaks(
        smoothed,
        height=median_back * 1.2,
        prominence=median_back * 1.5,
        width=1,
        distance=max(1, int(0.02 * len(s)))
    )

    # Handle case where find_peaks returns empty props keys
    heights = props.get("peak_heights", np.array([], dtype=float))
    prominences = props.get("prominences", np.array([], dtype=float))
    widths = props.get("widths", np.array([], dtype=float))

    if len(peaks) < 2:
        return False, peaks, smoothed, props, [False] * len(peaks)

    sorted_idx = np.argsort(heights)[::-1]
    zeroth_idx = int(sorted_idx[0])
    first_idx = int(sorted_idx[1])

    zeroth_pos = int(peaks[zeroth_idx])
    first_pos = int(peaks[first_idx])

    valid_mask = []
    has_streak = False

    for pk, prom, w, h in zip(peaks, prominences, widths, heights):
        pk = int(pk)

        # Always treat the two strongest peaks (zeroth + first) as valid
        if pk == zeroth_pos or pk == first_pos:
            valid_mask.append(True)
            continue

        after_first = pk > first_pos
        strong_enough = prom > 0.7 * float(prominences[first_idx])
        narrow_enough = w < 250
        above_background = h > median_back * 1.2

        if after_first and (strong_enough or narrow_enough or above_background):
            valid_mask.append(True)
            has_streak = True
        else:
            valid_mask.append(False)

    return has_streak, peaks, smoothed, props, valid_mask
