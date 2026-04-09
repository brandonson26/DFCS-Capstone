# low_snr.py
# ------------------------------------------------------------
# PURPOSE:
#   Detect "low SNR" extracted spectra using three complementary methods:
#
#   1. Global SNR (original):
#      Baseline + noise estimated from the lowest 10% of post-zeroth flux.
#      Flags if median SNR < threshold OR 25th-pct SNR < threshold.
#
#   2. Sustained signal:
#      Counts the longest consecutive run of pixels above 3-sigma.
#      A real spectrum has a sustained region of signal; pure noise does not.
#      Flags if the longest run is shorter than `min_sustained_px`.
#
#   3. First-order window SNR:
#      If the centroid of the first-order is known, computes SNR specifically
#      in a window around that position. This is more targeted than the global
#      check and catches cases where the spectrum is present but very faint.
#
#   Final flag: low_snr = (global_snr_flag) AND (sustained_flag OR fo_snr_flag)
#   Requiring the global check to agree prevents spurious flags from the
#   targeted checks alone.
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
from typing import Optional


def mad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to ~sigma for Gaussian noise."""
    x = np.asarray(x, dtype=float)
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def _longest_run_above(arr: np.ndarray, threshold: float) -> int:
    """Return the length of the longest consecutive run where arr > threshold."""
    above = arr > threshold
    max_run = 0
    cur_run = 0
    for v in above:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_run


def detect_low_snr(
    flux: np.ndarray,
    *,
    step: float,
    pre: float,
    after_offset_px: float = 5.0,
    lowest_frac: float = 0.10,
    snr_thresh_median: float = 5.0,
    snr_thresh_p25: float = 2.0,
    min_sustained_px: int = 10,
    sustained_sigma: float = 3.0,
    fo_window_px: float = 50.0,
    fo_snr_thresh: float = 3.0,
    first_order_dist: Optional[float] = None,
    zeroth_order_dist: Optional[float] = None,
    zeroth_snr_thresh: float = 5.0,
    zeroth_window_px: float = 30.0,
    clip_negative: bool = True,
) -> tuple[bool, dict]:
    """
    Parameters
    ----------
    flux : 1D array
        Extracted profile (starts pre pixels before zeroth order).
    step : float
        Sampling step in pixels.
    pre : float
        Pixels before zeroth included at the start of the profile.
    after_offset_px : float
        How far after zeroth to start analysis (skips the bright zeroth peak).
    lowest_frac : float
        Fraction of lowest values used to estimate baseline + noise.
    snr_thresh_median, snr_thresh_p25 : float
        Global SNR thresholds (method 1).
    min_sustained_px : int
        Minimum consecutive pixels above noise to consider signal real (method 2).
    sustained_sigma : float
        Sigma multiplier for the sustained-signal threshold (method 2).
    fo_window_px : float
        Half-width (pixels) of the window around the first-order centroid (method 3).
    fo_snr_thresh : float
        Minimum SNR in the first-order window (method 3).
    first_order_dist : float, optional
        Distance along the extraction line of the first-order centroid.
        If provided, enables method 3 (first-order window SNR).
    clip_negative : bool
        Clip flux to zero before analysis.

    Returns
    -------
    low_snr_flag : bool
    info : dict with all metrics
    """
    flux = np.asarray(flux, dtype=float)
    if flux.size == 0:
        return True, {"low_snr": True, "reason": "empty_flux",
                      "snr_median": 0.0, "snr_p25": 0.0,
                      "baseline": 0.0, "noise_sigma": 0.0}

    if clip_negative:
        flux = np.clip(flux, 0.0, None)

    step = float(step) if step else 1.0
    pre = float(pre)

    # Index just after zeroth (skip the bright zeroth peak).
    i_after = int(round((pre + after_offset_px) / max(step, 1e-12)))
    i_after = int(np.clip(i_after, 0, flux.size - 1))

    post = flux[i_after:]
    if post.size < 10:
        return True, {"low_snr": True, "reason": "too_few_post_samples",
                      "snr_median": 0.0, "snr_p25": 0.0,
                      "baseline": float(np.median(post)) if post.size else 0.0,
                      "noise_sigma": float(mad(post)) if post.size else 0.0}

    # ── Method 1: Global SNR ────────────────────────────────────────────────
    k = max(5, int(round(post.size * float(lowest_frac))))
    k = int(np.clip(k, 1, post.size))
    lowest = np.partition(post, k - 1)[:k]

    baseline = float(np.mean(lowest))
    noise_sigma = float(mad(lowest)) + 1e-9

    signal = np.clip(post - baseline, 0.0, None)
    snr = signal / noise_sigma
    snr_median = float(np.median(snr))
    snr_p25 = float(np.percentile(snr, 25))
    global_snr_flag = bool(
        (snr_median < float(snr_thresh_median)) or (snr_p25 < float(snr_thresh_p25))
    )

    # ── Method 2: Sustained signal ──────────────────────────────────────────
    threshold = baseline + sustained_sigma * noise_sigma
    longest_run = _longest_run_above(post, threshold)
    sustained_flag = bool(longest_run < int(min_sustained_px))

    # ── Method 3: First-order window SNR ────────────────────────────────────
    fo_snr = None
    fo_snr_flag = False
    if first_order_dist is not None:
        # Convert first_order_dist to an index in the full flux array.
        fo_idx = int(round(first_order_dist / max(step, 1e-12)))
        fo_idx = int(np.clip(fo_idx, 0, flux.size - 1))
        half_w = int(round(fo_window_px / max(step, 1e-12)))
        i0 = max(0, fo_idx - half_w)
        i1 = min(flux.size, fo_idx + half_w + 1)
        fo_window = flux[i0:i1]
        if fo_window.size >= 3:
            fo_signal = float(np.max(fo_window)) - baseline
            fo_snr = fo_signal / noise_sigma
            fo_snr_flag = bool(fo_snr < float(fo_snr_thresh))

    # ── Method 4: Zeroth order SNR ──────────────────────────────────────────
    # The zeroth order is the brightest source in the image. If even it has a
    # peak flux < zeroth_snr_thresh × background noise, the observation is too
    # faint to extract a usable spectrum regardless of any other metric.
    zeroth_snr = None
    zeroth_snr_flag = False
    if zeroth_order_dist is not None:
        zo_idx = int(round(zeroth_order_dist / max(step, 1e-12)))
        zo_idx = int(np.clip(zo_idx, 0, flux.size - 1))
        half_w = int(round(zeroth_window_px / max(step, 1e-12)))
        i0 = max(0, zo_idx - half_w)
        i1 = min(flux.size, zo_idx + half_w + 1)
        zo_window = flux[i0:i1]
        if zo_window.size >= 3:
            zeroth_peak = float(np.max(zo_window))
            zeroth_snr = zeroth_peak / noise_sigma
            zeroth_snr_flag = bool(zeroth_snr < float(zeroth_snr_thresh))

    # ── Combined decision ───────────────────────────────────────────────────
    # Zeroth order SNR is a standalone trigger — if the source itself is too
    # faint, flag immediately regardless of other checks.
    # Otherwise: global SNR must agree plus at least one targeted check.
    if zeroth_snr_flag:
        low_snr_flag = True
    elif fo_snr is not None:
        low_snr_flag = bool(global_snr_flag and (sustained_flag or fo_snr_flag))
    else:
        low_snr_flag = bool(global_snr_flag and sustained_flag)

    info = {
        "low_snr": low_snr_flag,
        # Method 1
        "snr_median": snr_median,
        "snr_p25": snr_p25,
        "baseline": baseline,
        "noise_sigma": noise_sigma,
        "global_snr_flag": global_snr_flag,
        # Method 2
        "longest_sustained_run_px": int(longest_run * step),
        "sustained_flag": sustained_flag,
        "min_sustained_px": int(min_sustained_px),
        # Method 3
        "fo_snr": fo_snr,
        "fo_snr_flag": fo_snr_flag,
        # Method 4
        "zeroth_snr": zeroth_snr,
        "zeroth_snr_flag": zeroth_snr_flag,
        # Metadata
        "post_len": int(post.size),
        "i_after": int(i_after),
        "thresholds": {
            "snr_thresh_median": float(snr_thresh_median),
            "snr_thresh_p25": float(snr_thresh_p25),
            "lowest_frac": float(lowest_frac),
            "after_offset_px": float(after_offset_px),
            "sustained_sigma": float(sustained_sigma),
            "fo_snr_thresh": float(fo_snr_thresh),
            "zeroth_snr_thresh": float(zeroth_snr_thresh),
        },
    }
    return low_snr_flag, info
