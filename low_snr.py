# low_snr.py
# ------------------------------------------------------------
# PURPOSE:
#   Detect "low SNR" extracted spectra in a simple, robust way.
#
#   You already extract a 1D profile `flux` along the dispersion line.
#   This module estimates:
#     - baseline background level (from the lowest-flux region after zeroth)
#     - noise sigma (MAD-based) from that same low region
#     - SNR statistics (median and 25th percentile) on the post-zeroth spectrum
#
#   Default operational rule:
#     low_snr = (snr_median < 5.0) OR (snr_p25 < 2.0)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np


def mad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to ~sigma for Gaussian noise."""
    x = np.asarray(x, dtype=float)
    m = np.median(x)
    return 1.4826 * np.median(np.abs(x - m))


def detect_low_snr(
    flux: np.ndarray,
    *,
    step: float,
    pre: float,
    after_offset_px: float = 5.0,
    lowest_frac: float = 0.10,
    snr_thresh_median: float = 5.0,
    snr_thresh_p25: float = 2.0,
    clip_negative: bool = True,
) -> tuple[bool, dict]:
    """
    Parameters
    ----------
    flux : 1D array
        Extracted profile along the dispersion path (starts pre pixels before zeroth).
    step : float
        Sampling step in pixels (same as args.step).
    pre : float
        Pixels before zeroth included at start (same as args.pre).
    after_offset_px : float
        How far after zeroth we start analyzing to avoid the bright zeroth peak.
    lowest_frac : float
        Fraction of lowest values (after zeroth) used to estimate baseline + noise.
    snr_thresh_median : float
        If median SNR < this => low SNR.
    snr_thresh_p25 : float
        If 25th percentile SNR < this => low SNR.

    Returns
    -------
    low_snr_flag : bool
    info : dict with metrics
    """

    flux = np.asarray(flux, dtype=float)
    if flux.size == 0:
        return True, {
            "low_snr": True,
            "reason": "empty_flux",
            "snr_median": 0.0,
            "snr_p25": 0.0,
            "baseline": 0.0,
            "noise_sigma": 0.0,
        }

    if clip_negative:
        flux = np.clip(flux, 0.0, None)

    step = float(step) if step else 1.0
    pre = float(pre)

    # Index just after zeroth (skip the zeroth peak area)
    i_after = int(round((pre + after_offset_px) / max(step, 1e-12)))
    i_after = int(np.clip(i_after, 0, flux.size - 1))

    post = flux[i_after:]
    if post.size < 10:
        # Not enough samples after zeroth to be meaningful
        return True, {
            "low_snr": True,
            "reason": "too_few_post_samples",
            "snr_median": 0.0,
            "snr_p25": 0.0,
            "baseline": float(np.median(post)) if post.size else 0.0,
            "noise_sigma": float(mad(post)) if post.size else 0.0,
        }

    # Baseline + noise estimated from lowest region after zeroth
    k = max(5, int(round(post.size * float(lowest_frac))))
    k = int(np.clip(k, 1, post.size))
    lowest = np.partition(post, k - 1)[:k]

    baseline = float(np.mean(lowest))
    noise_sigma = float(mad(lowest)) + 1e-9  # avoid divide-by-zero

    # Signal above baseline
    signal = post - baseline
    signal = np.clip(signal, 0.0, None)

    snr = signal / noise_sigma

    snr_median = float(np.median(snr))
    snr_p25 = float(np.percentile(snr, 25))

    low_snr_flag = bool((snr_median < float(snr_thresh_median)) or (snr_p25 < float(snr_thresh_p25)))

    info = {
        "low_snr": low_snr_flag,
        "snr_median": snr_median,
        "snr_p25": snr_p25,
        "baseline": baseline,
        "noise_sigma": noise_sigma,
        "post_len": int(post.size),
        "i_after": int(i_after),
        "thresholds": {
            "snr_thresh_median": float(snr_thresh_median),
            "snr_thresh_p25": float(snr_thresh_p25),
            "lowest_frac": float(lowest_frac),
            "after_offset_px": float(after_offset_px),
        },
    }
    return low_snr_flag, info