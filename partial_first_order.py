# partial_first_order.py
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Detect when a satellite's first-order spectrum is cut off at the image
#   boundary — i.e. the spectrum would continue beyond the detector edge.
#
# APPROACH — Horizontal Edge Probe:
#   The extraction path runs from just before the zeroth order to the image
#   edge in the direction of the brighter first order. At the very last point
#   of the path (x_end, y_end) a horizontal line is sampled: 100 pixels to
#   the left and 100 pixels to the right (200 px total).
#
#   If the first-order spectrum is COMPLETE (fully inside the image):
#       → The spectrum has faded out before reaching the edge.
#       → The horizontal probe at the edge sees flat background — no spike.
#
#   If the first-order spectrum is PARTIAL (cut off at the edge):
#       → The spectrum is still active at the boundary.
#       → The horizontal probe crosses live spectral flux and shows a peak
#         that is ≥ 150 % of the mean background level along the probe.
#
#   A separate diagnostic graph is saved alongside the spectrum PNG showing
#   the horizontal profile, the background mean, and the threshold.
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def path_reaches_edge(xs: np.ndarray, ys: np.ndarray,
                      W: int, H: int, tol_px: float = 2.5) -> Tuple[bool, float]:
    """Return (reaches_edge, dist_to_nearest_edge) for the last path point."""
    if len(xs) == 0:
        return False, float("inf")
    xL, yL = float(xs[-1]), float(ys[-1])
    edge_dist = min(abs(xL), abs(xL - (W - 1)), abs(yL), abs(yL - (H - 1)))
    return edge_dist <= tol_px, edge_dist


def detect_partial_first_order(
    img: np.ndarray,
    x_end: float,
    y_end: float,
    W: int,
    H: int,
    half_width: int = 100,
    spike_ratio: float = 3.0,
) -> Tuple[bool, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Detect partial first order by sampling a horizontal line at the far image
    edge endpoint of the extraction path.

    At (x_end, y_end) a 200-pixel horizontal line (±half_width px) is sampled
    from the raw image. If any peak in that profile is ≥ spike_ratio times the
    mean background level of the probe, the first order is flagged as partial.

    Parameters
    ----------
    img : 2D array
        Image to sample (img_raw for real photon counts).
    x_end, y_end : float
        Far endpoint of the extraction path — at the image boundary.
    W, H : int
        Image width and height.
    half_width : int
        Pixels to sample left and right of the endpoint (default 100 → 200 px total).
    spike_ratio : float
        A peak must be >= spike_ratio × background mean to flag partial (default 3.0 = 200% above).

    Returns
    -------
    is_partial : bool
    info : dict
        Diagnostic metrics.
    offsets : 1D array
        X offsets from endpoint (−half_width … +half_width).
    values : 1D array
        Sampled image flux at each offset position.
    """
    # ── Build horizontal sample positions ────────────────────────────────────
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    sample_x = np.clip(x_end + offsets, 0, W - 1)
    sample_y = np.full_like(offsets, np.clip(y_end, 0, H - 1))

    # Bilinear interpolation along the horizontal probe.
    values = map_coordinates(img, [sample_y, sample_x], order=1, mode="nearest").astype(float)

    # ── Compute background mean and spike threshold ───────────────────────────
    # Use the mean of the whole probe as the background reference.
    # In a genuine background region every value is roughly equal, so the mean
    # is a stable estimate. If the spectrum is present, the mean is pulled up
    # slightly but the peak will still be >> spike_ratio × mean.
    bg_mean = float(np.mean(values))
    threshold = spike_ratio * bg_mean

    # ── Look for peaks above the threshold ───────────────────────────────────
    peak_indices, peak_props = find_peaks(values, height=threshold, prominence=bg_mean * 0.3)

    is_partial = len(peak_indices) > 0
    peak_heights = peak_props.get("peak_heights", np.array([], dtype=float))

    info = {
        "is_partial":      is_partial,
        "bg_mean":         round(bg_mean, 4),
        "threshold":       round(threshold, 4),
        "spike_ratio":     spike_ratio,
        "half_width_px":   half_width,
        "n_peaks":         int(len(peak_indices)),
        "peak_offsets_px": [round(float(offsets[i]), 1) for i in peak_indices],
        "peak_heights":    [round(float(h), 2) for h in peak_heights],
        "end_edge":        {"x": round(x_end, 1), "y": round(y_end, 1)},
    }

    return is_partial, info, offsets, values


def save_partial_first_order_png(
    dest_dir: Path,
    offsets: np.ndarray,
    values: np.ndarray,
    bg_mean: float,
    threshold: float,
    is_partial: bool,
) -> Path:
    """
    Save a diagnostic graph of the horizontal edge probe used for partial
    first-order detection.

    The graph shows:
      - Blue line  : sampled flux along the 200-px horizontal probe
      - Orange line: background mean level
      - Red dashed : spike threshold (50 % above background mean)
      - Title indicates whether partial first order was flagged
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_png = dest_dir / "partial_first_order_probe.png"

    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot(offsets, values, color="steelblue", linewidth=1.2, label="Edge probe flux")
    ax.axhline(bg_mean,    color="orange", linewidth=1.2, linestyle="-",  label=f"Background mean ({bg_mean:.1f})")
    ax.axhline(threshold,  color="red",    linewidth=1.2, linestyle="--", label=f"Threshold ×1.5 ({threshold:.1f})")

    ax.set_xlabel("Horizontal offset from edge point (pixels)")
    ax.set_ylabel("Flux (image units)")
    ax.set_title(
        f"Partial First Order Edge Probe — {'PARTIAL (flagged)' if is_partial else 'No spike detected'}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    return out_png
