# partial_first_order.py

import numpy as np


def path_reaches_edge(xs, ys, W, H, tol_px=2.5):
    """
    Check that extraction path actually reaches detector edge.
    """
    if len(xs) == 0:
        return False, np.inf

    xL = xs[-1]
    yL = ys[-1]

    d_left = abs(xL - 0)
    d_right = abs(xL - (W - 1))
    d_top = abs(yL - 0)
    d_bottom = abs(yL - (H - 1))

    edge_dist = min(d_left, d_right, d_top, d_bottom)

    return edge_dist <= tol_px, edge_dist


def partial_first_order_photon_check(
    flux,
    step,
    pre,
    after_win_px=20,
    after_offset_px=5,
    tail_frac=0.05,
    lowest_frac=0.10,
    end_ratio_thresh=0.4,
):
    """
    Detect partial first order using photon counts.
    """

    flux = np.asarray(flux, dtype=float)
    flux = np.clip(flux, 0, None)

    # ---- zeroth index ----
    i0 = int(round(pre / step))
    i0 = np.clip(i0, 0, len(flux) - 1)

    # ---- after zeroth window ----
    i_after_start = int(round((pre + after_offset_px) / step))
    i_after_end = int(round((pre + after_offset_px + after_win_px) / step))

    i_after_start = np.clip(i_after_start, 0, len(flux))
    i_after_end = np.clip(i_after_end, i_after_start + 1, len(flux))

    after_region = flux[i_after_start:i_after_end]
    after_mean = np.mean(after_region) if len(after_region) else 0.0

    # ---- end of image ----
    tail_n = max(5, int(len(flux) * tail_frac))
    end_region = flux[-tail_n:]
    end_mean = np.mean(end_region)

    # ---- lowest area baseline ----
    post = flux[i_after_start:]
    if len(post):
        k = max(5, int(len(post) * lowest_frac))
        lowest_vals = np.partition(post, k - 1)[:k]
        baseline = np.mean(lowest_vals)
    else:
        baseline = 0.0

    after_net = max(0, after_mean - baseline)
    end_net = max(0, end_mean - baseline)

    ratio = end_net / after_net if after_net > 1e-9 else 0

    partial = ratio >= end_ratio_thresh

    metrics = {
        "after_net": after_net,
        "end_net": end_net,
        "baseline": baseline,
        "ratio": ratio,
    }

    return partial, metrics