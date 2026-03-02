#!/usr/bin/env python3
"""
capstone.py

Pipeline split:
  - capstone.py: FITS load, HDU inventory/selection, header plausibility,
                 visual confirmation (log scale), background subtract,
                 call find_zeroth_order + find_first_order, line + DS9-like spectrum,
                 outputs PNGs + CSV
  - find_zeroth_order.py: box+centroid for zeroth
  - find_first_order.py: fixed 400x100 direction boxes + compact point for first order

Outputs:
  - 02_points_and_line.png
  - 03_spectrum_pixel.png
  - spectrum_with_characterization.csv
"""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter, map_coordinates

from find_zeroth_order import find_zeroth_order, BoxResult as ZerothBox
from find_first_order import find_first_order, FirstOrderResult


# ----------------------------- utilities -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clip_nan_inf(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float, copy=True)
    bad = ~np.isfinite(a)
    if np.any(bad):
        a[bad] = np.nanmedian(a)
    return a

def percentile_clip(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.7) -> np.ndarray:
    lo = np.percentile(img, p_lo)
    hi = np.percentile(img, p_hi)
    return np.clip(img, lo, hi)

def safe_log_display(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = img.copy()
    m = np.median(x)
    x = x - m
    x = np.sign(x) * np.log1p(np.abs(x) + eps)
    return x

def header_subset(hdr) -> Dict[str, Any]:
    keys = ["DATE-OBS", "EXPTIME", "EXPOSURE", "GAIN", "RDNOISE", "SATURATE", "BUNIT",
            "NAXIS1", "NAXIS2", "IMAGETYP", "OBJECT", "FILTER"]
    out = {}
    for k in keys:
        if k in hdr:
            out[k] = hdr[k]
    return out

def header_plausibility(hdr: fits.Header) -> List[str]:
    notes: List[str] = []
    exptime = hdr.get("EXPTIME", hdr.get("EXPOSURE", None))
    if exptime is None:
        notes.append("Header: missing EXPTIME/EXPOSURE.")
    else:
        try:
            ex = float(exptime)
            if ex <= 0:
                notes.append(f"Header: non-positive exposure time ({ex}).")
        except Exception:
            notes.append(f"Header: EXPTIME/EXPOSURE not numeric ({exptime}).")

    for k in ["GAIN", "RDNOISE", "SATURATE"]:
        if k in hdr:
            try:
                float(hdr[k])
            except Exception:
                notes.append(f"Header: {k} present but not numeric ({hdr[k]}).")

    return notes


# ----------------------------- background -----------------------------

def estimate_background(img: np.ndarray, tile: int = 64) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    h, w = img.shape
    th = max(1, h // tile)
    tw = max(1, w // tile)

    bg_small = np.zeros((th, tw), dtype=float)
    for i in range(th):
        for j in range(tw):
            y0 = int(i * h / th); y1 = int((i + 1) * h / th)
            x0 = int(j * w / tw); x1 = int((j + 1) * w / tw)
            bg_small[i, j] = np.median(img[y0:y1, x0:x1])

    yy = np.linspace(0, th - 1, h)
    xx = np.linspace(0, tw - 1, w)
    y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
    y1 = np.clip(y0 + 1, 0, th - 1); x1 = np.clip(x0 + 1, 0, tw - 1)
    wy = yy - y0; wx = xx - x0

    bg = (
        (1 - wy)[:, None] * (1 - wx)[None, :] * bg_small[y0[:, None], x0[None, :]] +
        (1 - wy)[:, None] * (wx)[None, :]     * bg_small[y0[:, None], x1[None, :]] +
        (wy)[:, None]     * (1 - wx)[None, :] * bg_small[y1[:, None], x0[None, :]] +
        (wy)[:, None]     * (wx)[None, :]     * bg_small[y1[:, None], x1[None, :]]
    )
    return bg


# ----------------------------- HDU inventory / selection -----------------------------

def is_image_like(hdu) -> bool:
    if hdu.data is None:
        return False
    arr = np.asarray(hdu.data)
    return (arr.ndim == 2 and np.issubdtype(arr.dtype, np.number))

def pick_image_hdu(hdul: fits.HDUList, ext_override: Optional[int]) -> int:
    if ext_override is not None:
        return ext_override
    best = None
    for i, hdu in enumerate(hdul):
        if not is_image_like(hdu):
            continue
        arr = np.asarray(hdu.data)
        h, w = arr.shape
        score = 2.0
        if h >= 512 and w >= 512:
            score += 1.0
        exptime = float(hdu.header.get("EXPTIME", hdu.header.get("EXPOSURE", 0.0)) or 0.0)
        if exptime > 0:
            score += 1.0
        if best is None or score > best[0]:
            best = (score, i)
    if best is None:
        raise RuntimeError("No image-like (2D numeric) HDU found.")
    return best[1]


# ----------------------------- line + spectrum -----------------------------

def extend_line_to_image_edge(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> Tuple[float, float]:
    dx = x1 - x0
    dy = y1 - y0
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return x1, y1

    ts = []
    if abs(dx) > 1e-12:
        ts += [(0 - x0) / dx, ((w - 1) - x0) / dx]
    if abs(dy) > 1e-12:
        ts += [(0 - y0) / dy, ((h - 1) - y0) / dy]

    ts = [t for t in ts if t > 0]
    if not ts:
        return x1, y1

    t = min(ts)
    xe = float(np.clip(x0 + t * dx, 0, w - 1))
    ye = float(np.clip(y0 + t * dy, 0, h - 1))
    return xe, ye

def point_before_zeroth(xz: float, yz: float, xf: float, yf: float, pre: float) -> Tuple[float, float]:
    dx = xf - xz
    dy = yf - yz
    L = float(np.hypot(dx, dy))
    if L < 1e-9:
        return xz, yz
    ux, uy = dx / L, dy / L
    return xz - pre * ux, yz - pre * uy

def sample_line_profile(img: np.ndarray,
                        x1: float, y1: float,
                        x2: float, y2: float,
                        width: int = 1,
                        step: float = 1.0,
                        reducer: str = "mean") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length <= 1e-9:
        raise ValueError("Line length is ~0.")
    n = int(np.floor(length / step)) + 1
    t = np.linspace(0.0, 1.0, n)
    xs = x1 + t * dx
    ys = y1 + t * dy
    dist = t * length

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux

    width = max(1, int(width))
    if width == 1:
        vals = map_coordinates(img, np.vstack([ys, xs]), order=1, mode="nearest")
        return dist, xs, ys, vals

    half = (width - 1) / 2.0
    offs = np.linspace(-half, half, width)
    stack = []
    for o in offs:
        xw = xs + o * px
        yw = ys + o * py
        stack.append(map_coordinates(img, np.vstack([yw, xw]), order=1, mode="nearest"))
    stack = np.vstack(stack)
    vals = np.median(stack, axis=0) if reducer == "median" else np.mean(stack, axis=0)
    return dist, xs, ys, vals


# ----------------------------- plots + CSV -----------------------------

def save_points_and_line_png(outdir: Path,
                             img: np.ndarray,
                             zeroth: ZerothBox,
                             first_res: FirstOrderResult,
                             xL1: float, yL1: float, xL2: float, yL2: float) -> Path:
    plt.figure(figsize=(9, 6))
    disp = safe_log_display(percentile_clip(img))
    plt.imshow(disp, origin="lower", aspect="auto")

    # points only
    plt.scatter([zeroth.cx], [zeroth.cy], s=80, marker="x", c="red", label="Zeroth")
    fo = first_res.first_point
    plt.scatter([fo.cx], [fo.cy], s=70, marker="o", c="cyan", label="First")

    # extraction line
    plt.plot([xL1, xL2], [yL1, yL2], linewidth=1.6)
    plt.title(f"Points + extraction path | direction={first_res.direction}")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    p = outdir / "02_points_and_line.png"
    plt.savefig(p, dpi=170)
    plt.close()
    return p

def save_spectrum_png(outdir: Path, dist: np.ndarray, flux: np.ndarray) -> Path:
    plt.figure(figsize=(9, 4))
    plt.plot(dist, flux)
    plt.xlabel("Distance along extraction path (pixels)")
    plt.ylabel("Flux (image units)")
    plt.title("Spectrum (DS9-like Plot2D)")
    plt.tight_layout()
    p = outdir / "03_spectrum_pixel.png"
    plt.savefig(p, dpi=170)
    plt.close()
    return p

def write_csv(csv_path: Path,
              fits_path: Path,
              hdu_index: int,
              hdr_small: Dict[str, Any],
              plausibility: List[str],
              zeroth: ZerothBox,
              first_res: FirstOrderResult,
              line_pts: Tuple[float,float,float,float],
              dist: np.ndarray, xs: np.ndarray, ys: np.ndarray, flux: np.ndarray) -> None:
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(f"# source_fits={fits_path}\n")
        f.write(f"# hdu_index={hdu_index}\n")
        for k, v in hdr_small.items():
            f.write(f"# hdr_{k}={v}\n")
        for note in plausibility:
            f.write(f"# header_note={note}\n")

        f.write(f"# zeroth_box=({zeroth.x0},{zeroth.y0})-({zeroth.x1},{zeroth.y1})\n")
        f.write(f"# zeroth_centroid=({zeroth.cx:.3f},{zeroth.cy:.3f})\n")

        f.write(f"# first_direction={first_res.direction}\n")
        f.write(f"# first_mean_above={first_res.mean_above:.6g}\n")
        f.write(f"# first_mean_below={first_res.mean_below:.6g}\n")
        bx0,by0,bx1,by1 = first_res.search_box
        f.write(f"# first_search_box=({bx0},{by0})-({bx1},{by1})\n")
        fo = first_res.first_point
        f.write(f"# first_point_box=({fo.x0},{fo.y0})-({fo.x1},{fo.y1})\n")
        f.write(f"# first_point_centroid=({fo.cx:.3f},{fo.cy:.3f})\n")

        x1,y1,x2,y2 = line_pts
        f.write(f"# line_start=({x1:.3f},{y1:.3f})\n")
        f.write(f"# line_end=({x2:.3f},{y2:.3f})\n")

        f.write("distance_pix,x,y,flux\n")
        arr = np.column_stack([dist, xs, ys, flux])
        np.savetxt(f, arr, delimiter=",", fmt="%.10g")


# ----------------------------- main -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Capstone: DS9-like spectra extraction (box-based zeroth + fixed first-order boxes).")
    ap.add_argument("--fits", required=True, help="Path to FITS file")
    ap.add_argument("--outdir", default="outputs_capstone", help="Output directory")
    ap.add_argument("--ext", type=int, default=None, help="HDU index override (default: auto best)")

    # background
    ap.add_argument("--bg-tile", type=int, default=64)
    ap.add_argument("--smooth", type=float, default=1.0, help="Smoothing for detection image (bg-sub)")

    # zeroth
    ap.add_argument("--zeroth-box-w", type=int, default=100)
    ap.add_argument("--zeroth-box-h", type=int, default=100)
    ap.add_argument("--zeroth-step", type=int, default=4)
    ap.add_argument("--zeroth-score-mode", choices=["integrated", "compact_flux", "contrast"], default="compact_flux")

    # first-order fixed boxes (your standard)
    ap.add_argument("--first-fixed-w", type=int, default=400)
    ap.add_argument("--first-fixed-h", type=int, default=100)
    ap.add_argument("--first-pad", type=int, default=5)
    ap.add_argument("--first-inner-w", type=int, default=21, help="Inner window size to find compact point")
    ap.add_argument("--first-inner-h", type=int, default=21)

    # extraction line + sampling
    ap.add_argument("--pre", type=float, default=30.0, help="Pixels to start before zeroth along the line")
    ap.add_argument("--profile-on", choices=["raw", "bgsub"], default="bgsub")
    ap.add_argument("--width", type=int, default=5, help="Line width averaged perpendicular to path (DS9-like)")
    ap.add_argument("--reducer", choices=["mean", "median"], default="mean")
    ap.add_argument("--step", type=float, default=1.0)

    args = ap.parse_args()

    fits_path = Path(args.fits)
    if not fits_path.exists():
        raise SystemExit(f"File not found: {fits_path}")

    base_outdir = Path(args.outdir)
    fits_outdir = base_outdir / fits_path.stem
    ensure_dir(fits_outdir)

    with fits.open(fits_path) as hdul:
        # inventory & verify HDU0
        hdu0_ok = is_image_like(hdul[0]) if len(hdul) else False
        if not hdu0_ok:
            print("[WARN] HDU0 is not a 2D numeric image. Auto-selecting best image HDU (or use --ext).")

        hdu_index = pick_image_hdu(hdul, args.ext)
        hdu = hdul[hdu_index]
        hdr = hdu.header
        hdr_small = header_subset(hdr)
        plaus = header_plausibility(hdr)

        img_raw = clip_nan_inf(np.asarray(hdu.data))
        H, W = img_raw.shape

        # background subtract
        bg = estimate_background(img_raw, tile=args.bg_tile)
        img_bgsub = img_raw - bg

        # analysis mode: box-based auto
        img_detect = gaussian_filter(img_bgsub, args.smooth) if args.smooth > 0 else img_bgsub

        # zeroth (whole image)
        zeroth = find_zeroth_order(
            img_detect=img_detect,
            box_w=args.zeroth_box_w,
            box_h=args.zeroth_box_h,
            step=args.zeroth_step,
            score_mode=args.zeroth_score_mode
        )

        # first order (fixed 400x100 above/below + compact point)
        first_res = find_first_order(
            img_bgsub=img_bgsub,
            img_detect=img_detect,
            zeroth=zeroth,
            fixed_w=args.first_fixed_w,
            fixed_h=args.first_fixed_h,
            pad=args.first_pad,
            inner_win_w=args.first_inner_w,
            inner_win_h=args.first_inner_h
        )
        first_pt = first_res.first_point

        # extraction line: start before zeroth, pass through first point, extend to edge
        x_start, y_start = point_before_zeroth(zeroth.cx, zeroth.cy, first_pt.cx, first_pt.cy, pre=args.pre)
        x_end, y_end = extend_line_to_image_edge(zeroth.cx, zeroth.cy, first_pt.cx, first_pt.cy, w=W, h=H)

        img_profile = img_raw if args.profile_on == "raw" else img_bgsub
        dist, xs, ys, flux = sample_line_profile(
            img_profile, x_start, y_start, x_end, y_end,
            width=args.width, step=args.step, reducer=args.reducer
        )

        # outputs
        save_points_and_line_png(fits_outdir, img_raw, zeroth, first_res, x_start, y_start, x_end, y_end)
        save_spectrum_png(fits_outdir, dist, flux)
        write_csv(fits_outdir / "spectrum_with_characterization.csv",
                  fits_path=fits_path, hdu_index=hdu_index,
                  hdr_small=hdr_small, plausibility=plaus,
                  zeroth=zeroth, first_res=first_res,
                  line_pts=(x_start, y_start, x_end, y_end),
                  dist=dist, xs=xs, ys=ys, flux=flux)

    print(f"Wrote outputs to: {fits_outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
