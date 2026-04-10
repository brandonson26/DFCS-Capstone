# DFCS Capstone — Satellite Spectral Analysis Pipeline
**Guarding the Final Frontier**

An automated pipeline that processes FITS images from a ground-based diffraction-grating telescope. Drop a FITS file into a folder — the system automatically finds the spectral orders, extracts a 1D spectrum, runs quality checks, stores results in a database, and displays them on a web dashboard.

---

## Table of Contents
1. [Background Knowledge](#background-knowledge)
2. [System Requirements](#system-requirements)
3. [First-Time Setup](#first-time-setup)
4. [Running the Pipeline](#running-the-pipeline)
5. [How the Pipeline Works](#how-the-pipeline-works)
6. [Quality Flags](#quality-flags)
7. [Output Files](#output-files)
8. [Web Dashboard](#web-dashboard)
9. [TLE Catalog & Object Classification](#tle-catalog--object-classification)
10. [Project Structure](#project-structure)
11. [Database Management](#database-management)
12. [Environment Variables](#environment-variables)
13. [Troubleshooting](#troubleshooting)

---

## Background Knowledge

### What is a FITS file?
FITS (Flexible Image Transport System) is the standard file format used in astronomy. Each file contains a 2D image (the raw pixel data from the camera) plus a header with metadata such as exposure time, instrument name, object name, and observation date.

### What is a diffraction grating?
The telescope uses a diffraction grating — an optical element that splits incoming light into multiple "spectral orders":
- **Zeroth order** — the bright, undispersed source point (appears like a star or dot on the image)
- **First order** — the dispersed spectrum that appears above or below the zeroth order, spread out by wavelength like a rainbow

The goal of this pipeline is to find both of these automatically in every image and extract a 1D flux profile along the spectral axis.

### What is a TLE?
A TLE (Two-Line Element) set is a standardized format used to describe a satellite's orbital parameters. This pipeline uses a TLE catalog to identify whether a FITS file contains a satellite or a star, which affects how the data is processed and displayed.

---

## System Requirements

| Requirement | Notes |
|---|---|
| Docker | For running the containers |
| NVIDIA Container Toolkit | For GPU acceleration inside Docker |
| NVIDIA GPU | CUDA 12.x compatible (e.g. RTX 3500 Ada). Falls back to CPU if unavailable |
| ~4 GB disk space | For Docker images + database |

**Install Docker:** https://docs.docker.com/get-docker/

**Install NVIDIA Container Toolkit** (for GPU support):
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## First-Time Setup

### 1. Clone the repository
```bash
git clone https://github.com/brandonson26/DFCS-Capstone.git
cd DFCS-Capstone
```

### 2. Create your environment file
```bash
cp .env.example .env
```

The defaults in `.env.example` work out of the box. Only change them if you need custom database credentials.

### 3. Build the Docker images
```bash
docker compose build
```

This installs all Python dependencies (NumPy, SciPy, Astropy, CuPy, Flask, psycopg2, etc.) inside the containers. It may take a few minutes on first run.

---

## Running the Pipeline

### Start everything
```bash
docker compose up -d
```

This starts three services:
- **postgres** — the database
- **watcher** — the file watcher + processing pipeline (GPU-enabled)
- **webapp** — the web dashboard

### Process a FITS file
Copy any `.fit` or `.fits` file into the drop folder:
```bash
cp your_file.fits FITSfileDropFolder/
```

The watcher detects it automatically, waits for it to finish copying, then processes it. Results appear in `outputs_capstone/` and in the web dashboard within seconds.

### View results
Open [http://localhost:8000](http://localhost:8000) in your browser.

### Stop everything
```bash
docker compose down
```

### View live processing logs
```bash
docker logs -f capstone-watcher
```

---

## How the Pipeline Works

The pipeline runs fully automatically. Here is what happens from the moment a FITS file lands in `FITSfileDropFolder/`:

### Step 1 — File Detection (`IncomingFileEventHandler.py`)
The `watchdog` library monitors `FITSfileDropFolder/` for new `.fit` / `.fits` files. When one appears, a debounce mechanism waits until the file is fully written to disk (stable size and modification time), then submits it to a thread pool of up to 4 parallel workers.

### Step 2 — FITS Loading (`capstone.py`)
- Opens the file with `astropy.io.fits`
- Auto-selects the best image HDU (the largest 2D numeric array in the file)
- Extracts header metadata (exposure time, instrument, satellite name, object name)
- Runs a header plausibility check

### Step 3 — Object Classification (`tle_catalog.py`)
Reads the `OBJECT` field from the FITS header and compares it against `TLEs_202512231.txt`:

| Result | Meaning |
|---|---|
| `satellite` | Object name matches a satellite in the TLE catalog |
| `star` | Object name is present but not in the TLE catalog |
| `unknown` | No `OBJECT` header field present |

Matching is **fuzzy**: hyphens, spaces, and underscores are stripped before comparison, and common abbreviations (e.g. `DTV10` → DIRECTV-10) are resolved through a built-in alias table in `tle_catalog.py`. Classification drives all subsequent logic.

### Step 4 — Background Subtraction (`capstone.py`)
Divides the image into a 64×64 tile grid and takes the median of each tile (robust against bright sources). Bilinearly interpolates the tile medians back to full image resolution to create a smooth background map, then subtracts it. The result (`img_bgsub`) has approximately zero flux in empty sky regions and positive flux where real sources are.

### Step 5 — GPU Acceleration
A Gaussian blur is applied for noise reduction before source detection. If CuPy + CUDA are available, both the blur and the bilinear interpolation during spectrum extraction run on the GPU in VRAM. Automatically falls back to CPU (NumPy + SciPy) if no GPU is present.

### Step 6 — Zeroth Order Detection (`find_zeroth_order.py`)
Builds a **Summed Area Table** (integral image) so any rectangular sum over the image can be computed in O(1) time regardless of image size. Slides a box across the entire image scoring each position by:

```
score = total_flux / (1 + spatial_spread)
```

This scoring prefers positions that are both **bright and compact** — identifying the undispersed point source (zeroth order dot) rather than diffuse background. Returns a sub-pixel flux-weighted centroid `(zeroth.cx, zeroth.cy)`.

### Step 7 — First Order Detection (`find_first_order.py`)
Centers a 400-pixel wide strip on the zeroth order's x-position. Compares mean positive flux in the region **above** vs. **below** the zeroth order box to determine which side the first-order spectrum falls on (the brighter side wins). Runs the same compact-flux search in both regions with a smaller 21×21 window.

Returns:
- `first_pt` — the brighter first-order centroid (the chosen side)
- `above_point` / `below_point` — centroids on each side if detected

### Step 8 — Extraction Line Setup
The extraction line is set differently depending on object type:

**Satellites:**
The extraction is one-sided. The line runs from **200 pixels before the zeroth order** (in the opposite direction of the first order) all the way to the **image edge in the first-order direction**. This captures the zeroth order and everything between it and the image boundary where the first-order spectrum lives.

```
[image edge] ←── [200px before zeroth] ──→ [zeroth order] ──→ [first order] ──→ [image edge]
              ←────────────── extraction path ──────────────────────────────────────────────→
```

**Stars:**
Stars produce a symmetric spectrum (both sides of the zeroth order contain a first-order). The line runs **edge-to-edge through both first-order points**, with the zeroth order sitting between them. This captures the full spectrum on both sides.

### Step 9 — Spectrum Extraction (`capstone.py`)
Samples a 5-pixel-wide swath along the extraction line using bilinear interpolation (GPU-accelerated if available). At each step along the line, averages the flux across the 5-pixel width perpendicular to the line. Produces a 1D array of (distance, flux) — the extracted spectrum.

### Step 10 — Quality Checks
See [Quality Flags](#quality-flags) section below.

### Step 11 — Output Routing
Results are saved into folders named after the quality flags that were triggered:

```
outputs_capstone/
  good_data/<fits_stem>/
  star_streak/<fits_stem>/
  overexposure/<fits_stem>/
  partial_first_order/<fits_stem>/
  background_gradient/<fits_stem>/
  low_snr/<fits_stem>/
```

If multiple flags are true, the file is copied into **all** matching folders. Each folder contains the spectrum PNG and (for satellites) the partial first order diagnostic graph.

### Step 12 — Database Write (`db/db.py`)
Results are upserted into PostgreSQL using the file's SHA-256 hash as the unique key. Re-processing the same file updates the existing record rather than creating a duplicate.

Tables written:
- `files` — path, instrument, satellite name, `object_type`, quality status, FITS header subset
- `file_flags` — one row per quality flag with boolean result and full JSON diagnostics
- `runs` — output directory paths for this processing run

---

## Quality Flags

Each flag is stored in the database with a boolean result and a JSON `info` blob containing all diagnostic metrics for review.

### `star_streak` — Satellites only
Detects a star that crossed the field of view during the satellite's exposure, producing an unwanted streak in the image that contaminates the extracted spectrum.

**Detection method:** Finds all peaks in the smoothed 1D spectrum. The two brightest are classified as the zeroth order and first order. Any additional peak appearing **beyond** the first order along the extraction path is a streak candidate if it meets at least one criterion:
- Prominence > 70% of the first-order prominence
- Width < 250 pixels (narrow spike)
- Height > 1.2× the median background

**Effect on quality:** If `star_streak = True`, the file is assigned `quality_status = contaminated`. Stars are never flagged for star streak.

### `overexposure`
Detects saturated pixels in the zeroth-order region. An overexposed zeroth order causes charge bleeding that corrupts the surrounding spectrum.

### `partial_first_order` — Satellites only
Detects when the first-order spectrum is cut off at the image edge — the satellite's spectrum would continue beyond the detector boundary if the image were larger.

**Detection method:** At the far endpoint of the extraction path `(x_end, y_end)` — which sits at the image edge — a **200-pixel horizontal probe** is sampled: 100 pixels to the left and 100 pixels to the right. If any peak in this profile exceeds **3× the mean background** (200% above background), the spectrum is flagged as partial.

A **diagnostic graph** (`partial_first_order_probe.png`) is saved alongside the spectrum PNG in every output folder, showing:
- Blue line: the horizontal probe flux
- Orange line: the background mean
- Red dashed line: the detection threshold (3× background mean)

### `background_gradient` (`background_gradient.py`)
Detects a smooth brightness slope across the whole image caused by moonlight, scattered light, or atmospheric effects. A strong gradient means background subtraction may be imperfect.

### `low_snr` (`low_snr.py`)
Detects spectra where the signal-to-noise ratio is too low for reliable analysis. Uses four methods:

1. **Global SNR** — estimates baseline and noise from the lowest 10% of post-zeroth flux values. Flags if median SNR < 5 or 25th-percentile SNR < 2.
2. **Sustained signal** — checks for a consecutive run of at least 10 pixels above 3σ. Pure noise has no sustained runs.
3. **First-order window SNR** — computes SNR specifically in a ±50px window around the first-order centroid.
4. **Zeroth order SNR** *(primary trigger)* — if the peak flux at the zeroth order position is **less than 5× the background noise sigma**, the file is immediately flagged as low SNR. If the source itself is too faint, no further checks are needed.

---

## Output Files

Each processed file produces outputs in `outputs_capstone/<flag_folder>/<fits_stem>/`:

| File | Description |
|---|---|
| `03_spectrum_pixel.png` | 1D extracted spectrum plot (flux vs. distance along extraction path) |
| `partial_first_order_probe.png` | Horizontal edge probe diagnostic (satellites only) |

---

## Web Dashboard

Flask web app served at [http://localhost:8000](http://localhost:8000).

Results are grouped into three sections based on the `object_type` field from the database:

| Section | Contents |
|---|---|
| **Satellites** | Objects matched against the TLE catalog |
| **Stars** | Objects with a name not found in the TLE catalog |
| **Unknown** | Objects with no `OBJECT` header field |

Each row shows:
- Spectrum PNG thumbnail
- Quality status (`useable` or `contaminated`)
- All quality flags
- A detail page with full header metadata and JSON flag diagnostics

---

## TLE Catalog & Object Classification

The file `TLEs_202512231.txt` contains TLE sets for all satellites the telescope observes. It is read once at container startup.

**Current satellites in the catalog (17):**
GALAXY-3C, DIRECTV-10, DIRECTV-11, DIRECTV-12, DIRECTV-14, DIRECTV-15, SKYTERRA-1, SES-3, SES-20, SES-18, ECHOSTAR-10, ECHOSTAR-11, ANIK-F2, AMC-11, WILDBLUE-1, MEXSAT-3, AT&T-T16

### Adding a new satellite
Append its 3-line TLE block (name line + two data lines) to `TLEs_202512231.txt`, then rebuild:
```bash
docker compose build watcher
docker compose up -d watcher
```

### Adding a header name alias
If a FITS file uses an abbreviated name (e.g. `DTV10` instead of `DIRECTV-10`) that doesn't fuzzy-match the catalog, add it to `_ALIASES` in `tle_catalog.py`:
```python
_ALIASES: FrozenSet[str] = frozenset(_normalize(a) for a in [
    "DTV10", "DTV11", ...   # add your abbreviation here
])
```
Then rebuild the watcher container.

---

## Project Structure

```
DFCS-Capstone/
├── capstone.py                   # Main pipeline orchestrator
├── find_zeroth_order.py          # Zeroth order detection (integral image + compact flux)
├── find_first_order.py           # First order detection (above/below search boxes)
├── tle_catalog.py                # TLE catalog parser — satellite/star/unknown classification
├── TLEs_202512231.txt            # TLE catalog of known satellites
├── background_gradient.py        # Background gradient quality check
├── low_snr.py                    # Low SNR quality check (4 methods)
├── overexposure.py               # Overexposure quality check
├── partial_first_order.py        # Partial first order quality check + diagnostic graph
├── star_streak.py                # Star streak quality check (satellites only)
├── IncomingFileEventHandler.py   # Folder watcher entry point + thread pool
├── ImagesWatcher.py              # Filesystem event handler (watchdog)
├── db/
│   ├── db.py                     # PostgreSQL connection + upsert logic
│   ├── migrate.py                # Schema migration runner
│   ├── schema.sql                # Database table definitions
│   └── webapp.py                 # Flask web dashboard
├── Dockerfile                    # Watcher/pipeline container (CUDA 12.4.1)
├── Dockerfile.webapp             # Web dashboard container (python:3.12-slim)
├── docker-compose.yml            # Orchestrates postgres + watcher + webapp
├── requirements.txt              # Python package dependencies
├── .env.example                  # Template for database credentials
├── FITSfileDropFolder/           # Drop FITS files here to trigger processing
├── outputs_capstone/             # Generated PNGs (organized by quality flag)
└── Data/                         # Test FITS files
```

---

## Database Management

### Apply schema updates
After pulling new code changes, run migrations (safe — uses `IF NOT EXISTS`, will not delete data):
```bash
docker compose exec watcher python3 db/migrate.py
```

### Clear all data (keeps schema)
```bash
docker compose exec postgres psql -U capstone_user -d capstone_db \
  -c "TRUNCATE files, file_flags, runs RESTART IDENTITY CASCADE;"
```

### Full reset (deletes all data and volumes)
```bash
docker compose down -v
docker compose up -d
```

### Schema overview

| Table | Key columns |
|---|---|
| `files` | `path`, `sha256`, `object_type`, `quality_status`, `hdr_small` (JSON) |
| `file_flags` | `file_id`, `flag_id`, `value` (bool), `info` (JSON diagnostics) |
| `runs` | `file_id`, `outdir`, `run_name`, `dest_dirs` |

---

## Environment Variables

Copy `.env.example` to `.env`. The defaults work out of the box for local development:

| Variable | Default | Description |
|---|---|---|
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `capstone_db` | Database name |
| `DB_USER` | `capstone_user` | Database user |
| `DB_PASSWORD` | `capstone_pass` | Database password |

---

## Troubleshooting

### FITS file is not being processed
- Check that the file has a `.fit` or `.fits` extension
- Check `file_watcher.log` in the project root for error messages
- Run `docker logs capstone-watcher` for container-level errors

### `could not select device driver "nvidia"`
The NVIDIA Container Toolkit is not installed or the Docker daemon was not restarted after installation. Follow the installation steps in [System Requirements](#system-requirements) and run `sudo systemctl restart docker`.

### `password authentication failed` after recreating containers
The PostgreSQL volume has stale credentials. Run:
```bash
docker compose down -v && docker compose up -d
```

### Web dashboard shows no results
The watcher may still be processing. Wait a few seconds and refresh. Check `file_watcher.log` for errors.

### GPU not being used
Run `docker logs capstone-watcher | grep -i backend` to confirm the compute backend. If it shows `cpu`, either no GPU is available or the NVIDIA Container Toolkit is not configured.
