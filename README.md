# DFCS Capstone — Satellite Spectral Analysis Pipeline
**Guarding the Final Frontier**

An automated pipeline that detects and extracts spectra from FITS images of satellites observed through a diffraction grating telescope. Drop a FITS file into a folder and the system automatically finds the spectral orders, runs quality checks, stores results in a database, and displays them on a web dashboard.

---

## How It Works

The telescope uses a diffraction grating that splits a satellite's light into spectral "orders":
- **Zeroth order** — the bright, undispersed source point (like a dot)
- **First order** — the dispersed spectrum that appears above or below the zeroth order

The pipeline finds both, draws an extraction line between them edge-to-edge across the image, samples the flux along that line, and produces a 1D spectrum — similar to DS9's Plot2D function.

---

## Quick Start

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed (for GPU acceleration)
- An NVIDIA GPU (pipeline falls back to CPU if unavailable)

### Setup (one time)

**1. Clone the repository:**
```bash
git clone https://github.com/brandonson26/DFCS-Capstone.git
cd DFCS-Capstone
```

**2. Create your environment file:**
```bash
cp .env.example .env
```

**3. Build the Docker images:**
```bash
docker compose build
```

### Run

**Start everything (database + pipeline watcher + web dashboard):**
```bash
docker compose up -d
```

**Drop a FITS file into the watch folder:**
```bash
cp your_file.fits FITSfileDropFolder/
```
The pipeline automatically detects it, processes it with GPU acceleration, and writes results to the database.

**View results:**
Open [http://localhost:8000](http://localhost:8000) in your browser.

**Stop everything:**
```bash
docker compose down
```

---

## Project Structure

```
DFCS-Capstone/
├── capstone.py                   # Main pipeline: load, detect, extract, flag, route
├── find_zeroth_order.py          # Zeroth order detection (integral image + compact flux scoring)
├── find_first_order.py           # First order detection (above/below search boxes)
├── tle_catalog.py                # TLE catalog parser — classifies objects as satellite/star/unknown
├── TLEs_202512231.txt            # TLE catalog of known satellites
├── background_gradient.py        # Background gradient quality flag
├── low_snr.py                    # Low signal-to-noise quality flag
├── overexposure.py               # Overexposure quality flag
├── partial_first_order.py        # Partial first order quality flag
├── star_streak.py                # Star streak quality flag (satellites only)
├── ImagesWatcher.py              # Filesystem event handler (watchdog)
├── IncomingFileEventHandler.py   # Folder watcher entry point + thread pool
├── db/
│   ├── db.py                     # PostgreSQL connection + upsert logic
│   ├── migrate.py                # Schema migration runner
│   ├── schema.sql                # Database table definitions
│   └── webapp.py                 # Flask web dashboard (Satellites / Stars sections)
├── Dockerfile                    # Watcher/pipeline container (CUDA-enabled)
├── Dockerfile.webapp             # Web dashboard container (lightweight)
├── docker-compose.yml            # Orchestrates postgres + watcher + webapp
├── requirements.txt              # Python dependencies
├── .env.example                  # Template for database credentials
├── FITSfileDropFolder/           # Drop FITS files here to trigger processing
├── outputs_capstone/             # Generated spectrum PNGs (organized by quality flag)
└── Data/                         # Test FITS files
```

---

## Pipeline Stages

### 1. File Watcher — `IncomingFileEventHandler.py`
Watches `FITSfileDropFolder/` using the `watchdog` library. When a `.fit` or `.fits` file appears, it waits for the file to finish copying (stable size + modification time), then submits it to a thread pool (up to 4 parallel workers). A debounce mechanism prevents duplicate processing.

### 2. FITS Loading & Object Classification — `capstone.py` + `tle_catalog.py`
Opens the file with `astropy.io.fits`, auto-selects the best image HDU (largest 2D numeric array), and extracts header metadata (exposure time, instrument, satellite name, object name). Runs header plausibility checks.

The `OBJECT` field from the header is immediately classified against `TLEs_202512231.txt`:

| Result | Meaning |
|---|---|
| `satellite` | Object name matches an entry in the TLE catalog |
| `star` | Object name is present but not in the TLE catalog |
| `unknown` | No `OBJECT` header field present |

Matching is fuzzy — hyphens, spaces, and underscores are stripped before comparison, and common abbreviations (e.g. `DTV10` → DIRECTV-10) are resolved via a built-in alias table. This classification flows through the entire pipeline: it controls which quality checks run, what quality status is assigned, and which section of the database and web dashboard the result appears under.

### 3. Background Subtraction
Divides the image into a 64×64 tile grid, takes the median of each tile (resistant to bright sources), bilinearly interpolates back to full resolution, and subtracts it. This removes sky background and sensor gradients leaving only the astronomical signal.

### 4. GPU Acceleration
Applies a Gaussian blur for noise reduction before source detection. On GPU (CuPy + CUDA), both the blur and the spectrum extraction interpolation run in VRAM. Falls back to CPU (NumPy + SciPy) automatically if no GPU is available.

### 5. Zeroth Order Detection — `find_zeroth_order.py`
Builds a **Summed Area Table** (integral image) so any rectangular sum can be computed in O(1) time. Slides a 100×100 pixel box across the entire image, scoring each position by:

```
score = total_flux / (1 + spatial_spread)
```

This prefers boxes that are both bright and compact — picking the undispersed point source over faint diffuse backgrounds. Returns a sub-pixel flux-weighted centroid.

### 6. First Order Detection — `find_first_order.py`
Centers a 400-pixel wide strip on the zeroth order's x-position. Compares mean flux in the region above vs. below the zeroth order box to determine dispersion direction. Runs a compact-flux search in both regions using the same integral image technique with a smaller 21×21 window. Returns the brightest compact point in each region.

### 7. Spectrum Extraction
Draws an extraction line edge-to-edge across the image:
- If both above and below first-order points are found: line runs through both, with zeroth order in between
- If only one is found: line runs from the opposite image edge through zeroth to the detected first-order edge

Samples a 5-pixel-wide swath perpendicular to the line using bilinear interpolation (GPU-accelerated), averaging across the width. Produces a 1D flux profile: distance along line vs. flux.

### 8. Quality Flags
Checks run depending on object type:

| Flag | Satellites | Stars | What it checks |
|---|---|---|---|
| `star_streak` | Yes | No | Multiple sharp peaks in the 1D profile — a star crossed the field during exposure |
| `overexposure` | Yes | Yes | Saturated pixels in the zeroth order region |
| `partial_first_order` | Yes | Yes | Flux drops at the end of the spectrum — first order cut off by the image edge |
| `background_gradient` | Yes | Yes | Smooth brightness slope across the whole image (moonlight, scattered light) |
| `low_snr` | Yes | Yes | Signal-to-noise ratio below threshold in the extracted spectrum |

Star streak is skipped for stars because stars do not produce satellite streaks. As a result, stars are never assigned a `contaminated` quality status — only satellites can be contaminated.

### 9. Output Routing
The spectrum PNG is copied into folders named by quality flag:

```
outputs_capstone/
  good_data/<stem>/
  star_streak/<stem>/
  overexposure/<stem>/
  partial_first_order/<stem>/
  background_gradient/<stem>/
  low_snr/<stem>/
```

A file with multiple flags is copied into all matching folders simultaneously.

### 10. Database Write — `db/db.py`
Every result is upserted into PostgreSQL:
- SHA-256 hash of the FITS file used as unique key — re-processing the same file updates the existing record
- `files` table — path, instrument, satellite, `object_type`, quality status, header metadata
- `file_flags` table — one row per flag with boolean value and diagnostic metrics
- `runs` table — output directory locations for each processing run

### 11. Web Dashboard — `db/webapp.py`
Flask web app at [http://localhost:8000](http://localhost:8000). Results are displayed in two separate sections driven by the `object_type` field in the database:

- **Satellites** — objects matched against the TLE catalog
- **Stars** — objects with a name not found in the TLE catalog
- **Unknown** — objects with no `OBJECT` header

Each row shows the spectrum PNG thumbnail, quality status, a download link for the original FITS file, and a detail page with all flags and header metadata.

---

## TLE Catalog & Object Classification

The file `TLEs_202512231.txt` contains the Two-Line Element (TLE) set for every satellite the telescope is tasked to observe. The pipeline reads this file at startup and uses it to automatically classify every incoming FITS file.

**Current satellites in the catalog (17):**
GALAXY-3C, DIRECTV-10, DIRECTV-11, DIRECTV-12, DIRECTV-14, DIRECTV-15, SKYTERRA-1, SES-3, SES-20, SES-18, ECHOSTAR-10, ECHOSTAR-11, ANIK-F2, AMC-11, WILDBLUE-1, MEXSAT-3, AT&T-T16

**Adding a new satellite:**
Add its 3-line TLE block (name + two data lines) anywhere in `TLEs_202512231.txt` and rebuild the Docker image:
```bash
docker compose build watcher
docker compose up -d watcher
```

**Adding a header abbreviation alias:**
If a FITS file uses a short name (e.g. `DTV10` instead of `DIRECTV-10`) that doesn't fuzzy-match the TLE catalog, add it to the `_ALIASES` set in `tle_catalog.py`:
```python
_ALIASES: FrozenSet[str] = frozenset(_normalize(a) for a in [
    "DTV10", "DTV11", ...   # add your abbreviation here
])
```

---

## Docker Services

| Service | Image | Purpose |
|---|---|---|
| `postgres` | `postgres:16` | Database |
| `watcher` | `capstone-watcher` (CUDA) | File watcher + pipeline processing |
| `webapp` | `capstone-webapp` | Web dashboard |

The watcher container is built from `Dockerfile` using `nvidia/cuda:12.4-runtime-ubuntu22.04` as the base image so CuPy GPU acceleration works inside Docker. GPU access requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on the host.

---

## Environment Variables

Copy `.env.example` to `.env` and set your values:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=capstone_db
DB_USER=capstone_user
DB_PASSWORD=capstone_pass
```

---

## Database Management

**Apply schema updates** after pulling new changes (safe — uses `IF NOT EXISTS`, will not delete data):
```bash
docker compose exec watcher python3 db/migrate.py
```

**Reset the database** (warning — deletes all data):
```bash
docker compose down -v
docker compose up -d
```

If you get `password authentication failed` after a compose recreate, reset the volume:
```bash
docker compose down -v && docker compose up -d
```

**Schema overview:**

| Table | Key columns |
|---|---|
| `files` | `path`, `sha256`, `object_type`, `quality_status`, `hdr_small` |
| `file_flags` | `file_id`, `flag_id`, `value`, `info` (JSON diagnostics) |
| `runs` | `file_id`, `outdir`, `run_name`, `dest_dirs` |
