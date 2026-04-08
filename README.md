# DFCS-Capstone
Guarding the Final Frontier

## Database Web Dashboard

Run a local web interface to view the PostgreSQL database tables.

1. Copy environment file:
```bash
cp .env.example .env
```
2. Start PostgreSQL in Docker:
```bash
docker compose up -d
```
3. The dashboard is now available from the same compose stack:
```bash
docker compose up -d
```
4. Open `http://localhost:8000`.

Notes:
- The first webapp boot may install Python dependencies inside the container.
- Use `http://localhost:8000` only after `docker compose up -d` reports both `postgres` and `webapp` as healthy/running.
- If you prefer to run the app directly on host (without Docker), run:
```bash
pip install flask psycopg2-binary python-dotenv
python3 db/webapp.py
```
 - `db/schema.sql` is auto-applied on first Postgres init.
- If you already had a Postgres volume from older runs, apply schema updates with:
```bash
python3 db/migrate.py
```
- If you changed DB password/user and get `password authentication failed`, reset the persisted volume:
```bash
docker compose down -v
docker compose up -d
```
- Ensure the values in `.env` match `POSTGRES_*` credentials used by the running container.

## Watch Folder Pipeline

Run this command to process files dropped into `FITSfileDropFolder` and write each file’s
flags/header metadata to PostgreSQL automatically:

```bash
python3 IncomingFileEventHandler.py --watch-path FITSfileDropFolder --outdir outputs_capstone
```

Optional:
- add `--scan-existing` to process any FITS files already in the folder when the watcher starts.
- add `--no-star-streak` to skip star streak detection.
- star-streak classification now runs only when header target looks satellite-related (`DTV10` / `DirecTV`) and is skipped for star target (`HD128998`).
- Dashboard `/` now shows a `Spectrum PNG` thumbnail column and a `Download FITS` link in the recent files table.

The watcher runs until you stop it (`Ctrl+C`) by design.

You can also run the watcher in Docker (watches continuously):
```bash
docker compose up -d watcher
```
Drop FITS files into `FITSfileDropFolder` and the watcher will continuously process them.

__________________________________________________________________________________________________________________________________________________________
Full Pipeline Overview
The system is a fully automated satellite spectral analysis pipeline. When a FITS file is dropped into a folder, it automatically detects spectral orders, runs quality checks, and stores results in a database. Here's every component explained:

The Data: FITS Files
FITS (Flexible Image Transport System) is the standard file format for astronomical images. Each file contains:

HDU (Header Data Unit) — the image data (a 2D pixel array)
Header — metadata like exposure time, instrument name, satellite name, object name
The images come from a ground-based telescope observing a satellite (DTV10/DirecTV). Because the telescope uses a diffraction grating, the light is split into spectral "orders":

Zeroth order — the bright, undispersed source point (like a dot)
First order — the dispersed spectrum above or below the zeroth order (like a rainbow streak)
Stage 1: File Watcher → IncomingFileEventHandler.py
This is the entry point when you run the pipeline. You start it with:


python3 IncomingFileEventHandler.py --watch-path FITSfileDropFolder --outdir outputs_capstone --device gpu
What it does:

FolderWatcher uses the watchdog library to watch the FITSfileDropFolder directory for new files.
When a .fit or .fits file appears, it first waits for the file to stabilize — it checks that the file size and modification time haven't changed for 3 consecutive checks, 1 second apart. This prevents processing half-uploaded files.
Once stable, it passes the file to a ThreadPoolExecutor (up to 4 workers in parallel) so multiple files can be processed simultaneously.
The _capstone_processor function inside builds an argparse.Namespace (a fake set of command-line arguments) and calls process_one_file() from capstone.py.
A debounce mechanism using _inflight set prevents the same file from being submitted twice if the filesystem fires multiple events.
All activity is logged to file_watcher.log (rotated daily, keeping 14 days of history).
Stage 2: Main Processing → capstone.py
This is the brain. process_one_file() does the following in order:

2a. Load the FITS File
Opens the file with astropy.io.fits
Automatically selects the best HDU (biggest 2D numeric array with exposure time)
Extracts the header metadata (exposure time, gain, satellite name, object name, etc.)
Runs header plausibility checks — warns if exposure time is missing, zero, or non-numeric
2b. Configure GPU or CPU
configure_compute_backend() checks if CuPy is installed and a CUDA GPU is available
If --device gpu, forces GPU (crashes if unavailable)
If --device auto, tries GPU and falls back to CPU silently
Sets global variables: XP (either numpy or cupy), ND_GAUSSIAN_FILTER, ND_MAP_COORDINATES
2c. Background Subtraction
detect_background_gradient() is called first on the raw image to check if moonlight or stray light is making the background uneven
estimate_background() divides the image into a 64×64 tile grid, takes the median of each tile (resistant to bright sources), then bilinearly interpolates back to full image size — this gives a smooth background map
img_bgsub = img_raw - bg — subtracts it, leaving only astronomical signal
2d. Smoothing for Detection
Applies a Gaussian blur (default σ=1.0 pixel) to img_bgsub to reduce pixel noise before searching
On GPU: cupyx.scipy.ndimage.gaussian_filter — runs on VRAM
On CPU: scipy.ndimage.gaussian_filter
Result is img_detect — used for finding sources; img_bgsub is used for flux measurement
Stage 3: Zeroth Order Detection → find_zeroth_order.py
Goal: Find the bright, compact, undispersed dot (zeroth order) in the image.

How it works:

Builds a Summed Area Table (integral image) — a precomputed lookup table where any rectangular sum can be computed in O(1) time with just 4 table lookups. This is extremely fast compared to looping.
Slides a 100×100 pixel box across the entire image (every 4 pixels = step), computing the total positive flux in each position
Uses compact_flux scoring (default): score = total_flux / (1 + spatial_spread) — this prefers a box that has high total flux AND is spatially concentrated (small spread = compact source). A spread-out faint haze would score lower than a bright compact point.
Finds the box with the highest score
Computes the flux-weighted centroid inside that box — sub-pixel precision
Returns a BoxResult with the box corners and centroid (cx, cy)
Stage 4: First Order Detection → find_first_order.py
Goal: Find the dispersed first-order spectrum, which appears either above or below the zeroth order.

How it works:

Centers a 400-pixel wide strip on the zeroth order's x-position
Splits the image into two search regions: above the zeroth order box and below the zeroth order box (with 5-pixel padding so the bright zeroth source doesn't bleed in)
Computes the mean positive flux in each region — whichever has more flux is assumed to contain the first order
In the winning region (and both regions), runs _find_compact_in_bounds() — the same integral image + compact_flux scoring but with a smaller 21×21 pixel window — to find the compact "brightest point" of the first-order spectrum
Returns a FirstOrderResult with: which direction ("above" or "below"), the compact point, and whether both above and below points were found
Stage 5: Extraction Line
Once both orders are located, the pipeline draws an extraction line connecting them:

If both above and below first-order points were found: line_through_image_edges() draws a line through both points extended to both image edges — a full edge-to-edge line
If only one was found: extend_line_to_image_edge() goes from a point slightly before the zeroth order all the way to the opposite edge
sample_line_profile() then samples pixel values along this line:

Uses map_coordinates (bilinear interpolation) to read sub-pixel values at each step
Samples a 5-pixel-wide swath perpendicular to the line and averages across the width (like DS9's Plot2D feature)
On GPU: cupyx.scipy.ndimage.map_coordinates — interpolation done in VRAM
The result is a 1D spectrum: distance along the line vs. flux.

Stage 6: Quality Flags
Five independent quality checks run on every file:

Flag	File	What it checks
star_streak	star_streak.py	Multiple sharp peaks in the 1D profile — a satellite streak crossing the field
overexposure	overexposure.py	Saturated pixels in the zeroth order region
partial_first_order	partial_first_order.py	Flux drops at the end of the spectrum — first order is cut off by the image edge
background_gradient	background_gradient.py	Smooth brightness slope across the whole image (moonlight, scattered light)
low_snr	low_snr.py	Signal-to-noise ratio below threshold in the extracted spectrum
Background gradient works by:

Masking bright sources (stars/satellite)
Fitting a tilted plane to the background using IRLS (Iteratively Reweighted Least Squares — robust to outliers)
Checking if the plane tilt is large relative to noise and background level
Low SNR works by:

Finding the baseline + noise from the lowest 10% of flux values after the zeroth order
Computing SNR = (signal - baseline) / noise for each pixel
Flagging if median SNR < 5 or 25th-percentile SNR < 2
Stage 7: Output Routing
Based on quality flags, the spectrum PNG is copied into multiple output folders:


outputs_capstone/
  good_data/<stem>/        ← no flags (or star_streak=False)
  star_streak/<stem>/      ← if streak detected
  overexposure/<stem>/     ← if saturated
  partial_first_order/<stem>/
  background_gradient/<stem>/
  low_snr/<stem>/
A file can land in multiple folders simultaneously if it has multiple flags. This lets you quickly sort through what's wrong with each observation.

Stage 8: Database Write → db/db.py
Every result is written to a PostgreSQL database (running in Docker):

SHA-256 hash of the FITS file is computed — used as the unique identifier
Upsert into files table (ON CONFLICT (sha256) DO UPDATE) — if the same file is dropped again, it replaces the old record instead of duplicating
Upsert into file_flags table — one row per flag per file, with the boolean value and diagnostic metrics (e.g., the SNR numbers, gradient strength)
Insert into runs table — records which output directories were used
Connection parameters come from the .env file (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).

Stage 9: Web Dashboard → db/webapp.py
A Flask web app that reads from the database and displays results — lets you browse processed files, their quality flags, and spectrum images through a browser.

Data Flow Summary

FITSfileDropFolder/
  └── new_file.fits
         ↓ (watchdog detects it)
  IncomingFileEventHandler.py
    └── waits for file stability
    └── submits to thread pool
         ↓
  capstone.py :: process_one_file()
    ├── Load FITS → raw image
    ├── Background gradient check
    ├── Subtract background
    ├── GPU Gaussian smooth
    ├── find_zeroth_order()  → (cx, cy) of bright point
    ├── find_first_order()   → (cx, cy) of spectrum point + direction
    ├── Draw extraction line edge-to-edge
    ├── GPU sample_line_profile() → 1D spectrum
    ├── Quality checks: streak, overexposure, partial, gradient, SNR
    ├── Save spectrum PNG
    ├── Copy PNG to output bucket folders
    └── Write to PostgreSQL DB (upsert)