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

The watcher runs until you stop it (`Ctrl+C`) by design.

You can also run the watcher in Docker (watches continuously):
```bash
docker compose up -d watcher
```
Drop FITS files into `FITSfileDropFolder` and the watcher will continuously process them.
