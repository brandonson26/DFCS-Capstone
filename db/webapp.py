import json
import os
from pathlib import Path

from flask import Flask
from flask import abort
from flask import jsonify
from flask import render_template_string
from flask import send_file

from db import get_conn

app = Flask(__name__)

DEFAULT_OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "/app")
SPECTRUM_IMAGE_NAME = "03_spectrum_pixel.png"
SECTION_LABELS = [
    ("satellite", "Satellites"),
    ("star", "Stars"),
    ("unknown", "Unknown"),
]

TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Capstone DB</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: ui-sans-serif, system-ui, Arial; margin: 20px; color: #0f172a; }
    table { border-collapse: collapse; width: 100%; max-width: 1300px; }
    th, td { border: 1px solid #d1d5db; padding: 6px; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; max-width: 1300px; }
    .thumb { max-width: 160px; max-height: 90px; border: 1px solid #cbd5e1; display: block; }
  </style>
</head>
<body>
  <h1>Capstone DB Dashboard</h1>
  <div class="card">
    {% for section in sections %}
    <h3>{{ section.label }} ({{ section.rows|length }})</h3>
    <table>
      <tr><th>Path</th><th>Spectrum PNG</th><th>Status</th><th>Download FITS</th><th>More Info</th></tr>
      {% for row in section.rows %}
      <tr>
        <td>{{ row.path }}</td>
        <td>{% if row.png_url %}<a href="{{ row.png_url }}" target="_blank"><img src="{{ row.png_url }}" class="thumb" alt="spectrum"></a>{% else %}n/a{% endif %}</td>
        <td>{{ row.quality_status }}</td>
        <td>{% if row.download_url %}<a href="{{ row.download_url }}">download</a>{% else %}n/a{% endif %}</td>
        <td><a href="/files/{{ row.id }}">more info</a></td>
      </tr>
      {% endfor %}
    </table>
    {% endfor %}
  </div>
  <div class="card">
    <p>Health: {{ db_ok }}</p>
  </div>
</body>
</html>
"""


FILE_DETAIL_TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>File {{ file.id }} - Details</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: ui-sans-serif, system-ui, Arial; margin: 20px; color: #0f172a; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; max-width: 1300px; }
    table { border-collapse: collapse; width: 100%; max-width: 1300px; }
    th, td { border: 1px solid #d1d5db; padding: 6px; text-align: left; vertical-align: top; }
    th { background: #f3f4f6; width: 180px; }
    pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
    .small { color: #4b5563; font-size: 13px; margin-top: 4px; }
  </style>
</head>
<body>
  <h1>File Detail</h1>
  <p class="small"><a href="/">← Back to dashboard</a></p>
  <div class="card">
    <h3>File Metadata</h3>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>ID</td><td>{{ file.id }}</td></tr>
      <tr><td>Path</td><td>{{ file.path }}</td></tr>
      <tr><td>SHA256</td><td>{{ file.sha256 }}</td></tr>
      <tr><td>HDU Index</td><td>{{ file.hdu_index }}</td></tr>
      <tr><td>Instrument</td><td>{{ file.instrument or '' }}</td></tr>
      <tr><td>Satellite</td><td>{{ file.satellite or '' }}</td></tr>
      <tr><td>Object Type</td><td>{{ file.object_type or 'unknown' }}</td></tr>
      <tr><td>Quality Status</td><td>{{ file.quality_status }}</td></tr>
      <tr><td>Ingested At</td><td>{{ file.ingested_at }}</td></tr>
      <tr><td>Header (hdr_small)</td><td><pre>{{ file.hdr_small }}</pre></td></tr>
      <tr>
        <td>Actions</td>
        <td>
          <a href="{{ file.download_url }}">Download FITS</a>
          {% if file.png_url %} | <a href="{{ file.png_url }}" target="_blank">View Spectrum PNG</a>{% endif %}
        </td>
      </tr>
    </table>
  </div>
  <div class="card">
    <h3>Flags</h3>
    {% if flags %}
    <table>
      <tr><th>Flag</th><th>Value</th><th>Info</th></tr>
      {% for item in flags %}
      <tr>
        <td>{{ item.flag }}</td>
        <td>{{ item.value }}</td>
        <td><pre>{{ item.info }}</pre></td>
      </tr>
      {% endfor %}
    </table>
    {% else %}
    <p>No flags were recorded for this file.</p>
    {% endif %}
  </div>
</body>
</html>
"""


@app.get("/health")
def health():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return jsonify({"ok": True})
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500



def table_count(cur, name: str) -> int:
    cur.execute(f"SELECT COUNT(*) FROM {name}")
    return int(cur.fetchone()[0])


def _parse_dest_dirs(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed if isinstance(x, str)]
        except json.JSONDecodeError:
            pass
    return []


def _latest_run_dest_dirs(file_id: int) -> list[str]:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.dest_dirs
                FROM runs r
                WHERE r.file_id = %s
                ORDER BY r.id DESC
                LIMIT 1
                """,
                (file_id,),
            )
            row = cur.fetchone()

    if not row:
        return []
    return _parse_dest_dirs(row[0])


def _candidate_search_dirs() -> list[Path]:
    return [
        Path("/app"),
        Path("/app/FITSfileDropFolder"),
        Path("/app/outputs"),
    ]


def _find_png_in_runs(dest_dirs: list[str]) -> str | None:
    if not dest_dirs:
        return None

    for dest in dest_dirs:
        if not dest:
            continue

        d_path = Path(dest)
        if not d_path.is_absolute():
            d_path = Path(DEFAULT_OUTPUT_ROOT) / d_path

        candidate = d_path / SPECTRUM_IMAGE_NAME
        if candidate.exists():
            return str(candidate)

    return None


def _find_fits_path(raw_path: str | None) -> Path | None:
    if not raw_path:
        return None

    p = Path(raw_path)
    if p.exists():
        return p

    # Fallback for host/container path mismatches: locate by filename in mounted folders.
    for base in _candidate_search_dirs():
        if not base.exists():
            continue
        matches = list(base.rglob(p.name))
        if matches:
            return matches[0]

    return None


def _build_row_dict(file_row: tuple) -> dict:
    file_id, path, quality_status, instrument, satellite, object_type, ingested_at, sha256 = file_row
    dest_dirs = _latest_run_dest_dirs(file_id)
    png_abs_path = _find_png_in_runs(dest_dirs)

    return {
        "id": file_id,
        "path": path,
        "quality_status": quality_status,
        "instrument": instrument,
        "satellite": satellite,
        "object_type": object_type or "unknown",
        "ingested_at": str(ingested_at),
        "sha256": sha256,
        "png_url": f"/file-plot/{file_id}" if png_abs_path else None,
        "download_url": f"/file-download/{file_id}",
    }


def _get_file_by_id(file_id: int) -> dict | None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, path, sha256, hdu_index, instrument, satellite,
                       object_type, quality_status, ingested_at, hdr_small
                FROM files
                WHERE id = %s
                """,
                (file_id,),
            )
            row = cur.fetchone()

    if not row:
        return None

    file_id, path, sha256, hdu_index, instrument, satellite, object_type, quality_status, ingested_at, hdr_small = row
    dest_dirs = _latest_run_dest_dirs(file_id)
    png_abs_path = _find_png_in_runs(dest_dirs)

    return {
        "id": file_id,
        "path": path,
        "sha256": sha256,
        "hdu_index": hdu_index,
        "instrument": instrument,
        "satellite": satellite,
        "object_type": object_type or "unknown",
        "quality_status": quality_status,
        "ingested_at": str(ingested_at),
        "hdr_small": json.dumps(hdr_small, indent=2, sort_keys=True) if hdr_small is not None else "{}",
        "png_url": f"/file-plot/{file_id}" if png_abs_path else None,
        "download_url": f"/file-download/{file_id}",
    }


def _latest_files(limit: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, path, quality_status, instrument, satellite, object_type, ingested_at, sha256
                FROM files
                ORDER BY ingested_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [_build_row_dict(r) for r in rows]


def recent_sections(limit: int = 25):
    rows = _latest_files(limit)
    grouped = {key: [] for key, _ in SECTION_LABELS}
    for row in rows:
        obj_type = row.get("object_type", "unknown")
        if obj_type not in grouped:
            obj_type = "unknown"
        grouped[obj_type].append(row)

    return [
        {"label": label, "rows": grouped[key]}
        for key, label in SECTION_LABELS
    ]


@app.get("/")
def index():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                counts = {
                    "files": table_count(cur, "files"),
                    "flags": table_count(cur, "flags"),
                    "runs": table_count(cur, "runs"),
                }
        db_ok = "connected"
    except Exception as exc:
        counts = {"files": "n/a", "flags": "n/a", "runs": "n/a"}
        db_ok = f"disconnected: {exc}"

    return render_template_string(
        TEMPLATE,
        counts=counts,
        sections=recent_sections(25),
        db_ok=db_ok,
    )


@app.get("/file-plot/<int:file_id>")
def file_plot(file_id: int):
    dest_dirs = _latest_run_dest_dirs(file_id)
    png_abs_path = _find_png_in_runs(dest_dirs)
    if not png_abs_path:
        return jsonify({"error": "PNG not found for file"}), 404

    return send_file(png_abs_path, mimetype="image/png")


@app.get("/file-download/<int:file_id>")
def file_download(file_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT path FROM files WHERE id = %s", (file_id,))
            row = cur.fetchone()

    if not row:
        abort(404, description="File not found in DB")

    fits_path = _find_fits_path(row[0])
    if not fits_path:
        abort(404, description="File not accessible from web container")

    return send_file(
        str(fits_path),
        as_attachment=True,
        download_name=fits_path.name,
    )


@app.get("/files")
def files_page():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, path, quality_status, instrument, satellite, object_type, ingested_at, sha256 FROM files ORDER BY ingested_at DESC LIMIT 500"
                )
                rows = cur.fetchall()
        return jsonify([_build_row_dict(r) for r in rows])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/files/<int:file_id>")
def file_detail_page(file_id: int):
    file_row = _get_file_by_id(file_id)
    if not file_row:
        abort(404, description="File not found")

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT f.name, ff.value, ff.info
                    FROM file_flags ff
                    JOIN flags f ON f.id = ff.flag_id
                    WHERE ff.file_id = %s
                    ORDER BY f.name
                    """,
                    (file_id,),
                )
                flag_rows = cur.fetchall()
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    flags = [
        {"flag": name, "value": bool(value), "info": json.dumps(info, indent=2, sort_keys=True) if info is not None else "{}"}
        for name, value, info in flag_rows
    ]

    return render_template_string(
        FILE_DETAIL_TEMPLATE,
        file=file_row,
        flags=flags,
    )


@app.get("/file_flags")
def file_flags_page():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ff.file_id, f.name, ff.value, ff.info
                    FROM file_flags ff
                    JOIN flags f ON f.id = ff.flag_id
                    ORDER BY ff.file_id DESC, f.name
                    """
                )
                rows = cur.fetchall()
        return jsonify([
            {"file_id": r[0], "flag": r[1], "value": bool(r[2]), "info": r[3]}
            for r in rows
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    host = os.getenv("WEB_HOST", "0.0.0.0")
    port = int(os.getenv("WEB_PORT", "8000"))
    debug = os.getenv("WEB_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    app.run(host=host, port=port, debug=debug)
