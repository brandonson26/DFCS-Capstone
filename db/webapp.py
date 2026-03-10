import os

from flask import Flask
from flask import jsonify
from flask import render_template_string
from db import get_conn

app = Flask(__name__)

TEMPLATE = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Capstone DB</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: ui-sans-serif, system-ui, Arial; margin: 20px; color: #0f172a; }
    table { border-collapse: collapse; width: 100%; max-width: 1100px; }
    th, td { border: 1px solid #d1d5db; padding: 6px; text-align: left; }
    th { background: #f3f4f6; }
    .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 12px; max-width: 1100px; }
  </style>
</head>
<body>
  <h1>Capstone DB Dashboard</h1>
  <div class="card">
    <h3>Quick checks</h3>
    <p>files: {{ counts.files }}</p>
    <p>flags: {{ counts.flags }}</p>
    <p>run records: {{ counts.runs }}</p>
    <p>file/flag links:
      <a href="/files">files</a> |
      <a href="/file_flags">file_flags</a>
    </p>
  </div>
  <div class="card">
    <h3>Recent files</h3>
    <table>
      <tr><th>Path</th><th>Status</th><th>Instrument</th><th>Satellite</th><th>Ingested At</th><th>SHA</th></tr>
      {% for row in recent_files %}
      <tr>
        <td>{{ row.path }}</td>
        <td>{{ row.quality_status }}</td>
        <td>{{ row.instrument or '' }}</td>
        <td>{{ row.satellite or '' }}</td>
        <td>{{ row.ingested_at }}</td>
        <td>{{ row.sha256 }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  <div class="card">
    <p>Health: {{ db_ok }}</p>
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


def recent_files(limit: int = 25):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT path, quality_status, instrument, satellite, ingested_at, sha256
                FROM files
                ORDER BY ingested_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [
        {
            "path": r[0],
            "quality_status": r[1],
            "instrument": r[2],
            "satellite": r[3],
            "ingested_at": str(r[4]),
            "sha256": r[5],
        }
        for r in rows
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
        recent_files=recent_files(25),
        db_ok=db_ok,
    )


@app.get("/files")
def files_page():
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, path, quality_status, instrument, satellite, sha256, ingested_at FROM files ORDER BY ingested_at DESC LIMIT 500")
                rows = cur.fetchall()
        return jsonify([
            {
                "id": r[0], "path": r[1], "quality_status": r[2],
                "instrument": r[3], "satellite": r[4], "sha256": r[5], "ingested_at": str(r[6]),
            }
            for r in rows
        ])
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


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
