from pathlib import Path

# local import so running "python3 db/migrate.py" works reliably
from db import get_conn

schema = Path("db/schema.sql").read_text(encoding="utf-8")

with get_conn() as conn:
    with conn.cursor() as cur:
        cur.execute(schema)
    conn.commit()

print("Applied db/schema.sql")
