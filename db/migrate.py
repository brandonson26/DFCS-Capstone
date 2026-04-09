from pathlib import Path

from db import get_conn

schema = Path("db/schema.sql").read_text(encoding="utf-8")

# Incremental migrations for columns added after initial schema deployment.
MIGRATIONS = [
    # Add object_type column if it doesn't exist yet (existing databases).
    """
    ALTER TABLE files
      ADD COLUMN IF NOT EXISTS object_type TEXT NOT NULL DEFAULT 'unknown'
        CHECK (object_type IN ('satellite', 'star', 'unknown'));
    """,
    """
    CREATE INDEX IF NOT EXISTS files_object_type_idx ON files (object_type);
    """,
]


def main() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Add new columns first so the schema's index creation can succeed.
            for migration in MIGRATIONS:
                cur.execute(migration)
            # Apply full schema (idempotent — uses IF NOT EXISTS everywhere).
            cur.execute(schema)
        conn.commit()
    print("Migration complete.")


if __name__ == "__main__":
    main()
