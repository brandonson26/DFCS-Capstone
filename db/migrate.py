from pathlib import Path

from db import get_conn

schema = Path("db/schema.sql").read_text(encoding="utf-8")


def main() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(schema)
        conn.commit()


if __name__ == "__main__":
    main()
