import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def upsert_file(
    cur,
    file_path: str,
    sha256: str,
    hdu_index: Optional[int],
    hdr_small: Optional[Dict[str, Any]],
    good_bad: Optional[str],
) -> int:
    cur.execute(
        """
        INSERT INTO files(path, sha256, hdu_index, hdr_small, good_bad)
        VALUES (%s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (sha256) DO UPDATE SET
          path = EXCLUDED.path,
          hdu_index = EXCLUDED.hdu_index,
          hdr_small = EXCLUDED.hdr_small,
          good_bad = EXCLUDED.good_bad,
          ingested_at = NOW()
        RETURNING id
        """,
        (file_path, sha256, hdu_index, json.dumps(hdr_small or {}), good_bad),
    )
    return cur.fetchone()[0]


def get_flag_id(cur, name: str) -> int:
    cur.execute(
        """
        INSERT INTO flags(name)
        VALUES (%s)
        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
        RETURNING id
        """,
        (name,),
    )
    return cur.fetchone()[0]


def upsert_flags(cur, file_id: int, flags: Dict[str, Any], infos: Optional[Dict[str, Any]] = None) -> None:
    infos = infos or {}
    for flag_name, flag_value in flags.items():
        flag_id = get_flag_id(cur, flag_name)
        info_obj = infos.get(flag_name)
        cur.execute(
            """
            INSERT INTO file_flags(file_id, flag_id, value, info)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (file_id, flag_id) DO UPDATE SET
              value = EXCLUDED.value,
              info = EXCLUDED.info
            """,
            (file_id, flag_id, bool(flag_value), json.dumps(info_obj) if info_obj is not None else None),
        )


def insert_run(cur, file_id: int, outdir: str, run_name: str, dest_dirs: List[str]) -> None:
    cur.execute(
        """
        INSERT INTO runs(file_id, outdir, run_name, dest_dirs)
        VALUES (%s, %s, %s, %s::jsonb)
        """,
        (file_id, outdir, run_name, json.dumps(dest_dirs)),
    )


def write_result_to_db(
    *,
    fits_path: Path,
    hdu_index: int,
    hdr_small: Dict[str, Any],
    outdir: Path,
    run_name: str,
    dest_dirs: List[Path],
    good_bad: str,
    flags: Dict[str, Any],
    flag_infos: Dict[str, Any],
) -> None:
    digest = sha256_file(fits_path)

    with get_conn() as conn:
        with conn.cursor() as cur:
            file_id = upsert_file(cur, str(fits_path), digest, hdu_index, hdr_small, good_bad)
            upsert_flags(cur, file_id, flags, flag_infos)
            insert_run(cur, file_id, str(outdir), run_name, [str(d) for d in dest_dirs])
        conn.commit()
