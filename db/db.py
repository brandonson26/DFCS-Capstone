import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import logging

import psycopg2
from psycopg2 import OperationalError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def connection_params() -> Dict[str, str | int]:
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "capstone_db"),
        "user": os.getenv("DB_USER", "capstone_user"),
        "password": os.getenv("DB_PASSWORD", "capstone_pass"),
    }


def get_conn():
    params = connection_params()
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            return psycopg2.connect(
                host=params["host"],
                port=params["port"],
                dbname=params["dbname"],
                user=params["user"],
                password=params["password"],
            )
        except OperationalError as exc:
            last_error = exc
            msg = str(exc).lower()
            logger.warning(
                "PostgreSQL connect attempt %s failed (host=%s db=%s user=%s): %s",
                attempt,
                params["host"],
                params["dbname"],
                params["user"],
                msg,
            )
            if "password authentication failed" in msg:
                raise RuntimeError(
                    "PostgreSQL rejected the credentials for DB_USER/DB_PASSWORD. "
                    "If this started after compose recreate, remove the old volume with "
                    "`docker compose down -v && docker compose up -d` (this resets DB users)."
                ) from exc
            if "connect to server" in msg and attempt < 3:
                time.sleep(1.5)
                continue
            raise

    if last_error:
        raise last_error


def upsert_file(
    cur,
    file_path: str,
    sha256: str,
    hdu_index: Optional[int],
    instrument: Optional[str],
    satellite: Optional[str],
    quality_status: str,
    hdr_small: Optional[Dict[str, Any]],
) -> int:
    cur.execute(
        """
        INSERT INTO files(path, sha256, hdu_index, instrument, satellite, quality_status, hdr_small)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (sha256) DO UPDATE SET
          path = EXCLUDED.path,
          hdu_index = EXCLUDED.hdu_index,
          instrument = EXCLUDED.instrument,
          satellite = EXCLUDED.satellite,
          quality_status = EXCLUDED.quality_status,
          hdr_small = EXCLUDED.hdr_small,
          ingested_at = NOW()
        RETURNING id
        """,
        (
            file_path,
            sha256,
            hdu_index,
            instrument,
            satellite,
            quality_status,
            json.dumps(hdr_small or {}),
        ),
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
    quality_status: str,
    flags: Dict[str, Any],
    flag_infos: Dict[str, Any],
    instrument: Optional[str] = None,
    satellite: Optional[str] = None,
) -> None:
    digest = sha256_file(fits_path)

    with get_conn() as conn:
        with conn.cursor() as cur:
            file_id = upsert_file(
                cur,
                str(fits_path),
                digest,
                hdu_index,
                instrument,
                satellite,
                quality_status,
                hdr_small,
            )
            upsert_flags(cur, file_id, flags, flag_infos)
            insert_run(cur, file_id, str(outdir), run_name, [str(d) for d in dest_dirs])
        conn.commit()


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
