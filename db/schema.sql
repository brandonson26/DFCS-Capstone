BEGIN;

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS files (
  id BIGSERIAL PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,
  sha256 CHAR(64) UNIQUE NOT NULL,
  hdu_index INT,
  instrument TEXT,
  satellite TEXT,
  quality_status TEXT NOT NULL DEFAULT 'useable' CHECK (quality_status IN ('useable', 'contaminated')),
  ingested_at TIMESTAMPTZ DEFAULT NOW(),
  hdr_small JSONB
);

CREATE INDEX IF NOT EXISTS files_quality_status_idx ON files (quality_status);
CREATE INDEX IF NOT EXISTS files_instrument_idx ON files (instrument);
CREATE INDEX IF NOT EXISTS files_satellite_idx ON files (satellite);

CREATE TABLE IF NOT EXISTS flags (
  id SERIAL PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS file_flags (
  file_id BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  flag_id INT NOT NULL REFERENCES flags(id) ON DELETE CASCADE,
  value BOOLEAN NOT NULL,
  info JSONB,
  PRIMARY KEY (file_id, flag_id)
);

CREATE TABLE IF NOT EXISTS runs (
  id BIGSERIAL PRIMARY KEY,
  file_id BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  outdir TEXT NOT NULL,
  run_name TEXT NOT NULL,
  dest_dirs JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

COMMIT;
