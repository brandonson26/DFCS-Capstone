"""
tle_catalog.py

Parses a TLE (Two-Line Element) catalog file and classifies an object name
as 'satellite', 'star', or 'unknown' based on whether it matches any entry.

TLE format (3-line):
    SATELLITE NAME
    1 NNNNNX NNNNNNX NNNNNNN.NNNNNNNN ...
    2 NNNNN NNN.NNNN NNN.NNNN ...

Only the name lines (not starting with "1 " or "2 ") are extracted.
"""

from __future__ import annotations
from pathlib import Path
from typing import FrozenSet

# Path to the TLE catalog — same directory as this file.
_DEFAULT_TLE_PATH = Path(__file__).resolve().parent / "TLEs_202512231.txt"


def _load_satellite_names(tle_path: Path) -> FrozenSet[str]:
    """Return the set of satellite names (uppercased) from a TLE file."""
    names: set[str] = set()
    if not tle_path.exists():
        return frozenset()
    with tle_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip TLE data lines (start with "1 " or "2 " followed by digits)
            if line[:2] in {"1 ", "2 "}:
                continue
            names.add(line.upper())
    return frozenset(names)


def _normalize(s: str) -> str:
    """Strip spaces, hyphens, underscores and uppercase for fuzzy matching."""
    return s.upper().replace("-", "").replace(" ", "").replace("_", "")


# Known abbreviations used in FITS OBJECT headers that don't exactly match
# the TLE catalog names. Add entries here whenever a new abbreviation appears.
_ALIASES: FrozenSet[str] = frozenset(_normalize(a) for a in [
    "DTV10", "DTV11", "DTV12", "DTV14", "DTV15",  # DIRECTV series
])

# Load catalog once at import time.
_SATELLITE_NAMES: FrozenSet[str] = _load_satellite_names(_DEFAULT_TLE_PATH)
_SATELLITE_NAMES_NORM: FrozenSet[str] = frozenset(_normalize(n) for n in _SATELLITE_NAMES) | _ALIASES


def get_satellite_names() -> FrozenSet[str]:
    """Return the raw satellite names loaded from the TLE catalog."""
    return _SATELLITE_NAMES


def classify_object(object_name: str | None) -> str:
    """
    Classify an object header value against the TLE catalog.

    Returns
    -------
    'satellite'  — object name matches an entry in the TLE catalog
    'star'       — object name is present but not in the TLE catalog
    'unknown'    — object name is absent or empty
    """
    if not object_name or not str(object_name).strip():
        return "unknown"

    norm = _normalize(str(object_name))

    for sat_norm in _SATELLITE_NAMES_NORM:
        # Match if either string contains the other (handles abbreviations
        # like "DTV10" ↔ "DIRECTV10" or "ATT T16" ↔ "ATTT16").
        if sat_norm in norm or norm in sat_norm:
            return "satellite"

    return "star"
