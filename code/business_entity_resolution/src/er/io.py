"""TSV/parquet IO, entity-id encoding, and per-stage resume markers.

Every pipeline stage writes into ``work/<stage>/`` and calls
``mark_stage_done`` on success; the next run calls ``stage_is_done`` first
and skips the stage if its config hash still matches, so a crashed
``run_all.sh`` resumes without redoing finished work.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

# Arrow's default (mimalloc) pool keeps freed buffers mapped: streaming a few
# GB of parquet left ~3GB of RSS behind with 0 bytes actually allocated,
# which is what pushed stage 06 to 8GB+ on a 16GB machine. The system
# allocator hands memory back (with MALLOC_ARENA_MAX=2, see run_all.sh).
pa.set_memory_pool(pa.system_memory_pool())


def release_memory() -> None:
    """Hand freed heap memory back to the OS between heavy phases: glibc keeps
    fragmented free chunks mapped otherwise (measured ~1GB after one big
    parquet read), which counts against the stage's memory cap."""
    import ctypes
    import gc

    gc.collect()
    pa.default_memory_pool().release_unused()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # not glibc
        pass

from .config import CODE_SOURCE, SOURCE_CODE


def read_source_tsv(path: Path) -> pd.DataFrame:
    """Read one ``*_sourceN.tsv`` / ground-truth file per the README's rules."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def encode_ids(series: pd.Series) -> pd.DataFrame:
    """Split a column of ``S1-1234`` style ids into (source int8, id int64)."""
    split = series.str.split("-", n=1, expand=True)
    source = split[0].map(SOURCE_CODE).astype("int8")
    num = split[1].astype("int64")
    return pd.DataFrame({"source": source, "id_num": num}, index=series.index)


def decode_id(source: int, id_num: int) -> str:
    return f"{CODE_SOURCE[int(source)]}-{int(id_num)}"


def encode_id_list(cell: str) -> list[tuple[int, int]]:
    """Encode a comma-separated ``matched_entity_ids`` cell; '' -> []."""
    if not cell:
        return []
    out = []
    for tok in cell.split(","):
        prefix, num = tok.split("-", 1)
        out.append((SOURCE_CODE[prefix], int(num)))
    return out


def join_ids(pairs: Iterable[tuple[int, int]]) -> str:
    return ",".join(decode_id(s, n) for s, n in pairs)


def config_hash(config: dict[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def stage_is_done(stage_dir: Path, config: dict[str, Any]) -> bool:
    marker = stage_dir / "_DONE"
    if not marker.exists():
        return False
    return marker.read_text().strip() == config_hash(config)


def mark_stage_done(stage_dir: Path, config: dict[str, Any]) -> None:
    (stage_dir / "_DONE").write_text(config_hash(config))


def load_normalized(
    norm_dir: Path,
    entity: str,
    scope: str,
    columns: list[str] | None = None,
    country: str | None = None,
    parse_numbers: bool = True,
) -> pd.DataFrame:
    """Load one entity's stage-02 normalized table (`entity` in
    {'s1','s2','s3'}, `scope` in {'train','test'}).

    `columns` / `country` push the column subset and country filter down
    into the parquet read, so a stage only ever materializes the slice it
    works on (row order within the country is the file order either way).
    `parse_numbers` turns `addr_numbers` back from its comma-joined string
    into a list; pass False to keep the compact string column (one Python
    list per row costs ~100 bytes each, i.e. GBs over a ~10M-row pool).
    """
    filters = [("country", "==", country)] if country is not None else None
    df = pd.read_parquet(norm_dir / f"{entity}_{scope}.parquet", columns=columns, filters=filters)
    df = df.reset_index(drop=True)
    if parse_numbers and "addr_numbers" in df.columns:
        df["addr_numbers"] = df["addr_numbers"].apply(lambda s: s.split(",") if s else [])
    return df


def load_pool(
    norm_dir: Path,
    scope: str,
    columns: list[str] | None = None,
    country: str | None = None,
    parse_numbers: bool = True,
) -> pd.DataFrame:
    """S2 rows then S3 rows, concatenated with a fresh RangeIndex. Stages
    03/04/05 all build per-country matrices/embeddings over this exact row
    order (S2 first, then S3, each in their original file order), so a pool
    row index means the same record everywhere. Filtering by `country` here
    gives the same rows, in the same order, as filtering the full pool.
    """
    s2 = load_normalized(norm_dir, "s2", scope, columns, country, parse_numbers)
    s2["source"] = np.int8(2)
    s3 = load_normalized(norm_dir, "s3", scope, columns, country, parse_numbers)
    s3["source"] = np.int8(3)
    return pd.concat([s2, s3], ignore_index=True)


def load_s1(
    norm_dir: Path,
    scope: str,
    columns: list[str] | None = None,
    country: str | None = None,
    parse_numbers: bool = True,
) -> pd.DataFrame:
    return load_normalized(norm_dir, "s1", scope, columns, country, parse_numbers)


def load_countries(norm_dir: Path, entity: str, scope: str) -> list[str]:
    """Sorted distinct countries of one normalized table, reading only the
    `country` column."""
    col = pd.read_parquet(norm_dir / f"{entity}_{scope}.parquet", columns=["country"])["country"]
    return sorted(col.dropna().unique().tolist())


def load_s1_splits(load_dir: Path) -> pd.DataFrame:
    """entity_id_num -> split for train S1 (from stage 01), compact."""
    return pd.read_parquet(load_dir / "s1_train.parquet", columns=["entity_id_num", "split"])


def pair_key(source, id_num) -> np.ndarray:
    """One int64 per (source, id_num) record key, for vectorized joins
    instead of a Python dict keyed by (source, id) tuples. Ids are < 2**32."""
    return (np.asarray(source, dtype=np.int64) << 32) | np.asarray(id_num, dtype=np.int64)
