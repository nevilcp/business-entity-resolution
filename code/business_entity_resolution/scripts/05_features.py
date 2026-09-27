#!/usr/bin/env python3
"""Stage 05: pairwise name/address/graph features for every candidate pair,
processed per country. Train rows get a 0/1 label from the ground truth;
test rows (every row of candidate_pairs.tsv) do not.

Reads work/01_load, work/02_normalize and work/04_blocking (the long-format
train_candidates/test_candidates parquet). Writes
work/05_features/{train,test}_features.parquet.

Memory (16GB laptop): at CNP k=50 the test side alone is ~86M pairs, so
nothing is ever held for all pairs at once. One scope and one country are
loaded at a time (only the record columns the features read); candidate
pairs are processed in S1-aligned chunks of ~`--chunk-pairs` rows, and each
chunk's float32 features are appended to the output parquet as soon as
they're computed. The text-similarity features (the slow, pure-Python part)
run in `--workers` forked processes that receive each chunk's record texts
and share the per-country IDF table copy-on-write; the rank/gap/"graph"
features are vectorized numpy in the parent.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import add_common_args, paths_from_args
from er.features import PAIR_FEATURE_NAMES, build_idf_chunked, pair_feature_matrix
from er.io import load_countries, load_pool, load_s1, load_s1_splits, mark_stage_done, pair_key, stage_is_done

OUTPUT_VERSION = 2
REC_COLS = ["entity_id_num", "name_clean", "name_nospace", "legal_form", "name_transliterated",
            "address_clean", "postcode", "addr_numbers"]
REC_FIELDS = REC_COLS[1:]
GRAPH_COLS = ["fused_score", "fused_rank", "gap_to_s1_best", "reverse_rank", "gap_to_record_best", "is_s3"]
FEATURE_COLS = ["s1_name_idf_sum", "s1_name_share_count", *PAIR_FEATURE_NAMES, *GRAPH_COLS]
NAME_IDF = PAIR_FEATURE_NAMES.index("name_idf_cosine")
ADDR_IDF = PAIR_FEATURE_NAMES.index("addr_idf_cosine")

_IDF: dict[str, float] = {}  # set in the parent before forking the worker pool


def records(df: pd.DataFrame) -> list[dict]:
    """Record dicts for `name_features`/`address_features`, with missing
    strings as None (pandas' NA would otherwise compare as a real value)."""
    cols = {c: df[c].tolist() for c in REC_FIELDS}
    out = []
    for vals in zip(*(cols[c] for c in REC_FIELDS)):
        rec = {c: (None if isinstance(v, float) else v) for c, v in zip(REC_FIELDS, vals)}
        rec["name_clean"] = rec["name_clean"] or ""
        rec["name_nospace"] = rec["name_nospace"] or ""
        rec["address_clean"] = rec["address_clean"] or ""
        rec["addr_numbers"] = rec["addr_numbers"].split(",") if rec["addr_numbers"] else []
        out.append(rec)
    return out


def _worker(task: tuple[list[dict], list[dict]]) -> np.ndarray:
    recs_a, recs_b = task
    return pair_feature_matrix(recs_a, recs_b, _IDF)


def group_starts(s1_row: np.ndarray) -> np.ndarray:
    """Start offsets of each run of equal S1 rows (candidates are stored
    grouped by S1)."""
    return np.concatenate([[0], np.flatnonzero(np.diff(s1_row) != 0) + 1])


def chunk_ranges(s1_row: np.ndarray, chunk_pairs: int):
    """[lo, hi) candidate ranges of about `chunk_pairs` rows that never split
    one S1's candidates across two chunks."""
    starts = np.append(group_starts(s1_row), len(s1_row))
    lo = 0
    while lo < len(s1_row):
        hi = starts[np.searchsorted(starts, lo + chunk_pairs, side="left")] if lo + chunk_pairs < len(s1_row) else len(s1_row)
        yield lo, int(hi)
        lo = int(hi)


def graph_features(s1_row: np.ndarray, fused: np.ndarray, rec_best: np.ndarray, rec_second: np.ndarray, source: np.ndarray):
    """fused_rank / gap_to_s1_best within each S1 group (rows are grouped by
    S1), plus the reverse-rank features against the per-record blocking
    aggregates."""
    n = len(fused)
    starts = group_starts(s1_row)
    group = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, n)))
    order = np.lexsort((-fused, group))
    rank = np.empty(n, dtype=np.float32)
    rank[order] = np.arange(n) - starts[group[order]] + 1
    best = np.maximum.reduceat(fused, starts)[group]
    reverse_rank = np.where(fused >= rec_best - 1e-9, 1, np.where(fused >= rec_second - 1e-9, 2, 3))
    return {
        "fused_score": fused, "fused_rank": rank, "gap_to_s1_best": best - fused,
        "reverse_rank": reverse_rank.astype(np.float32),
        "gap_to_record_best": np.maximum(rec_best - fused, 0.0),
        "is_s3": (source == 3).astype(np.float32),
    }


def triple_key(s1, source, mid) -> np.ndarray:
    return (np.asarray(s1, np.int64) << 32) | (np.asarray(source, np.int64) << 30) | np.asarray(mid, np.int64)


def country_features(
    scope: str, country: str, cand: pd.DataFrame, norm_dir: Path, reverse: pd.DataFrame,
    split_of: pd.Series | None, truth_keys: np.ndarray | None,
    writer_holder: dict, out_path: Path, workers: int, chunk_pairs: int,
) -> int:
    global _IDF
    t0 = time.time()
    pool_c = load_pool(norm_dir, scope, ["entity_id_num", *REC_FIELDS], country, parse_numbers=False)
    s1_c = load_s1(norm_dir, scope, ["entity_id_num", *REC_FIELDS], country, parse_numbers=False)
    if scope == "train":
        s1_c = s1_c[s1_c["entity_id_num"].isin(cand["s1_id_num"].unique())].reset_index(drop=True)

    _IDF = build_idf_chunked(
        [(n or "").split() + (a or "").split() for n, a in zip(
            pool_c["name_clean"].iloc[s:s + 500_000].fillna("").tolist(),
            pool_c["address_clean"].iloc[s:s + 500_000].fillna("").tolist())]
        for s in range(0, len(pool_c), 500_000)
    )
    names = s1_c["name_clean"].fillna("")
    share_count = names.map(names.value_counts()).to_numpy(np.float32)
    idf_sum = np.array([sum(_IDF.get(t, 0.0) for t in nm.split()) for nm in names.tolist()], dtype=np.float32)

    s1_row = pd.Index(s1_c["entity_id_num"].to_numpy()).get_indexer(cand["s1_id_num"].to_numpy())
    pool_keys = pd.Index(pair_key(pool_c["source"], pool_c["entity_id_num"]))
    pool_row = pool_keys.get_indexer(pair_key(cand["match_source"], cand["match_id_num"]))
    ok = (s1_row >= 0) & (pool_row >= 0)
    cand = cand[ok].reset_index(drop=True)
    s1_row, pool_row = s1_row[ok], pool_row[ok]

    rev_row = pool_keys.get_indexer(pair_key(reverse["source"], reverse["id_num"]))
    rec_best = np.zeros(len(pool_c), dtype=np.float32)
    rec_second = np.zeros(len(pool_c), dtype=np.float32)
    rec_best[rev_row[rev_row >= 0]] = reverse["best"].to_numpy(np.float32)[rev_row >= 0]
    rec_second[rev_row[rev_row >= 0]] = reverse["second"].to_numpy(np.float32)[rev_row >= 0]
    del pool_keys, rev_row
    print(f"  {scope}/{country}: {len(cand)} pairs, {len(s1_c)} S1, {len(pool_c)} pool, setup {time.time() - t0:.0f}s", flush=True)

    ranges = list(chunk_ranges(s1_row, chunk_pairs))

    def tasks():
        for lo, hi in ranges:
            yield records(s1_c.iloc[s1_row[lo:hi]]), records(pool_c.iloc[pool_row[lo:hi]])

    t0 = time.time()
    for (lo, hi), pair_feats in zip(ranges, bounded_map(_worker, tasks(), workers)):
        sl = slice(lo, hi)
        fused = 0.5 * (pair_feats[:, NAME_IDF] + pair_feats[:, ADDR_IDF])
        src = cand["match_source"].to_numpy()[sl]
        graph = graph_features(s1_row[sl], fused.astype(np.float32), rec_best[pool_row[sl]], rec_second[pool_row[sl]], src)
        cols = {
            "s1_id_num": cand["s1_id_num"].to_numpy()[sl],
            "match_source": src,
            "match_id_num": cand["match_id_num"].to_numpy()[sl],
            "s1_name_idf_sum": idf_sum[s1_row[sl]],
            "s1_name_share_count": share_count[s1_row[sl]],
            **{name: pair_feats[:, j] for j, name in enumerate(PAIR_FEATURE_NAMES)},
            **graph,
        }
        df = pd.DataFrame(cols)
        df[FEATURE_COLS] = df[FEATURE_COLS].astype(np.float32)
        if truth_keys is not None:
            keys = triple_key(df["s1_id_num"], df["match_source"], df["match_id_num"])
            pos = np.searchsorted(truth_keys, keys).clip(max=max(len(truth_keys) - 1, 0))
            df["label"] = (truth_keys[pos] == keys).astype(np.int8) if len(truth_keys) else np.int8(0)
            df["split"] = df["s1_id_num"].map(split_of).astype(str)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer_holder.get("w") is None:
            writer_holder["w"] = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer_holder["w"].write_table(table)
    print(f"  {scope}/{country}: features written in {time.time() - t0:.0f}s", flush=True)
    _IDF = {}
    return len(cand)


def bounded_map(fn, tasks, workers: int):
    """Ordered map over `tasks` on a forked pool, with at most 2x`workers`
    tasks in flight. (`Pool.imap` drains its input iterator eagerly, which
    would build every chunk's record dicts in the parent at once.)"""
    if workers <= 1:
        yield from map(fn, tasks)
        return
    from collections import deque

    with mp.get_context("fork").Pool(workers) as pool:
        pending: deque = deque()
        for task in tasks:
            pending.append(pool.apply_async(fn, (task,)))
            if len(pending) >= 2 * workers:
                yield pending.popleft().get()
        while pending:
            yield pending.popleft().get()


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--workers", type=int, default=4, help="feature worker processes (each shares the IDF table copy-on-write)")
    p.add_argument("--chunk-pairs", type=int, default=50_000, help="candidate pairs per worker task (memory: ~2x this many record dicts per task)")
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    norm_dir = paths.stage_dir("02_normalize")
    load_dir = paths.stage_dir("01_load")
    block_dir = paths.stage_dir("04_blocking")
    stage_dir = paths.stage_dir("05_features")

    config = {"data_dir": str(paths.data_dir), "seed": args.seed, "output_version": OUTPUT_VERSION}
    if stage_is_done(stage_dir, config):
        print("05_features: already done, skipping")
        return 0

    print("computing train features...")
    gt = pd.read_parquet(load_dir / "gt_pairs.parquet")
    truth_keys = np.unique(triple_key(gt["s1_id_num"], gt["match_source"], gt["match_id_num"]))
    del gt
    split_of = load_s1_splits(load_dir).set_index("entity_id_num")["split"]
    train_cand = pd.read_parquet(block_dir / "train_candidates.parquet", columns=["s1_id_num", "match_source", "match_id_num"])
    train_reverse = pd.read_parquet(block_dir / "train_reverse_agg.parquet")
    out_path = stage_dir / "train_features.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    holder: dict = {}
    n_train = 0
    for country in load_countries(norm_dir, "s1", "train"):
        ids = load_s1(norm_dir, "train", ["entity_id_num"], country)["entity_id_num"]
        cand = train_cand[train_cand["s1_id_num"].isin(ids)]
        if len(cand):
            n_train += country_features("train", country, cand, norm_dir, train_reverse, split_of, truth_keys,
                                        holder, tmp_path, args.workers, args.chunk_pairs)
    if holder.get("w") is not None:
        holder["w"].close()
        tmp_path.replace(out_path)
    print(f"  wrote {n_train} train feature rows")
    del train_cand, train_reverse, split_of, truth_keys

    print("computing test features...")
    test_path = block_dir / "test_candidates.parquet"
    test_reverse = pd.read_parquet(block_dir / "test_reverse_agg.parquet")
    out_path = stage_dir / "test_features.parquet"
    tmp_path = out_path.with_suffix(".parquet.tmp")
    holder = {}
    n_test = 0
    for country in load_countries(norm_dir, "s1", "test"):
        ids = load_s1(norm_dir, "test", ["entity_id_num"], country)["entity_id_num"]
        cand = pq.read_table(test_path, columns=["s1_id_num", "match_source", "match_id_num"],
                             filters=[("s1_id_num", "in", ids.tolist())]).to_pandas()
        if len(cand):
            n_test += country_features("test", country, cand, norm_dir, test_reverse, None, None,
                                       holder, tmp_path, args.workers, args.chunk_pairs)
        del cand
    if holder.get("w") is not None:
        holder["w"].close()
        tmp_path.replace(out_path)
    print(f"  wrote {n_test} test feature rows")

    mark_stage_done(stage_dir, config)
    print("05_features: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
