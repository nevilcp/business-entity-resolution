#!/usr/bin/env python3
"""Stage 04: token/q-gram blocking + CNP, run separately over the train pool
(all train S1, for the "competing S1" aggregates and for T/Vcal/Vtest
candidate generation) and the test pool (all test S1, for candidate_pairs.tsv).

CNP's k is chosen once, from Vcal recall on the train pool, then reused for
the test pool. The optional dense channel (work/03_dense/, stage 03) is
folded in only if it improves Vcal pairs-completeness at that k by >= 0.002.

Memory layout (sized for a 16GB-RAM laptop, not just the 32GB remote box):
  - one scope and one country are loaded at a time, only the text columns
    blocking needs, with `addr_numbers` kept as its compact string column;
  - the pool index is built by streaming over the pool (see
    er/blocking.py's module docstring), never as one list of key lists;
  - every S1 row is scored exactly once per country, in `QUERY_CHUNK_SIZE`
    chunks, keeping only int32 top-`k_max` ids plus per-pool-row reverse
    aggregates for the base and (if present) dense variants. Choosing k and
    the dense decision afterwards only slices those arrays, so no country's
    pool index has to stay alive while the others are processed;
  - recall/PQ are computed with vectorized numpy over the ground-truth
    pairs, not with a Python set per S1 row;
  - test candidates are written to disk country by country.

Outputs under work/04_blocking/:
  candidate_pairs.tsv               test candidates, every test S1 (the file copied to output/)
  test_candidates.parquet           the same, long format: s1_id_num, match_source, match_id_num, rank
  train_candidates.parquet          long format, T/Vcal/Vtest rows only: s1_id_num, split, match_source, match_id_num, rank
  train_reverse_agg.parquet         source, id_num, best, second, count  (over ALL train S1)
  test_reverse_agg.parquet          source, id_num, best, second, count  (over ALL test S1)
  report.json                       recall@k grid, chosen k, dense decision, RR/PQ, per country
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.blocking import (
    CNP_K_GRID,
    DENSE_MIN_GAIN,
    NGRAM_DF_CAP_ABS,
    POOL_CHUNK_SIZE,
    QUERY_CHUNK_SIZE,
    WORD_DF_CAP_ABS,
    accumulate_column_top2,
    build_channel_index,
    ngram_channel_keys,
    reduction_ratio,
    score_query_chunk,
    topk_per_row,
    word_channel_keys,
)
from er.config import add_common_args, paths_from_args
from er.io import load_countries, load_pool, load_s1, load_s1_splits, mark_stage_done, pair_key, stage_is_done

# Bumped whenever the output format changes, so a stale _DONE isn't reused.
OUTPUT_VERSION = 3  # bumped: candidates now carry the real per-candidate RRF score
TEXT_COLS = ["entity_id_num", "name_clean", "address_clean", "addr_numbers", "name_nospace"]
K_MAX = max(CNP_K_GRID)


def word_keys_of(df: pd.DataFrame) -> list[list[str]]:
    return [
        word_channel_keys(n, a, an.split(",") if an else [])
        for n, a, an in zip(df["name_clean"].tolist(), df["address_clean"].tolist(), df["addr_numbers"].tolist())
    ]


def ngram_keys_of(df: pd.DataFrame) -> list[list[str]]:
    return [ngram_channel_keys(x) for x in df["name_nospace"].tolist()]


def chunk_bounds(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(start + chunk_size, n)


def pool_chunks(pool_c: pd.DataFrame, keys_fn):
    """Zero-arg factory re-yielding the pool's keys chunk by chunk (called
    once per streaming pass of `build_channel_index`)."""
    return lambda: (keys_fn(pool_c.iloc[s:e]) for s, e in chunk_bounds(len(pool_c), POOL_CHUNK_SIZE))


def load_dense_ids(dense_dir: Path | None, scope: str, country: str, n_s1: int) -> np.ndarray | None:
    if dense_dir is None:
        return None
    path = dense_dir / f"{scope}_{country}_ids.npy"
    if not path.exists():
        return None
    ids = np.load(path, mmap_mode="r")
    if ids.shape[0] != n_s1:
        print(f"  WARN: {path.name} has {ids.shape[0]} rows, expected {n_s1}; ignoring the dense channel here")
        return None
    return ids


def block_country(
    s1_c: pd.DataFrame, pool_c: pd.DataFrame, dense_ids: np.ndarray | None, variants: tuple[str, ...], k: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Score every S1 row of one country once. For each requested variant
    ('base', 'dense') returns int32 top-`k` ids (n_s1 x k, -1 padded) and
    the pool-side reverse aggregates (best, second, count). The 'dense'
    variant falls back to 'base' when this country has no dense ids."""
    t0 = time.time()
    word_index = build_channel_index(pool_chunks(pool_c, word_keys_of), WORD_DF_CAP_ABS)
    ngram_index = build_channel_index(pool_chunks(pool_c, ngram_keys_of), NGRAM_DF_CAP_ABS)
    print(f"    pool index built in {time.time() - t0:.0f}s "
          f"(word vocab {len(word_index.vocab)}, 3-gram vocab {len(ngram_index.vocab)})", flush=True)

    n_s1, n_pool = len(s1_c), len(pool_c)
    want_dense = "dense" in variants and dense_ids is not None
    out = {
        v: {
            "ids": np.full((n_s1, k), -1, dtype=np.int32),
            # RRF-fused score for each of "ids"'s candidates, same shape/order --
            # carried through to candidate_pairs so 05_features can compare a
            # candidate's own blocking score against the reverse-agg best/second
            # below on the same scale (both RRF), instead of recomputing an
            # unrelated 0-1 name/address-cosine proxy at feature time.
            "scores": np.full((n_s1, k), -np.inf, dtype=np.float32),
            "best": np.zeros(n_pool, dtype=np.float32),
            "second": np.zeros(n_pool, dtype=np.float32),
            "count": np.zeros(n_pool, dtype=np.int32),
        }
        for v in variants if v == "base" or want_dense
    }

    t0 = time.time()
    for start, end in chunk_bounds(n_s1, QUERY_CHUNK_SIZE):
        chunk = s1_c.iloc[start:end]
        dense_chunk = np.asarray(dense_ids[start:end]) if want_dense else None
        fused_base, fused_dense = score_query_chunk(
            word_keys_of(chunk), ngram_keys_of(chunk), word_index, ngram_index, dense_chunk,
        )
        for v, fused in (("base", fused_base), ("dense", fused_dense)):
            if v not in out:
                continue
            o = out[v]
            accumulate_column_top2(fused, o["best"], o["second"], o["count"])
            ids_chunk, scores_chunk = topk_per_row(fused, k)
            o["ids"][start:end] = ids_chunk
            o["scores"][start:end] = scores_chunk
        del fused_base, fused_dense
    print(f"    {n_s1} S1 rows scored in {time.time() - t0:.0f}s", flush=True)

    if "dense" in variants and "dense" not in out:
        out["dense"] = out["base"]
    return out


def truth_rows(gt_pairs: pd.DataFrame, s1_ids: np.ndarray, pool_keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth pairs of this country as (local S1 row, local pool row)
    arrays, deduplicated, dropping matches that aren't in this pool."""
    s1_row = pd.Index(s1_ids).get_indexer(gt_pairs["s1_id_num"].to_numpy())
    sub = gt_pairs[s1_row >= 0]
    s1_row = s1_row[s1_row >= 0]
    pool_row = pd.Index(pool_keys).get_indexer(pair_key(sub["match_source"], sub["match_id_num"]))
    keep = pool_row >= 0
    pairs = np.unique(np.stack([s1_row[keep], pool_row[keep]], axis=1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def hit_positions(ids: np.ndarray, s1_row: np.ndarray, pool_row: np.ndarray, chunk: int = 1_000_000) -> np.ndarray:
    """Rank (0-based) at which each true pair's pool row appears in its S1
    row's candidate list, or ids.shape[1] if it's absent."""
    pos = np.full(len(s1_row), ids.shape[1], dtype=np.int32)
    for s, e in chunk_bounds(len(s1_row), chunk):
        eq = ids[s1_row[s:e]] == pool_row[s:e, None]
        has = eq.any(axis=1)
        pos[s:e] = np.where(has, eq.argmax(axis=1), ids.shape[1])
    return pos


def vcal_recall(per_country: dict[str, dict], variant: str) -> dict[int, float]:
    """Pairs-completeness over every country's Vcal true pairs, per k."""
    pos = np.concatenate([c["pos"][variant][c["is_vcal"]] for c in per_country.values()])
    return {k: float((pos < k).mean()) if len(pos) else 1.0 for k in CNP_K_GRID}


def reverse_agg_frame(pool_src: np.ndarray, pool_id: np.ndarray, agg: dict[str, np.ndarray]) -> pd.DataFrame:
    nz = agg["count"] > 0
    return pd.DataFrame({
        "source": pool_src[nz].astype(np.int8), "id_num": pool_id[nz].astype(np.int64),
        "best": agg["best"][nz], "second": agg["second"][nz], "count": agg["count"][nz],
    })


def long_candidates(s1_ids: np.ndarray, ids_k: np.ndarray, scores_k: np.ndarray, pool_src: np.ndarray, pool_id: np.ndarray) -> pd.DataFrame:
    """One row per (S1, candidate). Ids are stored as int32 (all < 1e9)."""
    rows, ranks = np.nonzero(ids_k >= 0)
    cols = ids_k[rows, ranks]
    return pd.DataFrame({
        "s1_id_num": s1_ids[rows].astype(np.int32),
        "match_source": pool_src[cols].astype(np.int8),
        "match_id_num": pool_id[cols].astype(np.int32),
        "rank": (ranks + 1).astype(np.int16),
        "score": scores_k[rows, ranks].astype(np.float32),
    })


def write_test_candidates(tsv, writer_holder: dict, long_path: Path,
                          s1_ids: np.ndarray, ids_k: np.ndarray, scores_k: np.ndarray, pool_src: np.ndarray, pool_id: np.ndarray,
                          chunk: int = 50_000) -> None:
    """Append one country's candidates to candidate_pairs.tsv and the long
    parquet, `chunk` S1 rows at a time (a whole country at once is ~40M long
    rows, whose DataFrame + Arrow copies were stage 04's memory peak)."""
    src_l, id_l = pool_src.tolist(), pool_id.tolist()
    for start, end in chunk_bounds(len(s1_ids), chunk):
        block = ids_k[start:end]
        lines = []
        for s1, row in zip(s1_ids[start:end].tolist(), block.tolist()):
            cands = ",".join(f"S{src_l[j]}-{id_l[j]}" for j in row if j >= 0)
            lines.append(f"S1-{s1}\t{cands}\n")
        tsv.write("".join(lines))
        table = pa.Table.from_pandas(
            long_candidates(s1_ids[start:end], block, scores_k[start:end], pool_src, pool_id), preserve_index=False,
        )
        if writer_holder.get("w") is None:
            writer_holder["w"] = pq.ParquetWriter(long_path, table.schema)
        writer_holder["w"].write_table(table)


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--dense-dir", default=None, help="work/03_dense dir; auto-detected under --work-dir if omitted")
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    norm_dir = paths.stage_dir("02_normalize")
    load_dir = paths.stage_dir("01_load")
    stage_dir = paths.stage_dir("04_blocking")
    if args.dense_dir:
        dense_dir = Path(args.dense_dir)
    else:
        auto = paths.work_dir / "03_dense"
        dense_dir = auto if any(auto.glob("*_ids.npy")) else None

    config = {
        "data_dir": str(paths.data_dir), "seed": args.seed,
        "dense_dir": str(dense_dir) if dense_dir else None, "output_version": OUTPUT_VERSION,
    }
    if stage_is_done(stage_dir, config):
        print("04_blocking: already done, skipping")
        return 0

    report: dict = {"countries": {}}

    # ======================================================================
    # Train scope: one scoring pass per country (base + dense variants at
    # k_max), then choose k / dense from the Vcal rows' stored hit ranks.
    # ======================================================================
    print("train scope...")
    splits = load_s1_splits(load_dir)
    gt_pairs = pd.read_parquet(load_dir / "gt_pairs.parquet")
    per_country: dict[str, dict] = {}
    for country in load_countries(norm_dir, "s1", "train"):
        s1_c = load_s1(norm_dir, "train", TEXT_COLS, country, parse_numbers=False)
        pool_c = load_pool(norm_dir, "train", TEXT_COLS, country, parse_numbers=False)
        if len(s1_c) == 0 or len(pool_c) == 0:
            continue
        print(f"  {country}: {len(s1_c)} S1 x {len(pool_c)} pool", flush=True)
        s1_ids = s1_c["entity_id_num"].to_numpy()
        split = s1_c[["entity_id_num"]].merge(splits, on="entity_id_num", how="left")["split"].fillna("").to_numpy()
        dense_ids = load_dense_ids(dense_dir, "train", country, len(s1_c))
        variants = ("base", "dense") if dense_dir is not None else ("base",)
        res = block_country(s1_c, pool_c, dense_ids, variants, K_MAX)

        pool_src = pool_c["source"].to_numpy(np.int8)
        pool_id = pool_c["entity_id_num"].to_numpy(np.int64)
        del s1_c, pool_c
        t_s1, t_pool = truth_rows(gt_pairs, s1_ids, pair_key(pool_src, pool_id))
        per_country[country] = dict(
            s1_ids=s1_ids, split=split, pool_src=pool_src, pool_id=pool_id, res=res,
            t_s1=t_s1, t_pool=t_pool, is_vcal=split[t_s1] == "Vcal",
            pos={v: hit_positions(r["ids"], t_s1, t_pool) for v, r in res.items()},
        )

    recall_base = vcal_recall(per_country, "base")
    chosen_k = next(k for k in sorted(CNP_K_GRID) if recall_base[k] >= recall_base[K_MAX] - 0.001)
    use_dense = False
    if dense_dir is not None and any(c["res"]["dense"] is not c["res"]["base"] for c in per_country.values()):
        recall_dense = vcal_recall(per_country, "dense")
        report["vcal_recall_by_k_dense"] = recall_dense
        use_dense = recall_dense[chosen_k] - recall_base[chosen_k] >= DENSE_MIN_GAIN
    variant = "dense" if use_dense else "base"

    print(f"chosen CNP k = {chosen_k}, dense channel {'included' if use_dense else 'excluded'}")
    report["cnp_k"] = chosen_k
    report["dense_included"] = use_dense
    report["vcal_recall_by_k_base"] = recall_base

    train_cands, train_reverse = [], []
    for country, c in per_country.items():
        res = c["res"][variant]
        ids_k = res["ids"][:, :chosen_k]
        scores_k = res["scores"][:, :chosen_k]
        hits = int((c["pos"][variant] < chosen_k).sum())
        total = len(c["t_s1"])
        n_cand = int((ids_k >= 0).sum())
        n_s1, n_pool = len(c["s1_ids"]), len(c["pool_id"])
        report["countries"][country] = {
            "n_s1": n_s1, "n_pool": n_pool,
            "pairs_completeness": hits / total if total else 1.0,
            "reduction_ratio": reduction_ratio(n_cand, n_s1, n_pool),
            "pair_quality": hits / n_cand if n_cand else 0.0,
        }
        train_reverse.append(reverse_agg_frame(c["pool_src"], c["pool_id"], res))
        keep = np.isin(c["split"], ["T", "Vcal", "Vtest"])
        cand = long_candidates(c["s1_ids"][keep], ids_k[keep], scores_k[keep], c["pool_src"], c["pool_id"])
        cand.insert(1, "split", pd.Categorical(np.repeat(c["split"][keep], (ids_k[keep] >= 0).sum(axis=1))))
        train_cands.append(cand)
        print(f"  {country}: pairs_completeness={report['countries'][country]['pairs_completeness']:.4f}")

    pd.concat(train_cands, ignore_index=True).to_parquet(stage_dir / "train_candidates.parquet", index=False)
    pd.concat(train_reverse, ignore_index=True).to_parquet(stage_dir / "train_reverse_agg.parquet", index=False)
    del per_country, train_cands, train_reverse, gt_pairs, splits

    # ======================================================================
    # Test scope: same scoring at the chosen k / dense, one country at a
    # time, streaming each country's candidates straight to disk.
    # ======================================================================
    print("test scope...")
    test_reverse = []
    tsv_tmp = stage_dir / "candidate_pairs.tsv.tmp"
    long_tmp = stage_dir / "test_candidates.parquet.tmp"
    holder: dict = {}
    with open(tsv_tmp, "w") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in load_countries(norm_dir, "s1", "test"):
            s1_c = load_s1(norm_dir, "test", TEXT_COLS, country, parse_numbers=False)
            pool_c = load_pool(norm_dir, "test", TEXT_COLS, country, parse_numbers=False)
            s1_ids = s1_c["entity_id_num"].to_numpy()
            pool_src = pool_c["source"].to_numpy(np.int8)
            pool_id = pool_c["entity_id_num"].to_numpy(np.int64)
            print(f"  {country}: {len(s1_c)} S1 x {len(pool_c)} pool", flush=True)
            if len(pool_c) == 0:
                ids_k = np.full((len(s1_c), chosen_k), -1, dtype=np.int32)
                scores_k = np.full((len(s1_c), chosen_k), -np.inf, dtype=np.float32)
            else:
                dense_ids = load_dense_ids(dense_dir, "test", country, len(s1_c)) if use_dense else None
                res = block_country(s1_c, pool_c, dense_ids, (variant,), chosen_k)[variant]
                ids_k = res["ids"]
                scores_k = res["scores"]
                test_reverse.append(reverse_agg_frame(pool_src, pool_id, res))
            del s1_c, pool_c

            write_test_candidates(f, holder, long_tmp, s1_ids, ids_k, scores_k, pool_src, pool_id)
            del ids_k, scores_k
    if holder.get("w") is not None:
        holder["w"].close()
        long_tmp.replace(stage_dir / "test_candidates.parquet")
    tsv_tmp.replace(stage_dir / "candidate_pairs.tsv")

    pd.concat(test_reverse, ignore_index=True).to_parquet(stage_dir / "test_reverse_agg.parquet", index=False)
    with open(stage_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    mark_stage_done(stage_dir, config)
    print("04_blocking: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
