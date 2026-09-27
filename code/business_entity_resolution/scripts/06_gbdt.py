#!/usr/bin/env python3
"""Stage 06: XGBoost pair scorer + decide() -> v1 submission.

Trains on T, early-stops on Vcal, calibrates on Vcal, tunes the
expected-F0.5 probability floor tau on Vcal, then applies the same decide()
to Vtest (for a held-out reported score) and to the full test set.

Memory (16GB laptop): train features are read as float32 and split into
T/Vcal/Vtest frames straight away; the test features (up to ~86M rows at
CNP k=50, several GB) are never loaded whole -- they're predicted in
`--predict-batch` row batches, keeping only compact id columns and the
score.

Writes work/06_gbdt/matching_results.tsv and report.json (Vcal/Vtest macro
F0.5 overall and per country, the tuned tau, and the LOCO diagnostic).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import add_common_args, paths_from_args
from er.decide import run_tier
from er.gbdt import leave_one_country_out, predict, train_gbdt
from er.io import mark_stage_done, release_memory, stage_is_done
from er.metrics import build_truth_dict
from er.submit import validate, write_matching_results

OUTPUT_VERSION = 2
ID_COLS = ["s1_id_num", "match_source", "match_id_num"]
NON_FEATURE_COLS = {"s1_id_num", "match_source", "match_id_num", "label", "split", "country"}


def feature_cols(schema_names: list[str]) -> list[str]:
    return [c for c in schema_names if c not in NON_FEATURE_COLS]


def compact_ids(df: pd.DataFrame) -> pd.DataFrame:
    """ids fit in int32 (all < 1e9); halves the id columns of the big frames."""
    return pd.DataFrame({
        "s1_id_num": df["s1_id_num"].to_numpy(np.int32),
        "match_source": df["match_source"].to_numpy(np.int8),
        "match_id_num": df["match_id_num"].to_numpy(np.int32),
    })


def load_split(path: Path, split: str, cols: list[str], country_of: pd.Series, batch_rows: int = 1_000_000):
    """One split of the train features: (X float32 array, y int8, compact id
    frame, country array). The split's rows are counted first, then batches
    are streamed into a preallocated X -- a filtered `read_parquet` instead
    materializes and fragments several GB for the ~14M-row T split."""
    pf = pq.ParquetFile(path)
    n = int(sum((b.column("split").to_numpy(zero_copy_only=False) == split).sum()
                for b in pf.iter_batches(batch_size=batch_rows, columns=["split"])))
    X = np.empty((n, len(cols)), dtype=np.float32)
    y = np.empty(n, dtype=np.int8)
    ids = []
    pos = 0
    for batch in pf.iter_batches(batch_size=batch_rows, columns=ID_COLS + cols + ["label", "split"]):
        df = batch.to_pandas()
        df = df[df["split"] == split]
        k = len(df)
        X[pos:pos + k] = df[cols].to_numpy(np.float32)
        y[pos:pos + k] = df["label"].to_numpy(np.int8)
        ids.append(compact_ids(df))
        pos += k
    ids = pd.concat(ids, ignore_index=True)
    country = pd.Series(ids["s1_id_num"].to_numpy(np.int64)).map(country_of).to_numpy()
    release_memory()
    return X, y, ids, country


def predict_test(booster, path: Path, cols: list[str], batch_rows: int) -> pd.DataFrame:
    parts = []
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=ID_COLS + cols):
        df = batch.to_pandas()
        part = compact_ids(df)
        part["raw_score"] = predict(booster, df[cols]).astype(np.float32)
        parts.append(part)
        del df
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--predict-batch", type=int, default=2_000_000, help="test rows predicted per batch")
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    load_dir = paths.stage_dir("01_load")
    feat_dir = paths.stage_dir("05_features")
    block_dir = paths.stage_dir("04_blocking")
    stage_dir = paths.stage_dir("06_gbdt")
    repo_root = paths.data_dir.resolve().parent

    config = {"data_dir": str(paths.data_dir), "seed": args.seed, "output_version": OUTPUT_VERSION}
    if stage_is_done(stage_dir, config):
        print("06_gbdt: already done, skipping")
        return 0

    print("loading features...")
    train_path = feat_dir / "train_features.parquet"
    cols = feature_cols(pq.read_schema(train_path).names)
    s1_train = pd.read_parquet(load_dir / "s1_train.parquet", columns=["entity_id_num", "country", "split"])
    s1_train = s1_train[s1_train["split"].isin(["T", "Vcal", "Vtest"])]
    country_of = s1_train.set_index("entity_id_num")["country"]
    X_t, y_t, _, country_t = load_split(train_path, "T", cols, country_of)
    X_cal, y_cal, vcal_df, _ = load_split(train_path, "Vcal", cols, country_of)
    X_vt, y_vt, vtest_df, _ = load_split(train_path, "Vtest", cols, country_of)
    gt_pairs = pd.read_parquet(load_dir / "gt_pairs.parquet")
    s1_test_ids = pd.read_parquet(load_dir / "s1_test.parquet", columns=["entity_id_num"])["entity_id_num"].to_numpy()

    print(f"training GBDT on {len(y_t)} T rows, early-stopping on {len(y_cal)} Vcal rows...", flush=True)
    booster = train_gbdt(X_t, y_t, X_cal, y_cal, feature_names=cols)

    print("leave-one-country-out diagnostic (US <-> India)...", flush=True)
    loco = leave_one_country_out(X_t, y_t, country_t, cols, "US", "India")
    print(f"  {loco}")
    del X_t, y_t, country_t
    release_memory()

    vcal_df["raw_score"] = predict(booster, X_cal).astype(np.float32)
    vcal_df["label"] = y_cal
    vtest_df["raw_score"] = predict(booster, X_vt).astype(np.float32)
    vtest_df["label"] = y_vt
    del X_cal, X_vt

    print("predicting test features in batches...", flush=True)
    test_df = predict_test(booster, feat_dir / "test_features.parquet", cols, args.predict_batch)
    release_memory()
    print(f"  {len(test_df)} test rows scored")

    vcal_ids = s1_train.loc[s1_train["split"] == "Vcal", "entity_id_num"]
    vtest_ids = s1_train.loc[s1_train["split"] == "Vtest", "entity_id_num"]
    vcal_truth = build_truth_dict(gt_pairs, vcal_ids)
    vtest_truth = build_truth_dict(gt_pairs, vtest_ids)
    country_of = country_of.to_dict()
    del gt_pairs, s1_train

    print("calibrating + tuning tau on Vcal...", flush=True)
    result = run_tier(vcal_df, vtest_df, test_df, ["raw_score"], s1_test_ids, country_of, vcal_truth, vtest_truth)
    print(f"  tau = {result['tau']}, Vcal macro F0.5 = {result['vcal_f05']:.4f}")
    print(f"  Vtest macro F0.5 = {result['vtest_f05']:.4f} {result['vtest_f05_by_country']}")

    id_cols = ID_COLS + ["raw_score", "prob"]
    for name in ("vcal", "vtest", "test"):
        df = result[f"{name}_df"]
        df["prob"] = df["prob"].astype(np.float32)
        df[id_cols].to_parquet(stage_dir / f"{name}_scored.parquet", index=False)

    matching_path = stage_dir / "matching_results.tsv"
    write_matching_results(result["test_preds"], matching_path)
    # Drop the big per-pair frames before validating (validation reads the
    # full test files back; both at once is this stage's memory peak).
    summary = {k: result[k] for k in ("tau", "vcal_f05", "vcal_f05_by_country", "vtest_f05", "vtest_f05_by_country")}
    del result, test_df, vcal_df, vtest_df
    release_memory()
    ok, errors, warnings = validate(matching_path, block_dir / "candidate_pairs.tsv", paths.test_dir, repo_root)
    for e in errors:
        print(f"VALIDATION ERROR: {e}")
    for w in warnings:
        print(f"VALIDATION WARNING: {w}")

    report = {
        **summary,
        "loco": loco, "validated": ok,
    }
    with open(stage_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    if not ok:
        print("06_gbdt: validation FAILED")
        return 1

    mark_stage_done(stage_dir, config)
    print("06_gbdt: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
