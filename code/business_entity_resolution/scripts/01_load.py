#!/usr/bin/env python3
"""Stage 01: load the raw TSVs, encode ids as int64, split train S1 into
T / Vcal / Vtest, and cache everything as parquet under work/01_load/.

Outputs (all under `<work-dir>/01_load/`):
  s1_train.parquet   entity_id_num, business_name, business_address, country, split
  s2_train.parquet   entity_id_num, business_name, business_address, country
  s3_train.parquet   (same shape as s2_train)
  gt_pairs.parquet   s1_id_num, match_source, match_id_num  (one row per true pair)
  s1_test.parquet, s2_test.parquet, s3_test.parquet  (same shape, no split/gt)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import N_TRAIN, N_VCAL, N_VTEST, add_common_args, paths_from_args
from er.io import encode_ids, mark_stage_done, read_source_tsv, stage_is_done
from er.splits import make_splits


def load_source(path: Path) -> pd.DataFrame:
    df = read_source_tsv(path)
    ids = encode_ids(df["entity_id"])
    out = pd.DataFrame({
        "entity_id_num": ids["id_num"],
        "business_name": df["business_name"],
        "business_address": df["business_address"],
        "country": df["country"],
    })
    return out


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = read_source_tsv(path)
    s1_ids = encode_ids(df["source1_entity_id"])["id_num"]
    matches = df["matched_entity_ids"].str.split(",")
    exploded = pd.DataFrame({
        "s1_id_num": s1_ids,
        "match_id": matches,
    }).explode("match_id")
    exploded = exploded[exploded["match_id"].notna() & (exploded["match_id"] != "")]
    match_ids = encode_ids(exploded["match_id"])
    return pd.DataFrame({
        "s1_id_num": exploded["s1_id_num"].to_numpy(),
        "match_source": match_ids["source"].to_numpy(),
        "match_id_num": match_ids["id_num"].to_numpy(),
    })


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--n-train", type=int, default=N_TRAIN, help="T split size (override for a small smoke-test dataset)")
    p.add_argument("--n-vcal", type=int, default=N_VCAL)
    p.add_argument("--n-vtest", type=int, default=N_VTEST)
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    stage_dir = paths.stage_dir("01_load")

    config = {"data_dir": str(paths.data_dir), "seed": args.seed, "n_train": args.n_train, "n_vcal": args.n_vcal, "n_vtest": args.n_vtest}
    if stage_is_done(stage_dir, config):
        print("01_load: already done, skipping")
        return 0

    print("loading train source files...")
    s1_train = load_source(paths.train_dir / "train_source1.tsv")
    s2_train = load_source(paths.train_dir / "train_source2.tsv")
    s3_train = load_source(paths.train_dir / "train_source3.tsv")
    gt_pairs = load_ground_truth(paths.train_dir / "train_ground_truth.tsv")

    print("computing T/Vcal/Vtest split...")
    s1_train = s1_train.reset_index(drop=True)
    s1_train["split"] = make_splits(s1_train, seed=args.seed, n_train=args.n_train, n_vcal=args.n_vcal, n_vtest=args.n_vtest).to_numpy()

    print("loading test source files...")
    s1_test = load_source(paths.test_dir / "test_source1.tsv")
    s2_test = load_source(paths.test_dir / "test_source2.tsv")
    s3_test = load_source(paths.test_dir / "test_source3.tsv")

    print("writing parquet...")
    s1_train.to_parquet(stage_dir / "s1_train.parquet", index=False)
    s2_train.to_parquet(stage_dir / "s2_train.parquet", index=False)
    s3_train.to_parquet(stage_dir / "s3_train.parquet", index=False)
    gt_pairs.to_parquet(stage_dir / "gt_pairs.parquet", index=False)
    s1_test.to_parquet(stage_dir / "s1_test.parquet", index=False)
    s2_test.to_parquet(stage_dir / "s2_test.parquet", index=False)
    s3_test.to_parquet(stage_dir / "s3_test.parquet", index=False)

    for name, split_df in (("T", s1_train), ("Vcal", s1_train), ("Vtest", s1_train)):
        n = (split_df["split"] == name).sum()
        print(f"split {name}: {n} S1 entities")

    mark_stage_done(stage_dir, config)
    print("01_load: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
