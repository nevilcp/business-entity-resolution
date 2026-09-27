#!/usr/bin/env python3
"""Build a small local smoke-test dataset: about 3k train S1 with their
matches plus distractor S2/S3 records, and about 2k test S1 across all 3
countries with their own candidate pool. Mirrors the real dataset's TSV
layout so the full pipeline (run_all.sh with small --n-train/--n-vcal/
--n-vtest) can run against it end-to-end on a laptop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def stratified_ids(s1: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    ids = []
    for _, group in s1.groupby("country"):
        take = min(len(group), max(1, round(n * len(group) / len(s1))))
        idx = rng.choice(group.index.to_numpy(), size=take, replace=False)
        ids.extend(group.loc[idx, "entity_id"])
    return ids


def sample_train(data_dir: Path, out_dir: Path, n_s1: int, distractor_ratio: int, rng: np.random.Generator) -> None:
    s1 = read_tsv(data_dir / "train/train_source1.tsv")
    gt = read_tsv(data_dir / "train/train_ground_truth.tsv")

    sampled_ids = set(stratified_ids(s1, n_s1, rng))
    s1_sample = s1[s1["entity_id"].isin(sampled_ids)]
    gt_sample = gt[gt["source1_entity_id"].isin(sampled_ids)]

    match_ids = set()
    for cell in gt_sample["matched_entity_ids"]:
        if cell:
            match_ids.update(cell.split(","))

    s2 = read_tsv(data_dir / "train/train_source2.tsv")
    s3 = read_tsv(data_dir / "train/train_source3.tsv")
    s2_match, s3_match = s2[s2["entity_id"].isin(match_ids)], s3[s3["entity_id"].isin(match_ids)]

    n_distractor = distractor_ratio * len(s1_sample)
    s2_pool = s2[~s2["entity_id"].isin(match_ids)]
    s3_pool = s3[~s3["entity_id"].isin(match_ids)]
    s2_distract = s2_pool.sample(n=min(n_distractor, len(s2_pool)), random_state=rng.integers(1 << 31))
    s3_distract = s3_pool.sample(n=min(n_distractor, len(s3_pool)), random_state=rng.integers(1 << 31))

    s2_sample = pd.concat([s2_match, s2_distract]).drop_duplicates("entity_id")
    s3_sample = pd.concat([s3_match, s3_distract]).drop_duplicates("entity_id")

    out_train = out_dir / "train"
    out_train.mkdir(parents=True, exist_ok=True)
    s1_sample.to_csv(out_train / "train_source1.tsv", sep="\t", index=False)
    s2_sample.to_csv(out_train / "train_source2.tsv", sep="\t", index=False)
    s3_sample.to_csv(out_train / "train_source3.tsv", sep="\t", index=False)
    gt_sample.to_csv(out_train / "train_ground_truth.tsv", sep="\t", index=False)
    print(f"  train: {len(s1_sample)} S1, {len(s2_sample)} S2, {len(s3_sample)} S3, {len(gt_sample)} ground-truth rows")


def sample_test(data_dir: Path, out_dir: Path, n_s1: int, distractor_ratio: int, rng: np.random.Generator) -> None:
    s1 = read_tsv(data_dir / "test/test_source1.tsv")
    sampled_ids = set(stratified_ids(s1, n_s1, rng))
    s1_sample = s1[s1["entity_id"].isin(sampled_ids)]

    s2 = read_tsv(data_dir / "test/test_source2.tsv")
    s3 = read_tsv(data_dir / "test/test_source3.tsv")
    countries = s1_sample["country"].unique()
    n_per_country = max(50, distractor_ratio * len(s1_sample) // max(1, len(countries)))

    s2_parts, s3_parts = [], []
    for country in countries:
        s2c, s3c = s2[s2["country"] == country], s3[s3["country"] == country]
        s2_parts.append(s2c.sample(n=min(n_per_country, len(s2c)), random_state=rng.integers(1 << 31)))
        s3_parts.append(s3c.sample(n=min(n_per_country, len(s3c)), random_state=rng.integers(1 << 31)))

    s2_sample = pd.concat(s2_parts).drop_duplicates("entity_id") if s2_parts else s2.iloc[0:0]
    s3_sample = pd.concat(s3_parts).drop_duplicates("entity_id") if s3_parts else s3.iloc[0:0]

    out_test = out_dir / "test"
    out_test.mkdir(parents=True, exist_ok=True)
    s1_sample.to_csv(out_test / "test_source1.tsv", sep="\t", index=False)
    s2_sample.to_csv(out_test / "test_source2.tsv", sep="\t", index=False)
    s3_sample.to_csv(out_test / "test_source3.tsv", sep="\t", index=False)
    print(f"  test: {len(s1_sample)} S1 ({list(countries)}), {len(s2_sample)} S2, {len(s3_sample)} S3")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="../../dataset")
    p.add_argument("--out-dir", default="../../dataset_sample")
    p.add_argument("--n-train-s1", type=int, default=3000)
    p.add_argument("--n-test-s1", type=int, default=2000)
    p.add_argument("--distractor-ratio", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    print("sampling train...")
    sample_train(data_dir, out_dir, args.n_train_s1, args.distractor_ratio, rng)
    print("sampling test...")
    sample_test(data_dir, out_dir, args.n_test_s1, args.distractor_ratio, rng)
    print(f"wrote sample dataset to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
