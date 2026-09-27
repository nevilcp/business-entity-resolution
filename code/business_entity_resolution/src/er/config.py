"""Shared paths, constants and CLI wiring for the ER pipeline."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

SEED = 2026

# Sizes of the stratified subsample of train S1 entities used for training
# and tuning (see splits.py). The remaining train S1 entities are still
# blocked for stage-04's "competing S1" statistics, just never fit or tuned
# on (see Implementation_Plan.md stage 04).
N_TRAIN = 300_000
N_VCAL = 50_000
N_VTEST = 50_000

SOURCE_CODE = {"S1": 1, "S2": 2, "S3": 3}
CODE_SOURCE = {v: k for k, v in SOURCE_CODE.items()}

# CNP keeps the top-k candidates per S1; the plan picks the smallest of these
# whose Vcal recall is within 0.001 of recall at 50.
CNP_K_GRID = (10, 15, 20, 30, 40, 50)


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    out_dir: Path
    work_dir: Path

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    def stage_dir(self, stage: str) -> Path:
        d = self.work_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--data-dir", default="../../dataset")
    parser.add_argument("--out-dir", default="../../output")
    parser.add_argument("--work-dir", default="../../work")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


def paths_from_args(args: argparse.Namespace) -> Paths:
    return Paths(
        data_dir=Path(args.data_dir),
        out_dir=Path(args.out_dir),
        work_dir=Path(args.work_dir),
    )
