"""The numpy/numba decide path must reproduce the straightforward pandas
groupby implementation it replaced."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.decide import decide, decide_entity, enforce_record_uniqueness, tune_tau


def _random_pairs(seed: int, n_s1: int = 300, n_rec: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s1 in range(n_s1):
        for _ in range(rng.integers(0, 8)):
            rows.append((s1 + 1000, int(rng.integers(2, 4)), int(rng.integers(0, n_rec)), float(rng.random() ** 3)))
    df = pd.DataFrame(rows, columns=["s1_id_num", "match_source", "match_id_num", "prob"])
    return df.drop_duplicates(["s1_id_num", "match_source", "match_id_num"]).reset_index(drop=True)


def _reference_dedupe(df):
    idx = df.groupby(["match_source", "match_id_num"])["prob"].idxmax()
    return df.loc[idx]


def _reference_decide(df, tau):
    out = {}
    for s1, g in df.groupby("s1_id_num"):
        g = g.sort_values("prob", ascending=False, kind="stable")
        m = decide_entity(g["prob"].to_numpy(), tau)
        out[int(s1)] = set(zip(g["match_source"].iloc[:m].astype(int), g["match_id_num"].iloc[:m].astype(int)))
    return out


def test_enforce_record_uniqueness_matches_groupby_idxmax():
    df = _random_pairs(0)
    got = enforce_record_uniqueness(df).sort_index()
    want = _reference_dedupe(df).sort_index()
    pd.testing.assert_frame_equal(got, want)


def test_decide_matches_reference_at_several_taus():
    df = enforce_record_uniqueness(_random_pairs(1))
    for tau in (0.05, 0.3, 0.6):
        got = decide(df, tau)
        want = _reference_decide(df, tau)
        assert {k: set(v) for k, v in got.items()} == want


def test_tune_tau_returns_a_grid_value():
    df = enforce_record_uniqueness(_random_pairs(2))
    truth = {int(s): set() for s in df["s1_id_num"].unique()}
    tau, score = tune_tau(df, truth)
    assert 0.05 - 1e-9 <= tau <= 0.95 + 1e-9 and 0.0 <= score <= 1.0
