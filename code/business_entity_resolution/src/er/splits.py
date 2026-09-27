"""Country-stratified T / Vcal / Vtest split of train Source-1 entities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_TRAIN, N_VCAL, N_VTEST, SEED


def make_splits(
    s1: pd.DataFrame, seed: int = SEED, n_train: int = N_TRAIN, n_vcal: int = N_VCAL, n_vtest: int = N_VTEST,
) -> pd.Series:
    """Return a 'split' label per row of `s1` (which must have a 'country'
    column), one of {'T', 'Vcal', 'Vtest', ''}.

    The split is a country-stratified random subsample of size
    n_train / n_vcal / n_vtest, proportional to each country's share of
    train S1. Every other row is left unassigned ('') and is still used by
    stage 04's "competing S1" blocking statistics, just never fit or tuned
    on (see Implementation_Plan.md stage 04 and stage 01). The sizes are
    parameters (not always config.py's full-scale defaults) so
    tools/make_sample.py's small smoke-test dataset can use a matching
    small split.
    """
    n_needed = n_train + n_vcal + n_vtest
    if len(s1) < n_needed:
        raise ValueError(f"need >= {n_needed} train S1 rows, got {len(s1)}")

    rng = np.random.default_rng(seed)
    split = pd.Series("", index=s1.index, dtype=object)

    for _, group in s1.groupby("country", sort=True):
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)
        frac = len(idx) / len(s1)
        n_t = round(n_train * frac)
        n_cal = round(n_vcal * frac)
        n_test = round(n_vtest * frac)
        split.loc[idx[:n_t]] = "T"
        split.loc[idx[n_t:n_t + n_cal]] = "Vcal"
        split.loc[idx[n_t + n_cal:n_t + n_cal + n_test]] = "Vtest"

    return split
