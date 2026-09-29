"""Helpers shared by the pair-rescoring tiers (stages 07/08): labels, the
within-S1 rank, and record text lookup for just the pairs being scored.

Text lookup keeps one scope's name/address columns as compact
pyarrow-backed pandas columns and gathers rows by position, instead of a
Python dict of every S1/S2/S3 record (tens of millions of dicts, which is
what made the original stages 07/08 need >16GB of RAM).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import load_pool, load_s1, pair_key

TEXT_COLS = ["entity_id_num", "name_clean", "address_clean", "legal_form"]


def triple_key(s1, source, mid) -> np.ndarray:
    """One int64 per (s1, source, id) pair; all ids are < 2**30."""
    return (np.asarray(s1, np.int64) << 32) | (np.asarray(source, np.int64) << 30) | np.asarray(mid, np.int64)


def label_pairs(df: pd.DataFrame, gt_pairs: pd.DataFrame) -> np.ndarray:
    truth = np.unique(triple_key(gt_pairs["s1_id_num"], gt_pairs["match_source"], gt_pairs["match_id_num"]))
    keys = triple_key(df["s1_id_num"], df["match_source"], df["match_id_num"])
    if len(truth) == 0:
        return np.zeros(len(df), dtype=np.int8)
    pos = np.searchsorted(truth, keys).clip(max=len(truth) - 1)
    return (truth[pos] == keys).astype(np.int8)


def rank_in_s1(df: pd.DataFrame, prob_col: str = "prob") -> np.ndarray:
    """1-based rank of each row's prob within its S1 (descending, ties in
    row order) -- `groupby().rank(ascending=False, method="first")`."""
    s1 = df["s1_id_num"].to_numpy()
    order = np.lexsort((-df[prob_col].to_numpy(), s1))
    s1_sorted = s1[order]
    starts = np.concatenate([[0], np.flatnonzero(s1_sorted[1:] != s1_sorted[:-1]) + 1])
    group_start = np.repeat(starts, np.diff(np.append(starts, len(s1))))
    rank = np.empty(len(s1), dtype=np.int64)
    rank[order] = np.arange(len(s1)) - group_start + 1
    return rank


def closest_to_half(sel: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    if len(sel) <= max_pairs:
        return sel
    dist = np.abs(sel["prob"].to_numpy() - 0.5)
    return sel.iloc[np.sort(np.argsort(dist, kind="stable")[:max_pairs])]


def set_prob(df: pd.DataFrame, sel: pd.DataFrame, refined: np.ndarray) -> None:
    """Overwrite `prob` for the rows of `sel` in place (no copy of what can
    be an ~86M-row frame). Matches the column's dtype explicitly: pandas >= 3
    raises on a lossy cross-precision `.loc` assignment."""
    df.loc[sel.index, "prob"] = np.asarray(refined, dtype=df["prob"].dtype)


class RecordTexts:
    """name_clean/address_clean for one scope's S1 and pool records."""

    _FIELDS = ["name_clean", "address_clean", "legal_form"]

    def __init__(self, norm_dir: Path, scope: str):
        s1 = load_s1(norm_dir, scope, TEXT_COLS, parse_numbers=False)
        self._s1_index = pd.Index(s1["entity_id_num"].to_numpy())
        self._s1 = s1[self._FIELDS].fillna("")
        pool = load_pool(norm_dir, scope, TEXT_COLS, parse_numbers=False)
        self._pool_index = pd.Index(pair_key(pool["source"], pool["entity_id_num"]))
        self._pool = pool[self._FIELDS].fillna("")

    @classmethod
    def _records(cls, table: pd.DataFrame, rows: np.ndarray) -> list[dict | None]:
        ok = rows >= 0
        sub = table.iloc[rows[ok]]
        cols = {c: sub[c].tolist() for c in cls._FIELDS}
        found = iter([dict(zip(cols, vals)) for vals in zip(*cols.values())])
        return [next(found) if o else None for o in ok.tolist()]

    def s1_records(self, s1_ids) -> list[dict | None]:
        return self._records(self._s1, self._s1_index.get_indexer(np.asarray(s1_ids)))

    def pool_records(self, source, id_num) -> list[dict | None]:
        return self._records(self._pool, self._pool_index.get_indexer(pair_key(source, id_num)))

    def pair_records(self, df: pd.DataFrame) -> tuple[list[dict | None], list[dict | None]]:
        return self.s1_records(df["s1_id_num"]), self.pool_records(df["match_source"], df["match_id_num"])
