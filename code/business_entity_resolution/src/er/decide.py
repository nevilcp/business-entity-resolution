"""Turn per-pair match probabilities into a submission.

The same `decide()` is used for every tier (v1 GBDT-only, v2 +cross-encoder,
v3 +LLM judge) -- only which probability column feeds it changes. Steps:
1. calibrate raw model scores to P(match) with a 1-D logistic regression
   fit on Vcal (Platt scaling);
2. each S2/S3 record keeps only its highest-probability S1 (a measured data
   fact: every S2/S3 record matches at most one S1);
3. per S1 entity, sort candidates by probability and pick the prefix size m
   that maximizes expected F0.5 under an independence assumption, then trim
   to a minimum-probability floor tau tuned on Vcal.

Steps 2-3 run on numpy arrays (a lexsort plus a numba pass over the sorted
groups) rather than pandas groupby: the test side is up to ~86M candidate
rows at CNP k=50, where a two-key `groupby().idxmax()` and a Python loop
over 1.7M `groupby` groups cost several GB and tens of minutes.
"""
from __future__ import annotations

import numba
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .io import pair_key
from .metrics import macro_f05, macro_f05_by_country

NO_MATCH: frozenset = frozenset()  # shared by every S1 predicted to have no match

BETA2 = 0.25


def _as_2d(raw_scores: np.ndarray) -> np.ndarray:
    return raw_scores.reshape(-1, 1) if raw_scores.ndim == 1 else raw_scores


def calibrate_scores(raw_scores: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit the per-tier logistic-regression stacker on Vcal. `raw_scores` is
    either one column (stage 06, the GBDT's own score) or several (stages
    07/08, [previous-tier probability, this tier's new raw score]) -- this
    is what makes it a *stacker* rather than a plain average of tiers."""
    lr = LogisticRegression()
    lr.fit(_as_2d(raw_scores), labels)
    return lr


def apply_calibration(lr: LogisticRegression, raw_scores: np.ndarray) -> np.ndarray:
    """P(match) as float32 (the test side has ~80M rows; float64 would add
    ~0.3GB per copy for no decision-relevant precision)."""
    return lr.predict_proba(_as_2d(raw_scores))[:, 1].astype(np.float32)


def enforce_record_uniqueness(df: pd.DataFrame, prob_col: str = "prob") -> pd.DataFrame:
    """Each (match_source, match_id_num) record keeps only its
    highest-probability S1 row; every other row referencing that record is
    dropped. Ties keep the earliest row, like `groupby().idxmax()`.
    Temporaries are freed as soon as they're used: at ~80M test rows each
    one is ~0.6GB."""
    key = pair_key(df["match_source"].to_numpy(), df["match_id_num"].to_numpy())
    neg_prob = np.negative(df[prob_col].to_numpy())
    order = np.lexsort((neg_prob, key))
    del neg_prob
    k_sorted = key[order]
    del key
    first = np.empty(len(order), dtype=bool)
    first[:1] = True
    np.not_equal(k_sorted[1:], k_sorted[:-1], out=first[1:])
    del k_sorted
    keep = order[first]
    del order, first
    keep.sort()
    return df.iloc[keep]


def expected_f05_prefix(probs_sorted_desc: np.ndarray, beta2: float = BETA2) -> tuple[int, float]:
    """(m, expected F0.5) maximizing expected F0.5 over prefix sizes m, under
    independence: for a fixed prefix, E[TP] = sum(p_i in prefix), precision =
    E[TP]/m, recall = E[TP]/E[total true matches] (approximated by the sum of
    ALL candidate probabilities). m=0's score is P(no true match at all) =
    prod(1 - p_i), the standard baseline for the "predict nothing" option.
    """
    n = len(probs_sorted_desc)
    if n == 0:
        return 0, 1.0
    best_m = 0
    best_score = float(np.prod(1.0 - probs_sorted_desc))
    total_expected = float(probs_sorted_desc.sum())
    cum = 0.0
    for m in range(1, n + 1):
        cum += probs_sorted_desc[m - 1]
        if total_expected <= 0:
            break
        precision = cum / m
        recall = cum / total_expected
        denom = beta2 * precision + recall
        score = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        if score > best_score:
            best_score, best_m = score, m
    return best_m, best_score


def decide_entity(probs_sorted_desc: np.ndarray, tau: float, beta2: float = BETA2) -> int:
    m, _ = expected_f05_prefix(probs_sorted_desc, beta2)
    while m > 0 and probs_sorted_desc[m - 1] < tau:
        m -= 1
    return m


@numba.njit(cache=True)
def _prefix_sizes(probs: np.ndarray, starts: np.ndarray, beta2: float) -> np.ndarray:
    """`expected_f05_prefix`'s m for every group of `probs` (sorted
    descending within each group; group g is probs[starts[g]:starts[g+1]])."""
    n_groups = starts.shape[0] - 1
    out = np.zeros(n_groups, dtype=np.int64)
    for g in range(n_groups):
        lo, hi = starts[g], starts[g + 1]
        best_m = 0
        best_score = 1.0
        total = 0.0
        for i in range(lo, hi):
            best_score *= 1.0 - probs[i]
            total += probs[i]
        cum = 0.0
        for m in range(1, hi - lo + 1):
            cum += probs[lo + m - 1]
            if total <= 0:
                break
            precision = cum / m
            recall = cum / total
            denom = beta2 * precision + recall
            score = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
            if score > best_score:
                best_score = score
                best_m = m
        out[g] = best_m
    return out


@numba.njit(cache=True)
def _trim_to_tau(probs: np.ndarray, starts: np.ndarray, m: np.ndarray, tau: float) -> np.ndarray:
    out = m.copy()
    for g in range(m.shape[0]):
        while out[g] > 0 and probs[starts[g] + out[g] - 1] < tau:
            out[g] -= 1
    return out


class _Groups:
    """`df_deduped` sorted by (s1, prob desc) once, with each S1's
    expected-F0.5 prefix size, so deciding at many taus is cheap. Probs stay
    in their stored precision (float32 for the big test frame); the numba
    kernels accumulate in float64."""

    def __init__(self, df_deduped: pd.DataFrame, prob_col: str, beta2: float):
        s1 = df_deduped["s1_id_num"].to_numpy()
        prob = df_deduped[prob_col].to_numpy()
        self.order = np.lexsort((np.negative(prob), s1))
        self.s1 = s1[self.order]
        self.probs = prob[self.order]
        self._src = df_deduped["match_source"].to_numpy()
        self._mid = df_deduped["match_id_num"].to_numpy()
        change = np.flatnonzero(self.s1[1:] != self.s1[:-1]) + 1
        self.starts = np.concatenate([[0], change, [len(self.s1)]]).astype(np.int64)
        self.m = _prefix_sizes(self.probs, self.starts, beta2)

    def decide(self, tau: float) -> dict[int, set[tuple[int, int]]]:
        m = _trim_to_tau(self.probs, self.starts, self.m, float(tau))
        group_s1 = self.s1[self.starts[:-1]].tolist()
        result: dict[int, set[tuple[int, int]]] = dict.fromkeys(group_s1, NO_MATCH)
        for g in np.flatnonzero(m > 0).tolist():
            rows = self.order[self.starts[g]:self.starts[g] + m[g]]
            result[group_s1[g]] = set(zip(self._src[rows].astype(int).tolist(), self._mid[rows].astype(int).tolist()))
        return result


def decide(
    df_deduped: pd.DataFrame, tau: float, prob_col: str = "prob", beta2: float = BETA2,
) -> dict[int, set[tuple[int, int]]]:
    """`df_deduped` must already have gone through `enforce_record_uniqueness`.
    S1s predicted to have no match map to the shared empty `NO_MATCH`."""
    return _Groups(df_deduped, prob_col, beta2).decide(tau)


def tune_tau(
    df_vcal_deduped: pd.DataFrame,
    truth: dict[int, set[tuple[int, int]]],
    prob_col: str = "prob",
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.05)
    groups = _Groups(df_vcal_deduped, prob_col, BETA2)
    best_tau, best_score = 0.5, -1.0
    for tau in grid:
        preds = groups.decide(float(tau))
        score = macro_f05(preds, truth)
        if score > best_score:
            best_score, best_tau = score, float(tau)
    return best_tau, best_score


def finish_tier(
    vcal_df: pd.DataFrame,
    vtest_df: pd.DataFrame,
    test_df: pd.DataFrame,
    s1_test_ids,
    country_of: dict[int, str],
    vcal_truth: dict[int, set[tuple[int, int]]],
    vtest_truth: dict[int, set[tuple[int, int]]],
    prob_col: str = "prob",
) -> dict:
    """The finish every tier (06/07/08) shares, given each row's final
    'prob' already computed: dedupe by record, tune tau on Vcal, and decide
    Vcal/Vtest/test. `vcal_df`/`vtest_df` need a 'label' column; `test_df`
    must not (it's the real, unlabeled submission set).
    """
    vcal_deduped = enforce_record_uniqueness(vcal_df, prob_col)
    vtest_deduped = enforce_record_uniqueness(vtest_df, prob_col)
    test_deduped = enforce_record_uniqueness(test_df, prob_col)

    tau, vcal_f05 = tune_tau(vcal_deduped, vcal_truth, prob_col)
    vcal_preds = decide(vcal_deduped, tau, prob_col)
    vcal_f05_country = macro_f05_by_country(vcal_preds, vcal_truth, country_of)

    vtest_preds = decide(vtest_deduped, tau, prob_col)
    vtest_f05 = macro_f05(vtest_preds, vtest_truth)
    vtest_f05_country = macro_f05_by_country(vtest_preds, vtest_truth, country_of)

    test_preds = decide(test_deduped, tau, prob_col)
    for s1_id in np.asarray(s1_test_ids).tolist():
        test_preds.setdefault(int(s1_id), NO_MATCH)

    return {
        "tau": tau, "vcal_f05": vcal_f05, "vcal_f05_by_country": vcal_f05_country,
        "vtest_f05": vtest_f05, "vtest_f05_by_country": vtest_f05_country,
        "test_preds": test_preds, "vcal_df": vcal_df, "vtest_df": vtest_df, "test_df": test_df,
    }


def run_tier(
    vcal_df: pd.DataFrame,
    vtest_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_cols: list[str],
    s1_test_ids,
    country_of: dict[int, str],
    vcal_truth: dict[int, set[tuple[int, int]]],
    vtest_truth: dict[int, set[tuple[int, int]]],
) -> dict:
    """Stage 06's case: fit the calibration stacker on all of Vcal (one raw
    column, the GBDT's own score) and apply it uniformly, then `finish_tier`.
    Stages 07/08 instead calibrate only the subset of rows their model
    actually scored and call `finish_tier` directly -- see cross_encoder.py.
    """

    def raw(df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([df[c].to_numpy() for c in raw_cols]) if len(raw_cols) > 1 else df[raw_cols[0]].to_numpy()

    lr = calibrate_scores(raw(vcal_df), vcal_df["label"].to_numpy())

    vcal_df = vcal_df.copy()
    vtest_df = vtest_df.copy()
    # test_df gets its 'prob' column added in place: it can be ~86M rows,
    # and a defensive copy would double the stage's peak memory.
    vcal_df["prob"] = apply_calibration(lr, raw(vcal_df))
    vtest_df["prob"] = apply_calibration(lr, raw(vtest_df))
    test_df["prob"] = apply_calibration(lr, raw(test_df))

    return finish_tier(vcal_df, vtest_df, test_df, s1_test_ids, country_of, vcal_truth, vtest_truth)
