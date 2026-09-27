"""Per-entity F_0.5 and macro-averaging, matching the README's scoring rule."""
from __future__ import annotations

import pandas as pd

BETA2 = 0.5 ** 2


def build_truth_dict(gt_pairs: pd.DataFrame, s1_ids) -> dict[int, set[tuple[int, int]]]:
    """Truth dict for every id in `s1_ids`, including entities with no true
    match (an empty set) -- needed so singletons still count in macro_f05."""
    s1_ids = {int(i) for i in s1_ids}
    truth: dict[int, set[tuple[int, int]]] = {i: set() for i in s1_ids}
    sub = gt_pairs[gt_pairs["s1_id_num"].isin(s1_ids)]
    for row in sub.itertuples(index=False):
        truth[int(row.s1_id_num)].add((int(row.match_source), int(row.match_id_num)))
    return truth


def f05(predicted: set, truth: set) -> float:
    """Per-S1-entity F_0.5.

    Singleton rule: empty truth + empty prediction -> 1.0; empty truth +
    any prediction -> 0.0 (a false match on a true singleton scores 0).
    """
    if not truth:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return (1 + BETA2) * precision * recall / (BETA2 * precision + recall)


def macro_f05(predictions: dict[str, set], truths: dict[str, set]) -> float:
    """Macro-average F_0.5 over every S1 id in `truths` (the eval set)."""
    if not truths:
        return 0.0
    scores = [f05(predictions.get(s1, set()), truth) for s1, truth in truths.items()]
    return sum(scores) / len(scores)


def macro_f05_by_country(
    predictions: dict[str, set],
    truths: dict[str, set],
    country_of: dict[str, str],
) -> dict[str, float]:
    """Same as macro_f05, but broken out per country (for the gating check
    and the per-country F0.5 the plan asks summary.json to report).
    """
    by_country: dict[str, list[float]] = {}
    for s1, truth in truths.items():
        score = f05(predictions.get(s1, set()), truth)
        by_country.setdefault(country_of[s1], []).append(score)
    return {c: sum(v) / len(v) for c, v in by_country.items()}
