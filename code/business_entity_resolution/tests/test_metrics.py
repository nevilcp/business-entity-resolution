import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.metrics import f05, macro_f05


def test_readme_example():
    predicted = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert f05(predicted, truth) == pytest.approx(0.714, abs=1e-3)


def test_singleton_true_negative_scores_one():
    assert f05(set(), set()) == 1.0


def test_singleton_false_positive_scores_zero():
    assert f05({"S2-1"}, set()) == 0.0


def test_missed_match_scores_zero():
    assert f05(set(), {"S2-1"}) == 0.0


def test_macro_f05_averages_over_every_truth_key():
    predictions = {"S1-1": {"S2-1"}, "S1-2": set()}
    truths = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S2-9"}}
    # S1-1 -> 1.0 (correct match), S1-2 -> 1.0 (correct singleton),
    # S1-3 -> 0.0 (missing prediction, no candidate given at all)
    assert macro_f05(predictions, truths) == pytest.approx(2 / 3, abs=1e-9)
