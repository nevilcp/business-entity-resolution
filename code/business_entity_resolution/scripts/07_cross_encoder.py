#!/usr/bin/env python3
"""Stage 07: cross-encoder refinement -> v2.

--mode fit   trains xlm-roberta-base on T (stage 04's candidates + ground
             truth only, so it can run in the GPU lane parallel to 05/06 --
             see Implementation_Plan.md's stage diagram) and saves it.
--mode score reads stage 06's calibrated probabilities, re-scores the
             top-10-by-rank pairs it hasn't ruled out with the fine-tuned
             model, fits a context stacker on Vcal (the prior probability,
             whether a pair was re-scored, its cross-encoder score, and how
             it ranks among its S1's other candidates) and finishes the
             tier (tau tuning, decide, submission, report) exactly like
             stage 06.
--mode all   runs fit then score (the default, for standalone use).

Writes work/07_cross_encoder/model/, matching_results.tsv, report.json, and
{vcal,vtest,test}_scored.parquet (the v2 probability, for stage 08).

Memory (16GB RAM / 8GB GPU laptop): record texts are looked up only for
the pairs actually trained on or scored (er/pairs.py), one scope at a time,
and selected pairs are scored in `--score-block` blocks, so no stage-wide
dict of every record exists. Stage 06's scored frames are updated in place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import add_common_args, paths_from_args
from er.cross_encoder import fine_tune, pair_text, score_pairs
from er.decide import apply_context_stacker, fit_context_stacker, finish_tier
from er.io import load_s1_splits, mark_stage_done, release_memory, stage_is_done
from er.pairs import RecordTexts, closest_to_half, label_pairs, rank_in_s1, triple_key
from er.metrics import build_truth_dict
from er.submit import validate, write_matching_results

DEFAULT_MODEL = "xlm-roberta-base"
OUTPUT_VERSION = 3  # bumped: wider uncertain-band + context stacker + hard-negative sampling


def build_training_pairs(norm_dir: Path, load_dir: Path, block_dir: Path, max_pairs: int, seed: int):
    splits = load_s1_splits(load_dir)
    t_ids = splits.loc[splits["split"] == "T", "entity_id_num"]
    gt = pd.read_parquet(load_dir / "gt_pairs.parquet")
    pos = gt[gt["s1_id_num"].isin(t_ids)].drop_duplicates()
    cand = pd.read_parquet(block_dir / "train_candidates.parquet", columns=["s1_id_num", "split", "match_source", "match_id_num", "rank"])
    cand = cand[cand["split"] == "T"].drop(columns=["split"])
    pos_keys = np.unique(triple_key(pos["s1_id_num"], pos["match_source"], pos["match_id_num"]))
    is_neg = ~np.isin(triple_key(cand["s1_id_num"], cand["match_source"], cand["match_id_num"]), pos_keys)
    # Hard negatives: candidates the cross-encoder will actually be asked to
    # discriminate at inference (07's uncertain band is rank<=10), not a
    # uniform sample of all 50 blocking candidates, most of which are easy.
    neg = cand[is_neg & (cand["rank"] <= 10)].drop(columns=["rank"])
    del cand

    texts = RecordTexts(norm_dir, "train")
    pos_a, pos_b = texts.pair_records(pos)
    keep = [i for i, (x, y) in enumerate(zip(pos_a, pos_b)) if x and y]
    texts_a = [pair_text(pos_a[i]) for i in keep]
    texts_b = [pair_text(pos_b[i]) for i in keep]
    labels = [1] * len(keep)

    rng = np.random.default_rng(seed)
    n_neg = min(len(neg), max(len(texts_a), max_pairs - len(texts_a)))
    neg = neg.iloc[np.sort(rng.choice(len(neg), n_neg, replace=False))] if len(neg) else neg
    neg_a, neg_b = texts.pair_records(neg)
    del texts
    for x, y in zip(neg_a, neg_b):
        if x and y:
            texts_a.append(pair_text(x))
            texts_b.append(pair_text(y))
            labels.append(0)

    if len(texts_a) > max_pairs:
        idx = rng.choice(len(texts_a), max_pairs, replace=False)
        texts_a = [texts_a[i] for i in idx]
        texts_b = [texts_b[i] for i in idx]
        labels = [labels[i] for i in idx]
    return texts_a, texts_b, labels


def select_uncertain(df: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    """Top-10-by-rank candidates the GBDT hasn't already ruled out (p<=0.97).
    No lower bound: many true pairs get buried by the GBDT below p=0.03 at
    rank 2-10 (measured on Vtest) and are worth the cross-encoder's look."""
    rank = rank_in_s1(df)
    band = (rank <= 10) & (df["prob"].to_numpy() <= 0.97)
    return closest_to_half(df[band], max_pairs)


def scored_arrays(df: pd.DataFrame, sel: pd.DataFrame, new_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(scored bool, new_prob float32) aligned to all of `df`'s rows, NaN/False
    where `sel` (a `select_uncertain` subset, same index as `df`) has no score."""
    scored = np.zeros(len(df), dtype=bool)
    new_prob = np.full(len(df), np.nan, dtype=np.float32)
    scored[sel.index] = True
    new_prob[sel.index] = new_scores
    return scored, new_prob


def score_subset(sel: pd.DataFrame, texts: RecordTexts, tokenizer, model, device, block: int) -> np.ndarray:
    out = np.zeros(len(sel), dtype=np.float32)
    for start in range(0, len(sel), block):
        part = sel.iloc[start:start + block]
        recs_a, recs_b = texts.pair_records(part)
        out[start:start + len(part)] = score_pairs(
            tokenizer, model, [pair_text(r) if r else "" for r in recs_a], [pair_text(r) if r else "" for r in recs_b], device,
        )
        del part, recs_a, recs_b
        release_memory()  # the widened uncertain band can put ~60 blocks through this loop
    return out


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--mode", choices=["fit", "score", "all"], default="all")
    p.add_argument("--ce-model", default=DEFAULT_MODEL)
    p.add_argument("--max-train-pairs", type=int, default=1_500_000)
    # The wider top-10/p<=0.97 band needs ~7.3 pairs/S1 (measured on Vcal/Vtest);
    # at test scale (1.73M S1) that's ~13M pairs, so the cap is raised to cover
    # it without silently truncating to the pairs closest to 0.5.
    p.add_argument("--ce-max-pairs", type=int, default=15_000_000)
    p.add_argument("--score-block", type=int, default=200_000, help="pairs whose texts are gathered and scored at a time")
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    norm_dir = paths.stage_dir("02_normalize")
    load_dir = paths.stage_dir("01_load")
    block_dir = paths.stage_dir("04_blocking")
    gbdt_dir = paths.stage_dir("06_gbdt")
    stage_dir = paths.stage_dir("07_cross_encoder")
    repo_root = paths.data_dir.resolve().parent
    model_dir = stage_dir / "model"

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode in ("fit", "all"):
        fit_config = {"data_dir": str(paths.data_dir), "seed": args.seed, "ce_model": args.ce_model, "max_train_pairs": args.max_train_pairs, "output_version": OUTPUT_VERSION}
        if not stage_is_done(stage_dir / "_fit", fit_config):
            print("building cross-encoder training pairs...", flush=True)
            texts_a, texts_b, labels = build_training_pairs(norm_dir, load_dir, block_dir, args.max_train_pairs, args.seed)
            print(f"fine-tuning {args.ce_model} on {len(texts_a)} pairs ({sum(labels)} positive)...")
            tokenizer, model = fine_tune(args.ce_model, texts_a, texts_b, labels, device)
            del texts_a, texts_b, labels
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            del model, tokenizer
            if device.type == "cuda":
                torch.cuda.empty_cache()
            (stage_dir / "_fit").mkdir(parents=True, exist_ok=True)
            mark_stage_done(stage_dir / "_fit", fit_config)
        else:
            print("07_cross_encoder fit: already done, skipping")

    if args.mode not in ("score", "all"):
        return 0

    config = {"data_dir": str(paths.data_dir), "seed": args.seed, "ce_model": args.ce_model, "ce_max_pairs": args.ce_max_pairs, "output_version": OUTPUT_VERSION}
    if stage_is_done(stage_dir, config):
        print("07_cross_encoder: already done, skipping")
        return 0

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("loading fine-tuned cross-encoder...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    gt_pairs = pd.read_parquet(load_dir / "gt_pairs.parquet")
    id_cols = ["s1_id_num", "match_source", "match_id_num", "prob"]
    vcal_df = pd.read_parquet(gbdt_dir / "vcal_scored.parquet", columns=id_cols)
    vtest_df = pd.read_parquet(gbdt_dir / "vtest_scored.parquet", columns=id_cols)
    test_df = pd.read_parquet(gbdt_dir / "test_scored.parquet", columns=id_cols)
    vcal_df["label"] = label_pairs(vcal_df, gt_pairs)
    vtest_df["label"] = label_pairs(vtest_df, gt_pairs)

    print("selecting uncertain pairs...")
    vcal_sel = select_uncertain(vcal_df, args.ce_max_pairs)
    vtest_sel = select_uncertain(vtest_df, args.ce_max_pairs)
    test_sel = select_uncertain(test_df, args.ce_max_pairs)
    print(f"  Vcal {len(vcal_sel)}, Vtest {len(vtest_sel)}, test {len(test_sel)} of {len(test_df)}", flush=True)

    texts = RecordTexts(norm_dir, "train")
    vcal_ce = score_subset(vcal_sel, texts, tokenizer, model, device, args.score_block)
    vtest_ce = score_subset(vtest_sel, texts, tokenizer, model, device, args.score_block)
    del texts
    texts = RecordTexts(norm_dir, "test")
    test_ce = score_subset(test_sel, texts, tokenizer, model, device, args.score_block)
    del texts, model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("fitting the context stacker on Vcal...")
    vcal_scored, vcal_new = scored_arrays(vcal_df, vcal_sel, vcal_ce)
    vtest_scored, vtest_new = scored_arrays(vtest_df, vtest_sel, vtest_ce)
    test_scored, test_new = scored_arrays(test_df, test_sel, test_ce)
    stacker = fit_context_stacker(vcal_df, vcal_scored, vcal_new)
    vcal_df["prob"] = apply_context_stacker(stacker, vcal_df, vcal_scored, vcal_new)
    vtest_df["prob"] = apply_context_stacker(stacker, vtest_df, vtest_scored, vtest_new)
    test_df["prob"] = apply_context_stacker(stacker, test_df, test_scored, test_new)

    s1_train = pd.read_parquet(load_dir / "s1_train.parquet", columns=["entity_id_num", "country", "split"])
    s1_train = s1_train[s1_train["split"].isin(["Vcal", "Vtest"])]
    s1_test_ids = pd.read_parquet(load_dir / "s1_test.parquet", columns=["entity_id_num"])["entity_id_num"].to_numpy()
    vcal_ids = s1_train.loc[s1_train["split"] == "Vcal", "entity_id_num"]
    vtest_ids = s1_train.loc[s1_train["split"] == "Vtest", "entity_id_num"]
    vcal_truth = build_truth_dict(gt_pairs, vcal_ids)
    vtest_truth = build_truth_dict(gt_pairs, vtest_ids)
    country_of = s1_train.set_index("entity_id_num")["country"].to_dict()

    result = finish_tier(vcal_df, vtest_df, test_df, s1_test_ids, country_of, vcal_truth, vtest_truth)
    print(f"  tau = {result['tau']}, Vcal macro F0.5 = {result['vcal_f05']:.4f}")
    print(f"  Vtest macro F0.5 = {result['vtest_f05']:.4f} {result['vtest_f05_by_country']}")

    result["vcal_df"][id_cols].to_parquet(stage_dir / "vcal_scored.parquet", index=False)
    result["vtest_df"][id_cols].to_parquet(stage_dir / "vtest_scored.parquet", index=False)
    result["test_df"][id_cols].to_parquet(stage_dir / "test_scored.parquet", index=False)

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
        **summary, "validated": ok,
        "n_scored": {"vcal": len(vcal_sel), "vtest": len(vtest_sel), "test": len(test_sel)},
    }
    with open(stage_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    if not ok:
        print("07_cross_encoder: validation FAILED")
        return 1

    mark_stage_done(stage_dir, config)
    print("07_cross_encoder: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
