#!/usr/bin/env python3
"""Stage 08: LLM judge -> v3.

Reads stage 07's (v2) combined probability, sends the pairs it leaves
uncertain to Qwen3-4B-Instruct-2507 (logit(Yes) - logit(No) from a single
forward pass, no generation, with 4 fixed few-shot examples from T),
re-calibrates just that scored subset against Vcal, and finishes the tier
like stages 06/07.

`--llm-finetune lora` is accepted but a no-op (off by default: bitsandbytes/
peft support on Blackwell is a risk, per Implementation_Plan.md).

Writes work/08_llm_judge/matching_results.tsv and report.json.

Memory (16GB RAM / 8GB GPU laptop): record texts are looked up only for
the judged pairs (er/pairs.py), one scope at a time. The model loads in
plain bf16 on GPUs with >= 14GiB of VRAM and NF4-quantized (~3GB VRAM)
below that (er/llm.py's `choose_precision`). Without a CUDA GPU the stage is
skipped unless `--llm-allow-cpu` is given: a 4B model in fp32 on CPU needs
~16GB of RAM on its own.
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
from er.decide import apply_calibration, calibrate_scores, finish_tier
from er.io import load_s1_splits, mark_stage_done, release_memory, stage_is_done
from er.pairs import RecordTexts, closest_to_half, label_pairs, rank_in_s1, set_prob, triple_key
from er.llm import LLMHelper, resolve_model_name
from er.metrics import build_truth_dict
from er.submit import validate, write_matching_results

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_FALLBACK = "Qwen/Qwen3-4B"
N_EXAMPLES = 4


def build_examples(norm_dir: Path, load_dir: Path, block_dir: Path) -> list[dict]:
    splits = load_s1_splits(load_dir)
    t_ids = splits.loc[splits["split"] == "T", "entity_id_num"]
    gt = pd.read_parquet(load_dir / "gt_pairs.parquet")
    pos = gt[gt["s1_id_num"].isin(t_ids)].sort_values(["s1_id_num", "match_source", "match_id_num"])
    cand = pd.read_parquet(block_dir / "train_candidates.parquet", columns=["s1_id_num", "split", "match_source", "match_id_num"])
    cand = cand[cand["split"] == "T"]
    pos_keys = np.unique(triple_key(pos["s1_id_num"], pos["match_source"], pos["match_id_num"]))
    neg = cand[~np.isin(triple_key(cand["s1_id_num"], cand["match_source"], cand["match_id_num"]), pos_keys)]

    texts = RecordTexts(norm_dir, "train")
    examples = []
    for frame, label, limit in ((pos, True, N_EXAMPLES // 2), (neg, False, N_EXAMPLES)):
        recs_a, recs_b = texts.pair_records(frame.head(N_EXAMPLES))
        for a, b in zip(recs_a, recs_b):
            if a and b and len(examples) < limit:
                examples.append({"name_a": a["name_clean"], "addr_a": a["address_clean"],
                                 "name_b": b["name_clean"], "addr_b": b["address_clean"], "label": label})
    return examples[:N_EXAMPLES]


def select_for_judge(df: pd.DataFrame, max_pairs: int) -> pd.DataFrame:
    rank = rank_in_s1(df)
    prob = df["prob"]
    band = prob.between(0.15, 0.85) | ((rank == 1) & prob.between(0.35, 0.65))
    return closest_to_half(df[band.to_numpy()], max_pairs)


def score_subset(sel: pd.DataFrame, texts: RecordTexts, llm: LLMHelper, examples: list[dict], block: int = 100_000) -> np.ndarray:
    out = np.zeros(len(sel), dtype=np.float32)
    for start in range(0, len(sel), block):
        part = sel.iloc[start:start + block]
        recs_a, recs_b = texts.pair_records(part)
        pairs = [{"name_a": (a or {}).get("name_clean", ""), "addr_a": (a or {}).get("address_clean", ""),
                  "name_b": (b or {}).get("name_clean", ""), "addr_b": (b or {}).get("address_clean", "")}
                 for a, b in zip(recs_a, recs_b)]
        out[start:start + len(part)] = llm.judge_pairs(pairs, examples)
    return out


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--llm-model", default=DEFAULT_MODEL)
    p.add_argument("--llm-fallback-model", default=DEFAULT_FALLBACK)
    # Qwen3-4B NF4 judges ~6.5 pairs/s on an RTX 4060 laptop (compute-bound;
    # batch size doesn't change it; >=14GiB GPUs load bf16 and are faster). Vcal/Vtest bands are ~12.6k pairs each,
    # so they're always fully judged; this cap only trims the test band
    # (~586k pairs), keeping the pairs closest to 0.5. 50k -> ~3.2h for the
    # stage; 200k (the 12GB-box default) -> ~10h here.
    p.add_argument("--llm-max-pairs", type=int, default=50_000)
    p.add_argument("--llm-finetune", choices=["none", "lora"], default="none")
    p.add_argument("--llm-allow-cpu", action="store_true", help="run the judge on CPU if no GPU (needs ~4x the model size in RAM)")
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    norm_dir = paths.stage_dir("02_normalize")
    load_dir = paths.stage_dir("01_load")
    block_dir = paths.stage_dir("04_blocking")
    ce_dir = paths.stage_dir("07_cross_encoder")
    stage_dir = paths.stage_dir("08_llm_judge")
    repo_root = paths.data_dir.resolve().parent

    if args.llm_finetune == "lora":
        print("WARN: --llm-finetune lora is a no-op (off by default: bitsandbytes/peft on Blackwell is a risk)")

    config = {"data_dir": str(paths.data_dir), "seed": args.seed, "llm_model": args.llm_model, "llm_max_pairs": args.llm_max_pairs, "output_version": 2}
    if stage_is_done(stage_dir, config):
        print("08_llm_judge: already done, skipping")
        return 0

    import torch

    if not torch.cuda.is_available() and not args.llm_allow_cpu:
        print("08_llm_judge: no CUDA GPU visible, skipping the LLM tier (09_finalize will use the best earlier tier)")
        (stage_dir / "report.json").unlink(missing_ok=True)  # never let 09 pick up a stale v3
        return 0

    print("loading data...")
    gt_pairs = pd.read_parquet(load_dir / "gt_pairs.parquet")
    id_cols = ["s1_id_num", "match_source", "match_id_num", "prob"]
    vcal_df = pd.read_parquet(ce_dir / "vcal_scored.parquet", columns=id_cols)
    vtest_df = pd.read_parquet(ce_dir / "vtest_scored.parquet", columns=id_cols)
    test_df = pd.read_parquet(ce_dir / "test_scored.parquet", columns=id_cols)
    vcal_df["label"] = label_pairs(vcal_df, gt_pairs)
    vtest_df["label"] = label_pairs(vtest_df, gt_pairs)

    print("building the 4 fixed few-shot examples from T...")
    examples = build_examples(norm_dir, load_dir, block_dir)

    model_name = resolve_model_name(args.llm_model, args.llm_fallback_model)
    print(f"loading LLM judge: {model_name}")
    llm = LLMHelper(model_name=model_name)

    print("selecting pairs for the judge...")
    vcal_sel = select_for_judge(vcal_df, args.llm_max_pairs)
    vtest_sel = select_for_judge(vtest_df, args.llm_max_pairs)
    test_sel = select_for_judge(test_df, args.llm_max_pairs)
    print(f"  Vcal {len(vcal_sel)}, Vtest {len(vtest_sel)}, test {len(test_sel)} of {len(test_df)}", flush=True)

    texts = RecordTexts(norm_dir, "train")
    vcal_llm = score_subset(vcal_sel, texts, llm, examples)
    vtest_llm = score_subset(vtest_sel, texts, llm, examples)
    del texts
    texts = RecordTexts(norm_dir, "test")
    test_llm = score_subset(test_sel, texts, llm, examples)
    del texts, llm

    print("stacking [prior prob, LLM logit-diff] on the Vcal subset...")
    stack_feat = np.column_stack([vcal_sel["prob"].to_numpy(), vcal_llm])
    lr = calibrate_scores(stack_feat, vcal_sel["label"].to_numpy())
    set_prob(vcal_df, vcal_sel, apply_calibration(lr, stack_feat))
    set_prob(vtest_df, vtest_sel, apply_calibration(lr, np.column_stack([vtest_sel["prob"].to_numpy(), vtest_llm])))
    set_prob(test_df, test_sel, apply_calibration(lr, np.column_stack([test_sel["prob"].to_numpy(), test_llm])))

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
        print("08_llm_judge: validation FAILED")
        return 1

    mark_stage_done(stage_dir, config)
    print("08_llm_judge: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
