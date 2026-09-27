#!/usr/bin/env python3
"""Stage 09: gate each tier, copy the best passing one to output/, and write
work/reports/summary.json.

Gating: v1 (stage 06) is always the baseline. v2 (07) replaces it only if
Vcal macro F0.5 improves by >= 0.001 and no country's Vcal F0.5 drops by
more than 0.002; v3 (08) is gated the same way against whichever of v1/v2
is currently ahead. This stage is cheap and idempotent, so it always
re-runs rather than using the usual _DONE resume marker.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import add_common_args, paths_from_args
from er.submit import validate

MIN_IMPROVEMENT = 0.001
MAX_COUNTRY_DROP = 0.002
TIERS = [("v1", "06_gbdt"), ("v2", "07_cross_encoder"), ("v3", "08_llm_judge")]


def passes_gate(baseline: dict, candidate: dict) -> bool:
    if candidate["vcal_f05"] - baseline["vcal_f05"] < MIN_IMPROVEMENT:
        return False
    for country, f05 in baseline["vcal_f05_by_country"].items():
        cand_f05 = candidate["vcal_f05_by_country"].get(country)
        if cand_f05 is not None and cand_f05 < f05 - MAX_COUNTRY_DROP:
            return False
    return True


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    repo_root = paths.data_dir.resolve().parent
    block_dir = paths.stage_dir("04_blocking")
    reports_dir = paths.stage_dir("reports")
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    chosen_name, chosen_dir, chosen_report = None, None, None
    gating_log = []
    tier_reports = {}
    for name, stage in TIERS:
        stage_dir = paths.work_dir / stage
        report_path = stage_dir / "report.json"
        if not report_path.exists():
            gating_log.append({"tier": name, "status": "not run"})
            continue
        report = json.loads(report_path.read_text())
        tier_reports[name] = report
        if not report.get("validated", False):
            gating_log.append({"tier": name, "status": "failed validation, skipped"})
            continue
        if chosen_report is None:
            chosen_name, chosen_dir, chosen_report = name, stage_dir, report
            gating_log.append({"tier": name, "status": "baseline", "vcal_f05": report["vcal_f05"]})
            continue
        if passes_gate(chosen_report, report):
            gating_log.append({"tier": name, "status": "accepted", "vcal_f05": report["vcal_f05"], "prev_vcal_f05": chosen_report["vcal_f05"]})
            chosen_name, chosen_dir, chosen_report = name, stage_dir, report
        else:
            gating_log.append({"tier": name, "status": "rejected", "vcal_f05": report["vcal_f05"], "prev_vcal_f05": chosen_report["vcal_f05"]})

    if chosen_dir is None:
        print("09_finalize: no tier produced a validated submission")
        return 1

    print(f"chosen tier: {chosen_name} ({chosen_dir})")
    shutil.copy(chosen_dir / "matching_results.tsv", paths.out_dir / "matching_results.tsv")
    shutil.copy(block_dir / "candidate_pairs.tsv", paths.out_dir / "candidate_pairs.tsv")

    ok, errors, warnings = validate(
        paths.out_dir / "matching_results.tsv", paths.out_dir / "candidate_pairs.tsv", paths.test_dir, repo_root, check_ids=True,
    )
    for e in errors:
        print(f"VALIDATION ERROR: {e}")
    for w in warnings:
        print(f"VALIDATION WARNING: {w}")

    block_report = json.loads((block_dir / "report.json").read_text()) if (block_dir / "report.json").exists() else {}
    summary = {
        "chosen_tier": chosen_name,
        "gating_log": gating_log,
        "blocking": block_report,
        "tiers": tier_reports,
        "final_validated": ok,
    }
    with open(reports_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if not ok:
        print("09_finalize: final validation FAILED")
        return 1
    print("09_finalize: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
