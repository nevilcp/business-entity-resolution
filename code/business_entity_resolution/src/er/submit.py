"""Write matching_results.tsv / candidate_pairs.tsv, and validate them.

Reuses `validate()` from `utils/validate_submission.py` (found next to the
`dataset/` directory, i.e. the `student_resource/` checkout root) when it's
present, imported by file path per Implementation_Plan.md. The submission
zip doesn't ship `utils/`, so this falls back to a built-in format check.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from .io import decode_id


def write_matching_results(predictions: dict[int, set[tuple[int, int]]], out_path: Path) -> None:
    rows = [
        (decode_id(1, s1), ",".join(decode_id(s, n) for s, n in sorted(matches)))
        for s1, matches in predictions.items()
    ]
    df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    df.to_csv(out_path, sep="\t", index=False)


def _load_reference_validator(repo_root: Path):
    path = repo_root / "utils" / "validate_submission.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("validate_submission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builtin_validate(matching_path: Path, candidate_path: Path | None, test_dir: Path) -> tuple[list[str], list[str]]:
    """Minimal fallback covering the hard rejection rules only (no
    ID-existence check), used when utils/validate_submission.py isn't
    available (e.g. inside the packaged submission zip)."""
    errors: list[str] = []
    warnings: list[str] = ["utils/validate_submission.py not found; using the built-in fallback check."]
    s1_ids = set(pd.read_csv(test_dir / "test_source1.tsv", sep="\t", dtype=str)["entity_id"])

    df = pd.read_csv(matching_path, sep="\t", dtype=str, keep_default_na=False)
    if list(df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        errors.append(f"matching_results.tsv header is {list(df.columns)}")
    if df["source1_entity_id"].duplicated().any():
        errors.append("duplicate source1_entity_id rows in matching_results.tsv")
    missing = s1_ids - set(df["source1_entity_id"])
    if missing:
        errors.append(f"{len(missing)} test S1 entities missing from matching_results.tsv")
    for row in df.itertuples(index=False):
        ids = [i for i in row.matched_entity_ids.split(",") if i]
        if len(ids) != len(set(ids)):
            errors.append(f"duplicate ids in matched_entity_ids for {row.source1_entity_id}")
        if any(i.startswith("S1-") for i in ids):
            errors.append(f"{row.source1_entity_id} matches an S1 id")
    return errors, warnings


def _read_matches(matching_path: Path) -> dict[str, set[str]]:
    """{S1 id: matched ids} for the non-empty rows of matching_results.tsv."""
    out: dict[str, set[str]] = {}
    with open(matching_path, encoding="utf-8") as f:
        next(f, None)
        for line in f:
            s1, tab, rest = line.partition("\t")
            rest = rest.strip()
            if tab and rest:
                out[s1] = set(rest.split(","))
    return out


def _ids_in_first_column(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        next(f, None)
        return {line.split("\t", 1)[0].strip() for line in f if line.strip()}


def check_candidates(
    candidate_path: Path, test_dir: Path, matching_path: Path | None = None, check_ids: bool = False,
) -> tuple[list[str], list[str]]:
    """The reference validator's candidate_pairs.tsv rules, streamed.

    utils/validate_submission.py keeps a Python set of candidate id strings
    for every S1 row; on the full test set (1.7M rows x ~50 candidates) that
    alone peaks at ~10GB, more than a stage can spend on a 16GB laptop. This
    applies the same rules -- header, one row per required S1, no repeated
    ids within a row, S2-/S3- prefixes only, optional id existence, and the
    "matches should be a subset of candidates" warning -- one line at a
    time, holding only the (small) matched sets.
    """
    errors: list[str] = []
    warnings: list[str] = []
    name = candidate_path.name
    if not candidate_path.is_file():
        return [f"File not found: {candidate_path}"], warnings
    required = _ids_in_first_column(test_dir / "test_source1.tsv")
    valid_ids = None
    if check_ids:
        valid_ids = set()
        for src in ("test_source2.tsv", "test_source3.tsv"):
            valid_ids |= _ids_in_first_column(test_dir / src)
    matched = _read_matches(matching_path) if matching_path is not None else None

    seen: set[str] = set()
    dup_rows, intra_dupes, bad_prefix, unknown, not_subset = set(), set(), set(), set(), set()
    with open(candidate_path, encoding="utf-8") as f:
        header = [c.strip().lower() for c in f.readline().rstrip("\n").split("\t")]
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            return [f"{name}: unexpected header {header}"], warnings
        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition("\t")
            if not tab:
                if s1.strip():
                    errors.append(f"{name}: malformed row (no tab) at line {line_num}")
                continue
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)
            ids = rest.rstrip("\n").split(",") if rest.strip() else []
            id_set = set(ids)
            if len(ids) != len(id_set):
                intra_dupes.add(s1)
            for mid in id_set:
                if not mid.startswith(("S2-", "S3-")):
                    bad_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)
            if matched is not None and s1 in matched and matched[s1] - id_set:
                not_subset.add(s1)
    if matched is not None:
        not_subset |= set(matched) - seen

    for offenders, msg in (
        (dup_rows, "duplicate source1_entity_id row(s)"),
        (intra_dupes, "repeated ID inside a candidate list"),
        (bad_prefix, "candidate ids without an S2-/S3- prefix"),
        (unknown, "candidate ids not in the test Source-2/3 files"),
        (required - seen, "required S1 entity(ies) missing"),
        (seen - required, "row(s) using an S1 id that is not in the test set"),
    ):
        if offenders:
            errors.append(f"{name}: {msg}: {len(offenders)} total, e.g. {sorted(offenders)[:5]}")
    if not_subset:
        warnings.append(f"{len(not_subset)} S1 entity(ies) have matched ids not present in {name}, "
                        f"e.g. {sorted(not_subset)[:5]}")
    print(f"  {name}: {len(seen)} rows checked (streaming)")
    return errors, warnings


def validate(
    matching_path: Path, candidate_path: Path | None, test_dir: Path, repo_root: Path, check_ids: bool = False,
) -> tuple[bool, list[str], list[str]]:
    """Returns (passed, errors, warnings).

    matching_results.tsv (the scored file) goes through the reference
    validator as-is; candidate_pairs.tsv through the streaming
    `check_candidates` (same rules, a fraction of the memory).
    """
    module = _load_reference_validator(repo_root)
    if module is not None:
        errors, warnings = module.validate(str(matching_path), None, str(test_dir), check_ids)
    else:
        errors, warnings = _builtin_validate(matching_path, None, test_dir)
    if candidate_path is not None:
        cand_errors, cand_warnings = check_candidates(Path(candidate_path), Path(test_dir), Path(matching_path), check_ids)
        errors += cand_errors
        warnings += cand_warnings
    return len(errors) == 0, errors, warnings
