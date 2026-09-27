#!/usr/bin/env python3
"""Stage 02: build per-pool dictionaries (legal forms, Indic->Latin,
abbreviations) and apply normalize.py to every S1/S2/S3 record.

Reads work/01_load/*.parquet, writes work/02_normalize/{s1,s2,s3}_{train,test}.parquet
(entity_id_num, country + the normalize.py output fields) and dictionaries.json
(the built dictionaries, for inspection/reuse).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import add_common_args, paths_from_args
from er.dictionaries import (
    build_abbrev_map,
    build_legal_form_vocab,
    extend_dict_with_llm,
    positional_align_indic_dict,
)
from er.io import mark_stage_done, stage_is_done
from er.llm import LLMHelper, resolve_model_name
from er.normalize import (
    DEFAULT_LEGAL_FORMS,
    is_latin_token,
    nfkc_casefold,
    normalize_address,
    normalize_name,
)

DEFAULT_LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_LLM_FALLBACK = "Qwen/Qwen3-4B"
NORMALIZE_CHUNK = 500_000


def names_by_country(dfs: list[pd.DataFrame]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for df in dfs:
        for country, group in df.groupby("country")["business_name"]:
            out.setdefault(country, []).extend(group.tolist())
    return out


def build_india_pairs(
    s1: pd.DataFrame, gt_pairs: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
) -> list[tuple[list[str], list[str]]]:
    """(S2/S3 name tokens, S1 name tokens) for training India ground-truth
    pairs -- the S1 side is Latin, the match side is sometimes Devanagari
    (see Implementation_Plan.md's "7% of India pairs" data fact).
    """
    s1_india = s1[s1["country"] == "India"].set_index("entity_id_num")["business_name"]
    gt = gt_pairs[gt_pairs["s1_id_num"].isin(s1_india.index)]
    s2_names = s2.set_index("entity_id_num")["business_name"]
    s3_names = s3.set_index("entity_id_num")["business_name"]

    pairs = []
    for row in gt.itertuples(index=False):
        match_names = s2_names if row.match_source == 2 else s3_names
        match_name = match_names.get(row.match_id_num)
        if match_name is None:
            continue
        s1_name = s1_india.loc[row.s1_id_num]
        pairs.append((nfkc_casefold(match_name).split(), nfkc_casefold(s1_name).split()))
    return pairs


def collect_non_latin_tokens(names: list[str], existing_dict: dict[str, str]) -> set[str]:
    tokens = set()
    for name in names:
        for tok in nfkc_casefold(name).split():
            if not is_latin_token(tok) and tok not in existing_dict:
                tokens.add(tok)
    return tokens


def apply_normalize(
    df: pd.DataFrame,
    legal_forms_by_country: dict[str, set[str]],
    translit_dict: dict[str, str],
    abbrev_map: dict[tuple[str, str], str],
) -> pd.DataFrame:
    rows = []
    for country, name, addr in zip(df["country"], df["business_name"], df["business_address"]):
        legal_forms = legal_forms_by_country.get(country, DEFAULT_LEGAL_FORMS)
        n = normalize_name(name, country, translit_dict, legal_forms, abbrev_map)
        a = normalize_address(addr, country, translit_dict, abbrev_map)
        rows.append({
            "name_clean": n["name_clean"],
            "legal_form": n["legal_form"],
            "name_transliterated": n["transliterated"],
            "name_nospace": n["name_nospace"],
            "name_skeleton": n["name_skeleton"],
            "address_clean": a["address_clean"],
            "postcode": a["postcode"],
            "addr_numbers": ",".join(a["addr_numbers"]),
            "addr_transliterated": a["transliterated"],
        })
    result = pd.DataFrame(rows, index=df.index)
    return pd.concat([df[["entity_id_num", "country"]], result], axis=1)


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--use-llm-dict", action="store_true", default=True)
    p.add_argument("--no-llm-dict", dest="use_llm_dict", action="store_false")
    p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--llm-fallback-model", default=DEFAULT_LLM_FALLBACK)
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    load_dir = paths.stage_dir("01_load")
    stage_dir = paths.stage_dir("02_normalize")

    config = {
        "data_dir": str(paths.data_dir), "seed": args.seed,
        "use_llm_dict": args.use_llm_dict, "llm_model": args.llm_model,
    }
    if stage_is_done(stage_dir, config):
        print("02_normalize: already done, skipping")
        return 0

    print("loading stage 01 output...")
    s1_train = pd.read_parquet(load_dir / "s1_train.parquet")
    s2_train = pd.read_parquet(load_dir / "s2_train.parquet")
    s3_train = pd.read_parquet(load_dir / "s3_train.parquet")
    gt_pairs = pd.read_parquet(load_dir / "gt_pairs.parquet")
    s1_test = pd.read_parquet(load_dir / "s1_test.parquet")
    s2_test = pd.read_parquet(load_dir / "s2_test.parquet")
    s3_test = pd.read_parquet(load_dir / "s3_test.parquet")

    train_pool_names = names_by_country([s1_train, s2_train, s3_train])
    test_pool_names = names_by_country([s1_test, s2_test, s3_test])

    print("building legal-form vocabularies...")
    legal_forms_train = build_legal_form_vocab(train_pool_names)
    legal_forms_test = build_legal_form_vocab(test_pool_names)

    print("positional-aligning the Indic->Latin dictionary...")
    india_pairs = build_india_pairs(s1_train, gt_pairs, s2_train, s3_train)
    indic_dict = positional_align_indic_dict(india_pairs, min_support=2)
    print(f"  {len(indic_dict)} tokens from positional alignment")

    llm: LLMHelper | None = None
    if args.use_llm_dict:
        model_name = resolve_model_name(args.llm_model, args.llm_fallback_model)
        print(f"loading LLM for dictionary building: {model_name}")
        llm = LLMHelper(model_name=model_name)
        missing = collect_non_latin_tokens(
            train_pool_names.get("India", []) + test_pool_names.get("India", []), indic_dict,
        )
        print(f"  transliterating {len(missing)} tokens the alignment step missed...")
        indic_dict = extend_dict_with_llm(indic_dict, missing, llm.transliterate_batch)
        print(f"  Indic dictionary now has {len(indic_dict)} tokens")

    print("building abbreviation maps...")
    abbrev_train: dict[tuple[str, str], str] = {}
    abbrev_test: dict[tuple[str, str], str] = {}
    if llm is not None:
        abbrev_train = build_abbrev_map(train_pool_names, llm.expand_abbrev_batch)
        abbrev_test = build_abbrev_map(test_pool_names, llm.expand_abbrev_batch)
    print(f"  {len(abbrev_train)} train abbreviations, {len(abbrev_test)} test abbreviations")

    jobs = [
        ("s1_train", s1_train, legal_forms_train, abbrev_train),
        ("s2_train", s2_train, legal_forms_train, abbrev_train),
        ("s3_train", s3_train, legal_forms_train, abbrev_train),
        ("s1_test", s1_test, legal_forms_test, abbrev_test),
        ("s2_test", s2_test, legal_forms_test, abbrev_test),
        ("s3_test", s3_test, legal_forms_test, abbrev_test),
    ]
    # Free what only the dictionary-building step needed (two Python lists of
    # every business name, and the LLM) before the long normalization pass.
    del train_pool_names, test_pool_names, india_pairs
    if llm is not None:
        del llm
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name, df, legal_forms, abbrev in jobs:
        print(f"normalizing {name} ({len(df)} rows)...")
        # chunked, so at most NORMALIZE_CHUNK rows of Python strings exist at once
        tmp_path = stage_dir / f"{name}.parquet.tmp"
        writer = None
        for start in range(0, len(df), NORMALIZE_CHUNK):
            normed = apply_normalize(df.iloc[start:start + NORMALIZE_CHUNK], legal_forms, indic_dict, abbrev)
            table = pa.Table.from_pandas(normed, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema)
            writer.write_table(table.cast(writer.schema))
        if writer is not None:
            writer.close()
            tmp_path.replace(stage_dir / f"{name}.parquet")

    with open(stage_dir / "dictionaries.json", "w") as f:
        json.dump({
            "indic_dict": indic_dict,
            "abbrev_train": {f"{c}|{t}": v for (c, t), v in abbrev_train.items()},
            "abbrev_test": {f"{c}|{t}": v for (c, t), v in abbrev_test.items()},
            "legal_forms_train": {c: sorted(v) for c, v in legal_forms_train.items()},
            "legal_forms_test": {c: sorted(v) for c, v in legal_forms_test.items()},
        }, f, indent=2, ensure_ascii=False)

    mark_stage_done(stage_dir, config)
    print("02_normalize: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
