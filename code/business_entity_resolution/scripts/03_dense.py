#!/usr/bin/env python3
"""Stage 03 (GPU, optional channel): fine-tune multilingual-e5-small on
(S1, match) pairs from T with in-batch negatives, embed every S1/S2/S3
record in fp16, and run an exact top-60 search per country -- over the full
train pool (all train S1) and the full test pool. Skipped, with a warning,
if no CUDA GPU is visible: stage 04 only folds this channel in if it
measurably improves Vcal recall.

Writes work/03_dense/{scope}_{country}_ids.npy: (n_s1_country, 60) int32
pool-row indices, in the same row order io.load_pool()/load_s1() produce,
so stage 04 can align this channel with its own.

Memory (16GB RAM / 8GB GPU laptop): one scope and one country are loaded
at a time, name/address columns only. Pool embeddings are written to a
temporary disk memmap (the US train pool is ~4.7GB in fp16) and streamed
through the GPU by `exact_topk_search`, which keeps every similarity tile
bounded (see er/dense.py). `--skip-dense` skips the channel entirely.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import add_common_args, paths_from_args
from er.dense import e5_text, embed_texts, exact_topk_search, fine_tune
from er.io import load_countries, load_pool, load_s1, load_s1_splits, mark_stage_done, pair_key, stage_is_done

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


TEXT_COLS = ["entity_id_num", "name_clean", "address_clean"]


def texts_of(df: pd.DataFrame) -> list[str]:
    return [e5_text(n, a) for n, a in zip(df["name_clean"].fillna("").tolist(), df["address_clean"].fillna("").tolist())]


def build_t_pairs(norm_dir: Path, load_dir: Path, max_pairs: int, seed: int) -> list[tuple[str, str]]:
    """(S1 text, matched pool text) for T's ground-truth pairs, joined with
    pandas instead of dicts over every train record."""
    splits = load_s1_splits(load_dir)
    t_ids = splits.loc[splits["split"] == "T", "entity_id_num"]
    gt = pd.read_parquet(load_dir / "gt_pairs.parquet")
    gt = gt[gt["s1_id_num"].isin(t_ids)]

    s1 = load_s1(norm_dir, "train", TEXT_COLS, parse_numbers=False)
    s1 = s1[s1["entity_id_num"].isin(t_ids)]
    q = gt[["s1_id_num"]].merge(
        pd.DataFrame({"s1_id_num": s1["entity_id_num"], "q": texts_of(s1)}), on="s1_id_num", how="left",
    )["q"].to_numpy()
    del s1

    pool = load_pool(norm_dir, "train", TEXT_COLS, parse_numbers=False)
    pool_key = pd.Index(pair_key(pool["source"], pool["entity_id_num"]))
    row = pool_key.get_indexer(pair_key(gt["match_source"], gt["match_id_num"]))
    hit = pool.iloc[row[row >= 0]]
    d = np.full(len(gt), None, dtype=object)
    d[row >= 0] = texts_of(hit)
    del pool, pool_key, hit

    pairs = [(qq, dd) for qq, dd in zip(q, d) if isinstance(qq, str) and qq and dd]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]
    return pairs


def main() -> int:
    p = add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--dense-model", default=DEFAULT_MODEL)
    p.add_argument("--max-pairs", type=int, default=200_000, help="cap on T pairs used for fine-tuning")
    p.add_argument("--skip-dense", action="store_true", help="skip the optional dense channel (saves ~1h of GPU time)")
    args, _ = p.parse_known_args()
    paths = paths_from_args(args)
    norm_dir = paths.stage_dir("02_normalize")
    load_dir = paths.stage_dir("01_load")
    stage_dir = paths.stage_dir("03_dense")

    config = {"data_dir": str(paths.data_dir), "seed": args.seed, "dense_model": args.dense_model, "max_pairs": args.max_pairs}
    if stage_is_done(stage_dir, config):
        print("03_dense: already done, skipping")
        return 0

    if args.skip_dense:
        print("03_dense: --skip-dense given, skipping the optional dense channel")
        for old in stage_dir.glob("*_ids.npy"):
            old.unlink()
        return 0

    import torch
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        print("03_dense: no CUDA GPU visible, skipping the optional dense channel")
        mark_stage_done(stage_dir, config)
        return 0
    device = torch.device("cuda")

    # The fine-tuned model is saved so a crashed run resumes with the same
    # model for the countries it hadn't searched yet; a different config
    # invalidates both the model and any ids it produced.
    model_dir = stage_dir / "model"
    if model_dir.exists() and stage_is_done(model_dir, config):
        print("loading the already fine-tuned dense model...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir).to(device)
        model.eval()
    else:
        for old in stage_dir.glob("*_ids.npy"):
            old.unlink()
        print("building T pairs...")
        pairs = build_t_pairs(norm_dir, load_dir, args.max_pairs, args.seed)
        print(f"fine-tuning {args.dense_model} on {len(pairs)} T pairs...", flush=True)
        tokenizer, model = fine_tune(args.dense_model, pairs, device)
        del pairs
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        mark_stage_done(model_dir, config)
    model.half()
    torch.cuda.empty_cache()

    tmp_path = stage_dir / "_pool_emb.tmp.npy"
    for scope in ("train", "test"):
        for country in load_countries(norm_dir, "s1", scope):
            out_path = stage_dir / f"{scope}_{country}_ids.npy"
            if out_path.exists():
                print(f"  {scope}/{country}: already written, skipping")
                continue
            s1_c = load_s1(norm_dir, scope, TEXT_COLS, country, parse_numbers=False)
            pool_c = load_pool(norm_dir, scope, TEXT_COLS, country, parse_numbers=False)
            if len(s1_c) == 0 or len(pool_c) == 0:
                continue
            t0 = time.time()
            q_emb = embed_texts(tokenizer, model, texts_of(s1_c), device, "query")
            d_emb = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(len(pool_c), model.config.hidden_size))
            embed_texts(tokenizer, model, texts_of(pool_c), device, "passage", out=d_emb)
            d_emb.flush()
            print(f"  {scope}/{country}: embedded {len(s1_c)} + {len(pool_c)} texts in {time.time() - t0:.0f}s", flush=True)
            n_s1, n_pool = len(s1_c), len(pool_c)
            del s1_c, pool_c

            t0 = time.time()
            d_emb = np.load(tmp_path, mmap_mode="r")
            ids, _ = exact_topk_search(q_emb, d_emb, k=60, device=device)
            del d_emb, q_emb
            tmp_path.unlink()
            # write-then-rename, so a crash never leaves a truncated ids file
            # that a resumed run would mistake for a finished country
            np.save(stage_dir / f"_{scope}_{country}_ids.tmp.npy", ids.astype(np.int32))
            (stage_dir / f"_{scope}_{country}_ids.tmp.npy").replace(out_path)
            print(f"  {scope}/{country}: {n_s1} queries x {n_pool} pool searched in {time.time() - t0:.0f}s", flush=True)

    mark_stage_done(stage_dir, config)
    print("03_dense: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
