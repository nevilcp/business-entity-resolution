#!/usr/bin/env python3
"""Pre-fetch every Hugging Face model the pipeline needs, so the remote run
doesn't hit the network mid-stage. All three are MIT/Apache-2.0 and self-
hosted only (no API calls at inference time) per the challenge's model rule.

Combined size budget: e5-small (0.12B) + xlm-roberta-base (0.28B) +
Qwen3-4B-Instruct-2507 (4.0B) ~= 4.4B params, under the 8B cap.

``--models`` lets a laptop smoke test swap in tiny stand-ins (e.g.
``Qwen/Qwen3-0.6B`` for the LLM judge) without touching the stage code,
which only ever reads the model name from ``--llm-model`` / config.
"""
from __future__ import annotations

import argparse
import os
import sys

# The Xet accelerated-transfer backend (hf_xet) can fail to refresh its CAS
# token in restricted network environments (HfHubHTTPError 404 on
# .../xet-read-token/...). Force plain HTTP downloads, which always work.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Every model here is loaded via transformers.AutoModel*.from_pretrained,
# which only needs the safetensors weights (never .bin/.h5/.msgpack/onnx/
# openvino duplicates that some repos also publish).
WEIGHT_IGNORE_PATTERNS = [
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.ot",
    "*.onnx",
    "onnx/*",
    "openvino/*",
]

DEFAULT_MODELS = {
    "dense": "intfloat/multilingual-e5-small",
    "cross_encoder": "xlm-roberta-base",
    "llm": "Qwen/Qwen3-4B-Instruct-2507",
}
LLM_FALLBACK = "Qwen/Qwen3-4B"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense-model", default=DEFAULT_MODELS["dense"])
    p.add_argument("--cross-encoder-model", default=DEFAULT_MODELS["cross_encoder"])
    p.add_argument("--llm-model", default=DEFAULT_MODELS["llm"])
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    models = [args.dense_model, args.cross_encoder_model]
    llm_candidates = [args.llm_model]
    if args.llm_model == DEFAULT_MODELS["llm"]:
        llm_candidates.append(LLM_FALLBACK)

    for name in models:
        print(f"downloading {name} ...")
        snapshot_download(
            repo_id=name, cache_dir=args.cache_dir, ignore_patterns=WEIGHT_IGNORE_PATTERNS
        )
        print(f"downloading {name}: done")

    for name in llm_candidates:
        try:
            print(f"downloading {name} ...")
            snapshot_download(
                repo_id=name, cache_dir=args.cache_dir, ignore_patterns=WEIGHT_IGNORE_PATTERNS
            )
            print(f"downloading {name}: done")
            break
        except Exception as e:  # noqa: BLE001
            print(f"downloading {name}: failed ({e})", file=sys.stderr)
    else:
        print("could not download any LLM judge model", file=sys.stderr)
        return 1

    print("download_models: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
