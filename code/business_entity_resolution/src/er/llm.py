"""The one LLM used across the pipeline (Qwen3-4B-Instruct-2507 by default,
falling back to Qwen/Qwen3-4B): batched greedy transliteration and
abbreviation-expansion for stage 02's dictionaries, and the logit(Yes) -
logit(No) pair judge for stage 08. `--llm-model` lets a smoke test swap in a
tiny stand-in without touching this module.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Optional

# Qwen3-4B's weights are ~8GB in bf16; with activations for a batch of
# judge prompts it needs ~10-11GB. GPUs with at least this much VRAM (e.g. a
# 16GB RTX 5070 Ti) load it unquantized: several times faster than
# bitsandbytes NF4, whose on-the-fly dequantization made the judge
# compute-bound at ~6.5 pairs/s on an 8GB RTX 4060, and no dependency on
# bitsandbytes kernels for newer GPU architectures. Smaller GPUs keep NF4
# (~2.6GB of weights).
BF16_MIN_VRAM_GIB = 14


def choose_precision(device: str, total_vram_bytes: int | None, bnb_available: bool) -> str:
    """'bf16', 'nf4' or 'fp32' (CPU) for loading the LLM on this machine."""
    if device != "cuda":
        return "fp32"
    if total_vram_bytes is not None and total_vram_bytes >= BF16_MIN_VRAM_GIB * 2**30:
        return "bf16"
    if bnb_available:
        return "nf4"
    print("  WARN: bitsandbytes is not installed; loading the LLM in bf16 on a <14GiB GPU may run out of memory")
    return "bf16"


@dataclass
class LLMHelper:
    model_name: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    _tokenizer: object = field(default=None, repr=False)
    _model: object = field(default=None, repr=False)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device if torch.cuda.is_available() else "cpu"
        compute_dtype = getattr(torch, self.dtype) if device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        total_vram = torch.cuda.get_device_properties(0).total_memory if device == "cuda" else None
        precision = choose_precision(device, total_vram, importlib.util.find_spec("bitsandbytes") is not None)
        print(f"  LLM precision: {precision}"
              + (f" ({total_vram / 2**30:.1f} GiB VRAM)" if total_vram else ""), flush=True)

        if precision == "nf4":
            from transformers import BitsAndBytesConfig

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=compute_dtype,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                ),
                device_map={"": 0},
            )
        elif precision == "bf16":
            # loaded straight onto the GPU (no full CPU copy of the weights first)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=compute_dtype, device_map={"": 0},
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=compute_dtype).to(device)
        self._model.eval()
        self.device = device

    def _chat_prompt(self, user_msg: str) -> str:
        messages = [{"role": "user", "content": user_msg}]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_batch(self, prompts: list[str], max_new_tokens: int = 8) -> list[str]:
        import torch

        self._ensure_loaded()
        self._tokenizer.padding_side = "left"
        enc = self._tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1]:]
        return [self._tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]

    def transliterate_batch(self, tokens: list[str], batch_size: int = 64) -> list[str]:
        """Greedy-decode a Latin transliteration for each token (used to fill
        gaps in the positional-alignment Indic dictionary, stage 02)."""
        self._ensure_loaded()
        results: list[str] = []
        for i in range(0, len(tokens), batch_size):
            chunk = tokens[i:i + batch_size]
            prompts = [
                self._chat_prompt(
                    f"Transliterate this Indian-language business-name token to "
                    f"Latin script. Reply with only the transliteration, "
                    f"lowercase, no punctuation.\nToken: {tok}"
                )
                for tok in chunk
            ]
            out = self._generate_batch(prompts, max_new_tokens=8)
            results.extend(o.split()[0].lower() if o.split() else "" for o in out)
        return results

    def expand_abbrev_batch(self, tokens: list[str], country: str, batch_size: int = 64) -> list[Optional[str]]:
        """Greedy-decode a full-word expansion for each abbreviation
        (stage 02); the caller only accepts it if it's already in-vocabulary."""
        self._ensure_loaded()
        results: list[Optional[str]] = []
        for i in range(0, len(tokens), batch_size):
            chunk = tokens[i:i + batch_size]
            prompts = [
                self._chat_prompt(
                    f"In business addresses/names from {country}, what full word "
                    f"does the abbreviation '{tok}' usually stand for? Reply with "
                    f"only the single expanded word, lowercase."
                )
                for tok in chunk
            ]
            out = self._generate_batch(prompts, max_new_tokens=6)
            results.extend(o.split()[0].lower() if o.split() else None for o in out)
        return results

    def judge_prompt(self, pair: dict, examples: list[dict]) -> str:
        def fmt(p: dict) -> str:
            return (
                f"Record A: name=\"{p['name_a']}\" address=\"{p['addr_a']}\"\n"
                f"Record B: name=\"{p['name_b']}\" address=\"{p['addr_b']}\"\n"
                f"Same business? Answer Yes or No."
            )

        blocks = []
        for ex in examples:
            blocks.append(fmt(ex) + f"\nAnswer: {'Yes' if ex['label'] else 'No'}")
        blocks.append(fmt(pair) + "\nAnswer:")
        header = (
            "You judge whether two noisy business records refer to the same "
            "real-world business, given a name and address for each. Answer "
            "with exactly one word, Yes or No.\n\n"
        )
        return header + "\n\n".join(blocks)

    def judge_pairs(self, pairs: list[dict], examples: list[dict], batch_size: int = 32) -> list[float]:
        """logit(Yes) - logit(No) from a single forward pass per pair (no
        generation), for the pairs stage 08 scores.

        `logits_to_keep=1` asks the model to compute the LM head only for
        the last position instead of every position in the prompt -- with a
        ~150k-token vocab (Qwen3) and a many-hundred-token few-shot judge
        prompt, computing full-sequence logits per batch is what actually
        exhausts an 8-12GB GPU, not the model weights themselves.
        """
        import torch

        self._ensure_loaded()
        self._tokenizer.padding_side = "left"
        yes_id = self._tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = self._tokenizer.encode("No", add_special_tokens=False)[0]

        # Batches are formed over length-sorted prompts (less padding per
        # batch, so less wasted compute and a lower activation peak).
        prompts = [self.judge_prompt(p, examples) for p in pairs]
        #
        # `use_cache=False`: a plain forward otherwise builds a KV cache for
        # all 36 layers (~2GB at batch 32 x ~450 tokens) that is never used
        # since nothing is generated -- measured as the difference between
        # fitting and OOMing on an 8GB GPU. On a CUDA OOM anyway (unusually
        # long prompts), the batch is halved and retried.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        scores = [0.0] * len(prompts)
        i = 0
        while i < len(order):
            idx = order[i:i + batch_size]
            try:
                enc = self._tokenizer([prompts[j] for j in idx], return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    logits = self._model(**enc, logits_to_keep=1, use_cache=False).logits[:, -1, :]
            except torch.OutOfMemoryError:
                if batch_size == 1:
                    raise
                enc = logits = None
                torch.cuda.empty_cache()
                batch_size = max(1, batch_size // 2)
                print(f"  CUDA OOM in the LLM judge, retrying with batch_size={batch_size}", flush=True)
                continue
            diff = (logits[:, yes_id] - logits[:, no_id]).float().cpu().tolist()
            for j, d in zip(idx, diff):
                scores[j] = d
            i += len(idx)
        return scores


def resolve_model_name(preferred: str, fallback: str) -> str:
    """Try to load `preferred`'s config; if that fails (e.g. offline and only
    the fallback was pre-downloaded), use `fallback`."""
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(preferred)
        return preferred
    except Exception:  # noqa: BLE001
        return fallback
