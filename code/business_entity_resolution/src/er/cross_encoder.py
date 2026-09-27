"""xlm-roberta-base cross-encoder for the candidate pairs the GBDT is unsure
about (stage 07). Pair text is "name | address" for each side, max_len 128,
bf16, lr 2e-5.

Training (07a, `fine_tune`) only needs stage 04's candidates and the
ground truth (positives plus a sample of blocking-retrieved negatives from
T), so it can run in the GPU lane in parallel with stages 05/06. Scoring
(07b, `score_pairs`) is applied afterwards to the pairs stage 06's
calibrated probability leaves uncertain.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LEN = 128


def pair_text(rec: dict) -> str:
    return f"{rec['name_clean']} | {rec['address_clean']}"


class _PairDataset(Dataset):
    def __init__(self, texts_a: list[str], texts_b: list[str], labels: list[int] | None = None):
        self.texts_a, self.texts_b, self.labels = texts_a, texts_b, labels

    def __len__(self) -> int:
        return len(self.texts_a)

    def __getitem__(self, idx: int):
        if self.labels is None:
            return self.texts_a[idx], self.texts_b[idx]
        return self.texts_a[idx], self.texts_b[idx], self.labels[idx]


def fine_tune(
    model_name: str,
    texts_a: list[str],
    texts_b: list[str],
    labels: list[int],
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    # xlm-roberta-base's 250k-token word-embedding matrix is ~190M of its
    # ~280M parameters; its gradient + AdamW state alone would be ~2.3GB.
    # Freezing it keeps fine-tuning comfortably inside an 8GB GPU.
    model.get_input_embeddings().weight.requires_grad_(False)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    ds = _PairDataset(texts_a, texts_b, labels)
    batch_size = max(2, min(batch_size, len(ds)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for _ in range(epochs):
        for a, b, y in loader:
            enc = tokenizer(list(a), list(b), padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
            y_t = torch.as_tensor(y, device=device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
                out = model(**enc, labels=y_t)
            opt.zero_grad()
            out.loss.backward()
            opt.step()

    model.eval()
    return tokenizer, model


@torch.no_grad()
def score_pairs(tokenizer, model, texts_a: list[str], texts_b: list[str], device: torch.device, batch_size: int = 64) -> np.ndarray:
    """P(match) per pair, in input order. Batches are formed over
    length-sorted pairs so each batch pads to similar lengths."""
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    n = len(texts_a)
    order = np.argsort([len(a) + len(b) for a, b in zip(texts_a, texts_b)], kind="stable")
    scores = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        a = [texts_a[j] for j in idx]
        b = [texts_b[j] for j in idx]
        enc = tokenizer(a, b, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(**enc).logits.float()
        scores[idx] = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    return scores
