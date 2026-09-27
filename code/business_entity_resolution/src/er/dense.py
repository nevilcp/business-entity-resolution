"""Dense retrieval channel (stage 03, optional): fine-tune
multilingual-e5-small on (S1, match) pairs from T with in-batch negatives,
embed every record in fp16, and run an exact top-60 search per country with
chunked matmul -- no ANN index is needed at this pool size.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


def e5_text(name: str, address: str) -> str:
    return f"{name} {address}".strip()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class _PairDataset(Dataset):
    def __init__(self, queries: list[str], docs: list[str]):
        self.queries, self.docs = queries, docs

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        return self.queries[idx], self.docs[idx]


def fine_tune(
    model_name: str,
    pairs: list[tuple[str, str]],
    device: torch.device,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 2e-5,
    max_len: int = 64,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    ds = _PairDataset([f"query: {q}" for q, _ in pairs], [f"passage: {d}" for _, d in pairs])
    batch_size = max(2, min(batch_size, len(ds)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for _ in range(epochs):
        for queries, docs in loader:
            q_enc = tokenizer(list(queries), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            d_enc = tokenizer(list(docs), padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            q_emb = F.normalize(mean_pool(model(**q_enc).last_hidden_state, q_enc["attention_mask"]), dim=1)
            d_emb = F.normalize(mean_pool(model(**d_enc).last_hidden_state, d_enc["attention_mask"]), dim=1)
            logits = q_emb @ d_emb.T * 20.0
            labels = torch.arange(logits.shape[0], device=device)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    return tokenizer, model


@torch.no_grad()
def embed_texts(
    tokenizer, model, texts, device: torch.device, prefix: str,
    batch_size: int = 512, max_len: int = 64, out: np.ndarray | None = None,
) -> np.ndarray:
    """L2-normalized fp16 embeddings, one row per text, in input order.

    Texts are embedded in length-sorted batches (much less padding than file
    order), and written into `out` if given -- e.g. a disk-backed
    `np.lib.format.open_memmap`, so a ~6M-row pool (4.7GB in fp16) never has
    to sit in RAM next to everything else.
    """
    texts = list(texts)
    n = len(texts)
    dim = model.config.hidden_size
    if out is None:
        out = np.zeros((n, dim), dtype=np.float16)
    order = np.argsort(np.fromiter(map(len, texts), dtype=np.int64, count=n), kind="stable")
    use_amp = device.type == "cuda"
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        chunk = [f"{prefix}: {texts[j]}" for j in idx]
        enc = tokenizer(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            emb = mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
        out[idx] = F.normalize(emb.float(), dim=1).half().cpu().numpy()
    return out


@torch.no_grad()
def exact_topk_search(
    query_emb: np.ndarray, pool_emb: np.ndarray, k: int, device: torch.device,
    pool_chunk: int = 65_536, query_block: int = 8_192,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k inner-product search, chunked on both sides so peak GPU
    memory is bounded regardless of pool size: all queries (~1GB fp16 at
    1.3M rows) plus a running (n_q x k) top-k stay on the GPU, the pool is
    streamed through once in `pool_chunk` rows (it can be a disk memmap),
    and each similarity tile is at most query_block x pool_chunk
    (8192 x 65536 fp16 = 1GB). Returns (ids int64, scores float32)."""
    n_q, n_p = query_emb.shape[0], pool_emb.shape[0]
    if n_q == 0 or n_p == 0:
        return np.full((n_q, k), -1, dtype=np.int64), np.full((n_q, k), -np.inf, dtype=np.float32)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    q = torch.from_numpy(np.ascontiguousarray(query_emb)).to(device=device, dtype=dtype)
    best_vals = torch.full((n_q, k), -float("inf"), dtype=torch.float32, device=device)
    best_ids = torch.full((n_q, k), -1, dtype=torch.int64, device=device)
    for p_start in range(0, n_p, pool_chunk):
        chunk = torch.from_numpy(np.ascontiguousarray(pool_emb[p_start:p_start + pool_chunk])).to(device=device, dtype=dtype)
        chunk_k = min(k, chunk.shape[0])
        for q_start in range(0, n_q, query_block):
            q_end = min(q_start + query_block, n_q)
            sims = q[q_start:q_end] @ chunk.T
            vals, idx = torch.topk(sims, chunk_k, dim=1)
            del sims
            cat_vals = torch.cat([best_vals[q_start:q_end], vals.float()], dim=1)
            cat_ids = torch.cat([best_ids[q_start:q_end], idx + p_start], dim=1)
            top_vals, top_pos = torch.topk(cat_vals, k, dim=1)
            best_vals[q_start:q_end] = top_vals
            best_ids[q_start:q_end] = torch.gather(cat_ids, 1, top_pos)
        del chunk
    ids = best_ids.cpu().numpy()
    scores = best_vals.cpu().numpy()
    ids[~np.isfinite(scores)] = -1
    del q, best_vals, best_ids
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ids, scores
