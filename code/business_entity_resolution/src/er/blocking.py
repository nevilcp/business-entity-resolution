"""Token/Q-gram blocking with IDF-weighted cosine scoring, RRF fusion across
channels, and Cardinality Node Pruning (CNP) to a fixed top-k per S1.

The scoring pipeline is: build a per-pool (S2 union S3) IDF-weighted,
L2-normalized sparse matrix per channel, score S1 queries against it with a
sparse matmul, and keep a fixed-size top-k per S1 row with a numba
insertion-heap (`_topk_per_row`). The same primitive fuses channels with
Reciprocal Rank Fusion and prunes to the final CNP set, and (in CSC
orientation) aggregates the "competing S1" stats per S2/S3 record.

The document-frequency cap that keeps a channel's sparse matmul tractable
must be an absolute postings count, not a fraction of the pool: a
fraction-of-pool cap grows with the pool, so at the sample dataset's ~10k-row
pool it quietly capped almost nothing, while at the real ~6M-row pool it let
through keys with tens of thousands of postings each, blowing the S1 x pool
matmul's nnz up into the tens-to-hundreds of billions (measured: 46e9 for the
word channel, 636e9 for the 3-gram channel on the real US train pool -- days
of compute and >100GB just for one channel's intermediate matrix). A fixed
absolute cap keeps the per-query fan-out bounded regardless of pool size
(measured on the real pool: ~1e9 total nnz at the caps below, tractable in
seconds). See `score_query_chunk`/`QUERY_CHUNK_SIZE` for the other half of
the fix: even at a bounded per-query fan-out, scoring all of a country's S1
rows against the pool in one matmul still peaks at pool-size-independent but
still large memory, so the S1 side is additionally chunked.

Memory on a 16GB machine: the pool side is never held as one Python list of
key lists (at ~6M records x ~25 keys that alone is ~10GB of str objects and
is what OOM-killed the first chunked version). `build_channel_index` makes
two streaming passes over the pool in `POOL_CHUNK_SIZE` slices -- one to
count document frequencies, one to vectorize -- and stores the pool matrix
already transposed (vocab x pool, CSR), which is the orientation the
S1 x pool matmul needs, so scipy doesn't re-transpose a copy of it per chunk.
"""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numba
import numpy as np
from scipy import sparse

WORD_DF_CAP_ABS = 1000    # word channel: drop keys with more than this many pool postings
NGRAM_DF_CAP_ABS = 2000   # 3-gram channel: drop keys with more than this many pool postings
QUERY_CHUNK_SIZE = 50_000  # S1 rows scored against the pool per matmul call
POOL_CHUNK_SIZE = 500_000  # pool rows turned into keys at a time while indexing
CHANNEL_TOP_K = 60
RRF_K = 60
CNP_K_GRID = (10, 15, 20, 30, 40, 50)
CNP_RECALL_TOL = 0.001
DENSE_MIN_GAIN = 0.002


# --------------------------------------------------------------------------
# Key extraction
# --------------------------------------------------------------------------

def char_ngrams(s: str, n: int = 3) -> list[str]:
    if len(s) < n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def addr_number_street_keys(address_clean: str, addr_numbers: list[str]) -> list[str]:
    """Combined address-number + following-street-token keys, e.g. a record
    with '221 main st' and number '221' yields the key '221_main'."""
    toks = address_clean.split()
    numset = set(addr_numbers)
    return [f"{tok}_{toks[i + 1]}" for i, tok in enumerate(toks) if tok in numset and i + 1 < len(toks)]


def word_channel_keys(name_clean: str, address_clean: str, addr_numbers: list[str]) -> list[str]:
    return name_clean.split() + address_clean.split() + addr_number_street_keys(address_clean, addr_numbers)


def ngram_channel_keys(name_nospace: str) -> list[str]:
    return char_ngrams(name_nospace, 3)


# --------------------------------------------------------------------------
# IDF index + vectorization
# --------------------------------------------------------------------------

KeyChunks = Callable[[], Iterator[list[list[str]]]]


def _as_chunk_factory(keys: "list[list[str]] | KeyChunks", chunk_size: int = POOL_CHUNK_SIZE) -> KeyChunks:
    """A plain list of key lists is accepted too (tests, small inputs)."""
    if callable(keys):
        return keys
    return lambda: (keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size))


@dataclass
class ChannelIndex:
    """One channel's pool-side index. `pool_t` is the IDF-weighted,
    L2-normalized pool matrix stored transposed: (vocab x n_pool) CSR."""
    vocab: dict[str, int]
    idf: np.ndarray
    pool_t: sparse.csr_matrix

    @property
    def n_pool(self) -> int:
        return self.pool_t.shape[1]

    def query_topk(self, keys_chunk: list[list[str]], k: int = CHANNEL_TOP_K) -> np.ndarray:
        """Top-k pool rows (int32, -1 padded) for each query key list."""
        q = vectorize_keys(keys_chunk, self.vocab, self.idf)
        sim = q @ self.pool_t
        del q
        ids, _ = topk_per_row(sim, k)
        return ids


def build_channel_index(keys: "list[list[str]] | KeyChunks", df_cap_abs: float) -> ChannelIndex:
    """Build one channel's pool index in two streaming passes over `keys`
    (a list, or a zero-arg callable yielding chunks of per-record key lists,
    re-invoked for the second pass). `df_cap_abs` is an absolute
    postings-count cap (see module docstring for why not a pool fraction)."""
    chunks = _as_chunk_factory(keys)
    df: Counter = Counter()
    n = 0
    for chunk in chunks():
        n += len(chunk)
        df.update(itertools.chain.from_iterable(map(set, chunk)))

    vocab = {k: i for i, k in enumerate(k for k, c in df.items() if c <= df_cap_abs)}
    idf = np.fromiter((np.log(n / (1 + df[k])) for k in vocab), dtype=np.float32, count=len(vocab))
    del df

    pieces = [vectorize_keys(chunk, vocab, idf) for chunk in chunks()]
    pool = sparse.vstack(pieces, format="csr") if pieces else sparse.csr_matrix((0, len(vocab)), dtype=np.float32)
    del pieces
    pool_t = pool.T.tocsr()
    return ChannelIndex(vocab=vocab, idf=idf, pool_t=pool_t)


def build_index(keys_per_record: "list[list[str]] | KeyChunks", df_cap_abs: float):
    """(pool matrix n_pool x vocab CSR, vocab, idf) -- the untransposed view
    of `build_channel_index`, kept for tests/small inputs."""
    index = build_channel_index(keys_per_record, df_cap_abs)
    return index.pool_t.T.tocsr(), index.vocab, index.idf


def vectorize_keys(keys_per_record: list[list[str]], vocab: dict[str, int], idf: np.ndarray) -> sparse.csr_matrix:
    """Vectorize records into the given fixed vocab/idf (used for the S1
    query side, which must not add new tokens or change existing weights).

    Works on one flat array of key ids for the whole input (no per-record
    numpy objects, no Python (row, col, data) triples), and L2-normalizes
    the data in place instead of via a diagonal-matrix product copy.
    """
    n = len(keys_per_record)
    sets = [set(keys) for keys in keys_per_record]
    lengths = np.fromiter(map(len, sets), dtype=np.int64, count=n)
    flat = itertools.chain.from_iterable(sets)
    ids = np.fromiter((vocab.get(k, -1) for k in flat), dtype=np.int32, count=int(lengths.sum()))
    del sets

    rows = np.repeat(np.arange(n, dtype=np.int32), lengths)
    keep = ids >= 0
    indices = ids[keep]
    rows = rows[keep]
    del ids, keep
    counts = np.bincount(rows, minlength=n).astype(np.int64)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    data = idf[indices]
    norms = np.sqrt(np.bincount(rows, weights=data.astype(np.float64) ** 2, minlength=n)).astype(np.float32)
    norms[norms == 0] = 1.0
    data /= norms[rows]
    return sparse.csr_matrix((data, indices, indptr), shape=(n, len(vocab)), dtype=np.float32)


# --------------------------------------------------------------------------
# Fixed-size top-k per row (numba)
# --------------------------------------------------------------------------

@numba.njit(cache=True, parallel=True)
def _topk_per_row(indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, k: int):
    """For each CSR row, return the top-k (col index, value) pairs sorted
    descending by value, using a fixed-size insertion heap (k is small, so
    O(nnz * k) insertion beats a general heap here). Rows with fewer than k
    nonzeros are padded with id=-1 / score=-inf.
    """
    n = indptr.shape[0] - 1
    out_ids = np.full((n, k), -1, dtype=np.int32)
    out_scores = np.full((n, k), -np.inf, dtype=np.float32)
    for r in numba.prange(n):
        start, end = indptr[r], indptr[r + 1]
        cnt = 0
        for j in range(start, end):
            idx = indices[j]
            val = data[j]
            if cnt < k:
                pos = cnt
                cnt += 1
            elif val > out_scores[r, k - 1]:
                pos = k - 1
            else:
                continue
            while pos > 0 and out_scores[r, pos - 1] < val:
                out_scores[r, pos] = out_scores[r, pos - 1]
                out_ids[r, pos] = out_ids[r, pos - 1]
                pos -= 1
            out_scores[r, pos] = val
            out_ids[r, pos] = idx
    return out_ids, out_scores


def topk_per_row(mat: sparse.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """(ids int32, scores float32), each (n_rows x k). Passes scipy's own
    index/data buffers straight through (no int64/float32 copies of what
    can be a ~100M-nnz product matrix)."""
    mat = mat.tocsr()
    return _topk_per_row(mat.indptr, mat.indices, mat.data.astype(np.float32, copy=False), k)


@numba.njit(cache=True, parallel=True)
def _column_top2_count(indptr_csc: np.ndarray, data_csc: np.ndarray, n_cols: int):
    """Per-column best score, second-best score, and nonzero count -- the
    "competing S1" aggregates stage 04 keeps per S2/S3 record."""
    best = np.zeros(n_cols, dtype=np.float32)
    second = np.zeros(n_cols, dtype=np.float32)
    count = np.zeros(n_cols, dtype=np.int32)
    for c in numba.prange(n_cols):
        start, end = indptr_csc[c], indptr_csc[c + 1]
        b = -np.inf
        s = -np.inf
        cnt = 0
        for j in range(start, end):
            v = data_csc[j]
            cnt += 1
            if v > b:
                s = b
                b = v
            elif v > s:
                s = v
        best[c] = b if cnt > 0 else 0.0
        second[c] = s if cnt > 1 else 0.0
        count[c] = cnt
    return best, second, count


def column_top2_count(mat: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csc = mat.tocsc()
    return _column_top2_count(csc.indptr, csc.data.astype(np.float32), csc.shape[1])


@numba.njit(cache=True, parallel=True)
def _column_top2_count_update(
    indptr_csc: np.ndarray, data_csc: np.ndarray, best: np.ndarray, second: np.ndarray, count: np.ndarray,
) -> None:
    """In-place variant of `_column_top2_count` that folds one more chunk's
    columns into running (best, second, count) accumulators, so a country's
    full S1 population can be scored in S1-row chunks (see
    `QUERY_CHUNK_SIZE`) while still producing exactly the same per-pool-record
    aggregates as scoring it in one shot would."""
    n_cols = best.shape[0]
    for c in numba.prange(n_cols):
        start, end = indptr_csc[c], indptr_csc[c + 1]
        b = best[c]
        s = second[c]
        cnt = count[c]
        for j in range(start, end):
            v = data_csc[j]
            cnt += 1
            if v > b:
                s = b
                b = v
            elif v > s:
                s = v
        best[c] = b
        second[c] = s
        count[c] = cnt


def accumulate_column_top2(mat: sparse.csr_matrix, best: np.ndarray, second: np.ndarray, count: np.ndarray) -> None:
    """Fold `mat`'s columns into the running accumulators in place. `best`/
    `second`/`count` must start as zeros of length `mat.shape[1]`."""
    csc = mat.tocsc()
    _column_top2_count_update(csc.indptr, csc.data.astype(np.float32, copy=False), best, second, count)


# --------------------------------------------------------------------------
# Channel scoring + RRF fusion
# --------------------------------------------------------------------------

def score_query_chunk(
    word_keys_chunk: list[list[str]],
    ngram_keys_chunk: list[list[str]],
    word_index: ChannelIndex,
    ngram_index: ChannelIndex,
    dense_ids_chunk: np.ndarray | None = None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix | None]:
    """Score one chunk of S1 rows (see `QUERY_CHUNK_SIZE`) against a
    pre-built pool index for both channels, and RRF-fuse them. Returns
    (fused_base, fused_dense_or_None), each an (n_chunk x n_pool) CSR matrix.

    Chunking the S1 side keeps every intermediate sparse matrix's size
    bounded by `QUERY_CHUNK_SIZE * postings-per-query` rather than
    `n_s1 * postings-per-query` -- for a country the size of the real US
    train pool, scoring all 1.3M S1 rows in one matmul call would otherwise
    need to materialize a single sparse matrix of hundreds of millions to
    billions of nonzeros before it's ever truncated to a top-k.
    """
    n_pool = word_index.n_pool
    ids_w = word_index.query_topk(word_keys_chunk)
    ids_g = ngram_index.query_topk(ngram_keys_chunk)

    n_chunk = len(word_keys_chunk)
    channels_base = [ids_w, ids_g]
    fused_base = rrf_fuse_sparse(channels_base, n_chunk, n_pool)
    fused_dense = None
    if dense_ids_chunk is not None:
        fused_dense = rrf_fuse_sparse(channels_base + [dense_ids_chunk], n_chunk, n_pool)
    return fused_base, fused_dense


def rrf_fuse_sparse(channel_ids_list: list[np.ndarray], n_s1: int, n_pool: int, rrf_k: int = RRF_K) -> sparse.csr_matrix:
    """Build the (n_s1 x n_pool) RRF-fused score matrix from each channel's
    top-k pool-row-index array (n_s1 x channel_k, -1 padded)."""
    rows, cols, data = [], [], []
    for ids in channel_ids_list:
        for rank in range(ids.shape[1]):
            col = ids[:, rank]
            mask = col >= 0
            r = np.nonzero(mask)[0]
            if r.size == 0:
                continue
            rows.append(r)
            cols.append(col[mask])
            data.append(np.full(r.shape[0], 1.0 / (rrf_k + rank + 1), dtype=np.float32))
    if not rows:
        return sparse.csr_matrix((n_s1, n_pool), dtype=np.float32)
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    return sparse.coo_matrix((data, (rows, cols)), shape=(n_s1, n_pool), dtype=np.float32).tocsr()


# --------------------------------------------------------------------------
# Blocking-quality metrics
# --------------------------------------------------------------------------

def pc_hit_total(candidates: dict[int, set[int]], truth: dict[int, set[int]]) -> tuple[int, int]:
    """(true pairs retained, true pairs total), the raw counts behind
    pairs-completeness -- exposed separately so callers can aggregate
    across countries by summing counts rather than averaging ratios."""
    hit = total = 0
    for s1, true_matches in truth.items():
        if not true_matches:
            continue
        cand = candidates.get(s1, set())
        hit += len(cand & true_matches)
        total += len(true_matches)
    return hit, total


def pairs_completeness(candidates: dict[int, set[int]], truth: dict[int, set[int]]) -> float:
    """Fraction of true (s1, match) pairs present in the candidate set,
    over S1 entities that have >=1 true match (singletons contribute
    nothing to either side of a pairs-recall metric)."""
    hit, total = pc_hit_total(candidates, truth)
    return hit / total if total else 1.0


def reduction_ratio(n_candidate_pairs: int, n_s1: int, n_pool: int) -> float:
    full = n_s1 * n_pool
    return 1.0 - n_candidate_pairs / full if full else 0.0


def pair_quality(candidates: dict[int, set[int]], truth: dict[int, set[int]]) -> float:
    n_true_in_cand = sum(len(candidates.get(s1, set()) & t) for s1, t in truth.items())
    n_cand = sum(len(c) for c in candidates.values())
    return n_true_in_cand / n_cand if n_cand else 0.0


def select_cnp_k(
    fused: sparse.csr_matrix,
    s1_local_idx: np.ndarray,
    truth: dict[int, set[int]],
    pool_key_of_row: np.ndarray,
    grid: Iterable[int] = CNP_K_GRID,
    tol: float = CNP_RECALL_TOL,
) -> tuple[int, dict[int, float]]:
    """Pick the smallest k in `grid` whose pairs-completeness on the given
    S1 rows is within `tol` of pairs-completeness at max(grid). `s1_local_idx`
    maps fused-matrix row -> the S1 id used as a truth/candidates key.
    """
    k_max = max(grid)
    ids_max, _ = topk_per_row(fused, k_max)
    recalls = {}
    for k in sorted(grid):
        cand = {
            int(s1_local_idx[r]): {int(pool_key_of_row[c]) for c in ids_max[r, :k] if c >= 0}
            for r in range(ids_max.shape[0])
        }
        recalls[k] = pairs_completeness(cand, truth)
    best_recall = recalls[k_max]
    chosen = next(k for k in sorted(grid) if recalls[k] >= best_recall - tol)
    return chosen, recalls
