"""Correctness of the memory-bounded chunked scoring path added for the real
dataset's scale: chunking the S1 side must produce exactly the same result
as scoring everything in one shot, just with a bounded peak matrix size."""
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.blocking import (
    accumulate_column_top2,
    build_channel_index,
    build_index,
    column_top2_count,
    rrf_fuse_sparse,
    score_query_chunk,
    vectorize_keys,
)


def _random_keys(n: int, vocab_size: int, max_keys: int, seed: int) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        k = rng.integers(1, max_keys + 1)
        out.append([f"tok{i}" for i in rng.choice(vocab_size, size=k, replace=False)])
    return out


def test_vectorize_keys_matches_brute_force_coo():
    keys = _random_keys(30, 25, 6, seed=1)
    vocab = {f"tok{i}": i for i in range(25)}
    idf = np.arange(25, dtype=np.float32) + 1.0

    got = vectorize_keys(keys, vocab, idf)

    rows, cols, data = [], [], []
    for r, ks in enumerate(keys):
        for k in set(ks):
            j = vocab[k]
            rows.append(r)
            cols.append(j)
            data.append(idf[j])
    want = sparse.csr_matrix((data, (rows, cols)), shape=(len(keys), len(vocab)), dtype=np.float32)
    norms = np.sqrt(want.multiply(want).sum(axis=1)).A.ravel()
    norms[norms == 0] = 1.0
    want = sparse.diags(1.0 / norms).dot(want).tocsr()

    np.testing.assert_allclose(got.toarray(), want.toarray(), atol=1e-6)


def test_build_index_absolute_cap_drops_high_df_keys():
    # "common" appears in every record; df_cap_abs=5 with n=10 should drop it.
    keys = [["common", f"rare{i}"] for i in range(10)]
    mat, vocab, idf = build_index(keys, df_cap_abs=5)
    assert "common" not in vocab
    assert all(f"rare{i}" in vocab for i in range(10))


def test_score_query_chunk_matches_unchunked_scoring():
    rng = np.random.default_rng(0)
    pool_word = _random_keys(200, 40, 8, seed=2)
    pool_ngram = _random_keys(200, 30, 10, seed=3)
    s1_word = _random_keys(57, 40, 8, seed=4)
    s1_ngram = _random_keys(57, 30, 10, seed=5)

    word_index = build_channel_index(pool_word, df_cap_abs=1_000_000)
    ngram_index = build_channel_index(pool_ngram, df_cap_abs=1_000_000)

    # Unchunked reference: one call over all 57 S1 rows at once.
    fused_ref, _ = score_query_chunk(s1_word, s1_ngram, word_index, ngram_index)

    # Chunked: split the 57 S1 rows into pieces and stack results.
    chunk_size = 20
    pieces = []
    for start in range(0, 57, chunk_size):
        fused_chunk, _ = score_query_chunk(
            s1_word[start:start + chunk_size], s1_ngram[start:start + chunk_size], word_index, ngram_index,
        )
        pieces.append(fused_chunk)
    fused_stacked = sparse.vstack(pieces).tocsr()

    np.testing.assert_allclose(fused_stacked.toarray(), fused_ref.toarray(), atol=1e-6)


def test_accumulate_column_top2_matches_single_shot():
    rng = np.random.default_rng(6)
    dense = rng.random((90, 37)).astype(np.float32)
    dense[dense < 0.6] = 0.0  # sparsify
    mat = sparse.csr_matrix(dense)

    want_best, want_second, want_count = column_top2_count(mat)

    best = np.zeros(37, dtype=np.float32)
    second = np.zeros(37, dtype=np.float32)
    count = np.zeros(37, dtype=np.int32)
    for start in range(0, 90, 13):
        accumulate_column_top2(mat[start:start + 13], best, second, count)

    np.testing.assert_allclose(best, want_best, atol=1e-6)
    np.testing.assert_allclose(second, want_second, atol=1e-6)
    np.testing.assert_array_equal(count, want_count)


def test_streaming_index_build_matches_single_chunk():
    """Building the pool index from several streamed chunks must give the
    same vocab/idf/matrix as building it from one list."""
    keys = _random_keys(120, 50, 7, seed=7)
    ref_mat, ref_vocab, ref_idf = build_index(keys, df_cap_abs=10)

    chunked = lambda: (keys[i:i + 17] for i in range(0, len(keys), 17))
    mat, vocab, idf = build_index(chunked, df_cap_abs=10)

    assert vocab == ref_vocab
    np.testing.assert_allclose(idf, ref_idf)
    np.testing.assert_allclose(mat.toarray(), ref_mat.toarray(), atol=1e-6)
