import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.blocking import column_top2_count, topk_per_row


def brute_force_topk(mat: sparse.csr_matrix, k: int):
    dense = mat.toarray()
    n = dense.shape[0]
    ids = np.full((n, k), -1, dtype=np.int64)
    scores = np.full((n, k), -np.inf, dtype=np.float32)
    for r in range(n):
        row = dense[r]
        nz = np.nonzero(row)[0]
        order = nz[np.argsort(-row[nz])][:k]
        ids[r, : len(order)] = order
        scores[r, : len(order)] = row[order]
    return ids, scores


def random_sparse(n_rows: int, n_cols: int, density: float, seed: int) -> sparse.csr_matrix:
    rng = np.random.default_rng(seed)
    mask = rng.random((n_rows, n_cols)) < density
    data = rng.random((n_rows, n_cols)).astype(np.float32)
    return sparse.csr_matrix(data * mask)


def test_topk_per_row_matches_brute_force():
    mat = random_sparse(50, 200, density=0.1, seed=0)
    k = 5
    ids, scores = topk_per_row(mat, k)
    brute_ids, brute_scores = brute_force_topk(mat, k)
    np.testing.assert_allclose(scores, brute_scores, atol=1e-6)
    # ids may differ only where scores tie; check the (id, score) sets match per row instead
    for r in range(mat.shape[0]):
        got = {(int(i), round(float(s), 5)) for i, s in zip(ids[r], scores[r]) if i >= 0}
        want = {(int(i), round(float(s), 5)) for i, s in zip(brute_ids[r], brute_scores[r]) if i >= 0}
        assert got == want


def test_topk_per_row_handles_fewer_nonzeros_than_k():
    mat = sparse.csr_matrix(np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    ids, scores = topk_per_row(mat, k=5)
    assert list(ids[0][:2]) == [1, -1] or (ids[0][0] == 1 and scores[0][0] == 1.0)
    assert (ids[1] == -1).all()


def test_column_top2_count():
    # column 0: scores [3, 1, 2] -> best=3, second=2, count=3
    # column 1: scores [5] -> best=5, second=0 (only one), count=1
    mat = sparse.csr_matrix(np.array([[3.0, 5.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32))
    best, second, count = column_top2_count(mat)
    assert best[0] == 3.0 and second[0] == 2.0 and count[0] == 3
    assert best[1] == 5.0 and second[1] == 0.0 and count[1] == 1
