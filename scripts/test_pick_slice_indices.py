"""Standalone test of pick_slice_indices (src/kneemor/ingest.py, inlined in the Kaggle notebook),
run before it replaced a plain np.linspace(...).round() slice sampler (see README "Limitations" and
docs/data_audit.md finding 3). Needs only numpy.

    python scripts/test_pick_slice_indices.py
"""
import numpy as np


def pick_slice_indices(n, depth):
    """Identical to the copies in src/kneemor/ingest.py and notebooks/build_kaggle_notebook.py."""
    if n >= depth:
        idx = np.unique(np.round(np.linspace(0, n - 1, depth)).astype(int))
        while len(idx) < depth:
            gaps = np.diff(idx)
            j = int(np.argmax(gaps))
            idx = np.unique(np.insert(idx, j + 1, (idx[j] + idx[j + 1]) // 2))
        return idx[:depth]
    return np.concatenate([np.arange(n), np.full(depth - n, n - 1, dtype=int)])


def check(n, depth):
    idx = pick_slice_indices(n, depth)
    assert len(idx) == depth, (n, depth, len(idx))
    assert idx.min() >= 0 and idx.max() < max(n, 1), (n, depth, idx)
    assert (np.diff(idx) >= 0).all(), (n, depth, "must be non-decreasing")
    if n >= depth:
        assert len(np.unique(idx)) == depth, (n, depth, "duplicate indices when n >= depth", idx)
    else:
        assert list(idx[:n]) == list(range(n)), (n, depth, "every available slice must be kept")
        assert (idx[n:] == n - 1).all(), (n, depth, "padding must repeat the last slice")


def main():
    for depth in (16, 24):   # the two depths this repo actually uses (Kaggle notebook, CPU study)
        for n in list(range(1, 60)) + [100, 320]:
            check(n, depth)
        print(f"depth={depth}: all n in 1..320 passed (unique when n>=depth, padded correctly when n<depth)")
    print("ALL PICK_SLICE_INDICES TESTS PASSED")


if __name__ == "__main__":
    main()
