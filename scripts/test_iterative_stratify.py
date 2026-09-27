"""Standalone test of iterative multilabel stratification (Sechidis, Tsoumakas & Vlahavas,
ECML PKDD 2011), run and passing *before* the same implementation was inlined into
notebooks/build_kaggle_notebook.py (see README "Notebook pipeline fixes"). Needs only numpy.

    python scripts/test_iterative_stratify.py

Verifies: every example assigned exactly once, fold sizes balanced, and per-label positive
counts balanced across folds, at a scale matching the actual training pool (~4,349 studies,
12 imbalanced labels) plus two edge cases (a tiny 58-study set, and a label with one positive).
"""
import numpy as np


def iterative_stratify(Y, k, seed=42):
    """Y: [N, L] soft labels in {0, 0.5, 1}. 0.5 (uncertain) counts toward neither balancing
    target. Returns fold id in [0, k) per row, aiming for equal per-label positive counts and
    equal fold sizes. Identical to the copy inlined in notebooks/build_kaggle_notebook.py."""
    rng = np.random.default_rng(seed)
    N, L = Y.shape
    pos, valid = (Y >= 1), (Y != 0.5)
    remaining = np.ones(N, dtype=bool)
    fold_of = np.full(N, -1, dtype=int)
    desired_per_label = np.array([[pos[:, l].sum() / k] * k for l in range(L)])
    desired_size = np.full(k, N / k)
    fold_pos_count = np.zeros((L, k)); fold_size = np.zeros(k)
    order = np.arange(N)
    while remaining.any():
        counts = np.array([(pos[:, l] & valid[:, l] & remaining).sum() for l in range(L)])
        if counts.max() > 0:
            candidates = np.where(counts == counts[counts > 0].min())[0]
            l = rng.choice(candidates)
            idxs = order[remaining & pos[:, l] & valid[:, l]]
        else:
            idxs = order[remaining]     # no labelled positives left: balance remaining by size only
        rng.shuffle(idxs)
        for i in idxs:
            deficit = (desired_per_label[l] - fold_pos_count[l]) if counts.max() > 0 else (desired_size - fold_size)
            best = np.flatnonzero(deficit == deficit.max())
            f = rng.choice(best) if len(best) > 1 else best[0]
            fold_of[i] = f; fold_size[f] += 1
            fold_pos_count[:, f] += pos[i] & valid[i]
            remaining[i] = False
    return fold_of


def check(Y, k=5, seed=42):
    folds = iterative_stratify(Y, k, seed)
    N, L = Y.shape
    assert (folds >= 0).all() and (folds < k).all(), "every row must be assigned"
    assert len(set(folds.tolist())) == k, "every fold must be used"
    sizes = np.bincount(folds, minlength=k)
    print(f"  N={N} L={L}: fold sizes {sizes.tolist()} (target {N / k:.1f})")
    pos, valid = (Y >= 1), (Y != 0.5)
    worst = 0.0
    for l in range(L):
        per_fold = np.array([((folds == f) & pos[:, l] & valid[:, l]).sum() for f in range(k)])
        total = per_fold.sum()
        if total == 0:
            continue
        worst = max(worst, (per_fold.max() - per_fold.min()) / max(total / k, 1))
    print(f"  worst per-label fold-count relative spread: {worst:.2f}")
    return sizes, worst


def main():
    rng = np.random.default_rng(0)

    print("Test 1: realistic RSNA-like prevalence (12 labels, imbalanced, some rare, N=4349)")
    N, L = 4349, 12
    base_prev = np.array([0.25, 0.05, 0.15, 0.10, 0.30, 0.20, 0.22, 0.35, 0.08, 0.12, 0.18, 0.06])
    Y = np.zeros((N, L))
    for l in range(L):
        Y[:, l] = rng.random(N) < base_prev[l]
    Y[rng.random((N, L)) < 0.05] = 0.5
    sizes, worst = check(Y, 5, 42)
    assert abs(sizes.max() - sizes.min()) <= 2, f"fold sizes too unbalanced: {sizes}"
    assert worst < 0.5, f"per-label balance too poor: {worst}"

    print("\nTest 2: tiny gold-like set (n=58, 12 labels) into 5 folds -- small-n edge case")
    N = 58
    Y2 = np.zeros((N, L))
    for l in range(L):
        Y2[:, l] = rng.random(N) < base_prev[l]
    check(Y2, 5, 0)

    print("\nTest 3: a label with a single positive example (extreme rarity)")
    Y3 = np.zeros((100, 3))
    Y3[0, 0] = 1
    Y3[:, 1] = rng.random(100) < 0.5
    Y3[:, 2] = rng.random(100) < 0.5
    folds3 = iterative_stratify(Y3, 5, 1)
    assert (folds3 >= 0).all()
    print("  single-positive label handled without crash; that row's fold:", folds3[0])

    print("\nALL ITERATIVE STRATIFICATION TESTS PASSED")


if __name__ == "__main__":
    main()
