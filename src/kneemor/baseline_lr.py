"""Classical baseline: per-plane mean-pooled frozen tokens (3x512) -> standardise -> L2 logistic regression
per label. Same folds, same gold ensemble protocol as the neural heads. Silver 0.5 is binarised to 1."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.train import load_data, make_folds  # noqa: E402


def pooled(X, M, idx):
    z = np.stack([np.asarray(X[i], np.float32).mean((1, 2)) for i in idx])  # [n, P, C]
    z = z * M[idx][..., None]
    return z.reshape(len(idx), -1)


def main(out="results", C=0.05):
    X, M, pool, ip, Ys, gold, ig, Yg = load_data("/home/user/data/feats", "/home/user/data/meta")
    fold = make_folds(pool, Ys, 5, os.path.join(out, "folds.csv"))
    Zp, Zg = pooled(X, M, ip), pooled(X, M, ig)
    Yb = (Ys >= 0.5).astype(int)
    oof = np.zeros(Ys.shape, np.float32)
    gp = np.zeros((5,) + Yg.shape, np.float32)
    t0 = time.time()
    for f in range(5):
        tr, te = fold != f, fold == f
        for k in range(Ys.shape[1]):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            clf.fit(Zp[tr], Yb[tr, k])
            oof[te, k] = clf.predict_proba(Zp[te])[:, 1]
            gp[f, :, k] = clf.predict_proba(Zg)[:, 1]
    np.save(os.path.join(out, "preds", "logreg_s0_oof.npy"), oof)
    np.save(os.path.join(out, "preds", "logreg_s0_gold.npy"), gp.mean(0))
    json.dump(dict(model="logreg", seed=0, params=int(Zp.shape[1] * 12 + 12), train_seconds=time.time() - t0,
                   epochs=None, C=C), open(os.path.join(out, "preds", "logreg_s0_info.json"), "w"))
    print("logreg done", time.time() - t0)


if __name__ == "__main__":
    main()
