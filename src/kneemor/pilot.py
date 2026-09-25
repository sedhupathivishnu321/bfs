"""Leakage-safe pilot for the shared training budget.

Uses only the TRAINING part of CV fold 0, split 85/15 into inner-train / inner-val
(silver labels). The learning curve (inner-val macro AUC per epoch) picks one epoch
budget and learning rate that are then fixed for every head in the main experiments.
Neither the outer fold-0 test part nor the gold studies are touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.evaluate import macro_auc  # noqa: E402
from kneemor.models import build  # noqa: E402
from kneemor.train import augment, load_data, make_folds, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mvmor")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    X, M, pool, ip, Ys, *_ = load_data("/home/user/data/feats", "/home/user/data/meta")
    X = np.ascontiguousarray(X)
    fold = make_folds(pool, Ys, 5, "results/folds.csv")
    tr = np.where(fold != 0)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(tr)
    nv = int(0.15 * len(tr))
    iv, it = tr[:nv], tr[nv:]
    model = build(a.model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * int(np.ceil(len(it) / a.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    ys, vs = (Ys[iv] >= 1).astype(int), Ys[iv] != 0.5
    curve = []
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(it))
        for i in range(0, len(perm), a.bs):
            j = np.sort(ip[it[perm[i:i + a.bs]]])
            b = it[perm[i:i + a.bs]][np.argsort(ip[it[perm[i:i + a.bs]]])]
            x, m = augment(torch.from_numpy(np.asarray(X[j], np.float32)), torch.from_numpy(M[j].copy()))
            loss = F.binary_cross_entropy_with_logits(model(x, m), torch.from_numpy(Ys[b]))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        auc = macro_auc(ys, predict(model, X, M, ip[iv]), vs)[0]
        curve.append(auc)
        print(f"{a.model} lr={a.lr} ep{ep + 1} inner-val macroAUC={auc:.4f}", flush=True)
    os.makedirs("results/pilot", exist_ok=True)
    json.dump(dict(model=a.model, lr=a.lr, epochs=a.epochs, curve=curve),
              open(f"results/pilot/{a.model}_lr{a.lr}_e{a.epochs}.json", "w"))


if __name__ == "__main__":
    main()
