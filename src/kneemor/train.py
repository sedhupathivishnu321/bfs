"""Cross-validated training of study-level heads on frozen slice tokens.

Protocol (identical for every model):
  * Training pool  = studies WITHOUT expert labels (silver report labels, soft {0,.5,1}).
  * Gold test set  = the 58 expert-labelled studies. Never used for training,
                     model selection, early stopping or hyper-parameters.
  * 5-fold CV on the training pool (fixed folds, results/folds.csv). Each fold
    model predicts its held-out fold (OOF) and the gold set; gold predictions are
    averaged over the 5 fold models (ensemble).
  * Fixed epochs / optimiser for all heads (no early stopping -> no selection leakage).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.models import build, n_params  # noqa: E402
from kneemor.report_labeler import LABELS  # noqa: E402

PLANES = ["Sagittal", "Coronal", "Axial"]


def load_data(feat_dir, meta_dir):
    ids = open(os.path.join(feat_dir, "train_ids.txt")).read().split()
    X = np.load(os.path.join(feat_dir, "train_feats.npy"), mmap_mode="r")
    M = np.load(os.path.join(feat_dir, "train_mask.npy"))
    tr = pd.read_csv(os.path.join(meta_dir, "train.csv"))
    silver = pd.read_csv(os.path.join(meta_dir, "silver_labels.csv")).set_index("StudyInstanceUID")
    gold_ids = set(tr.loc[tr[LABELS].notna().all(axis=1), "StudyInstanceUID"])
    pos = {s: i for i, s in enumerate(ids)}
    pool = [s for s in ids if s not in gold_ids]
    gold = sorted(s for s in gold_ids if s in pos)
    Ys = silver.loc[pool, LABELS].values.astype(np.float32)
    Yg = tr.set_index("StudyInstanceUID").loc[gold, LABELS].values.astype(np.float32)
    ip = np.array([pos[s] for s in pool])
    ig = np.array([pos[s] for s in gold])
    return X, M, pool, ip, Ys, gold, ig, Yg


def make_folds(pool, Ys, k, path, seed=42):
    if os.path.exists(path):
        f = pd.read_csv(path).set_index("StudyInstanceUID").loc[pool, "fold"].values
        return f
    strat = np.clip((Ys >= 0.5).sum(1), 0, 5) * 2 + (Ys[:, LABELS.index("ACL")] >= 0.5)
    fold = np.zeros(len(pool), int)
    for i, (_, te) in enumerate(StratifiedKFold(k, shuffle=True, random_state=seed).split(pool, strat)):
        fold[te] = i
    pd.DataFrame(dict(StudyInstanceUID=pool, fold=fold)).to_csv(path, index=False)
    return fold


def augment(x, m, p_plane=0.15, p_tok=0.1):
    """Feature-space augmentation: whole-plane dropout (robustness to missing
    sequences) and random slice-token masking."""
    B, P = m.shape
    drop = (torch.rand(B, P) < p_plane) & m
    keep_one = (m & ~drop).sum(1) == 0
    drop[keep_one] = False
    m = m & ~drop
    tokmask = (torch.rand(x.shape[:4]) > p_tok).unsqueeze(-1)
    return x * tokmask, m


def predict(model, X, M, idx, bs=64, planes=None):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            x = torch.from_numpy(np.asarray(X[j], dtype=np.float32))
            m = torch.from_numpy(M[j].copy())
            if planes is not None:
                m = m & planes
            out.append(torch.sigmoid(model(x, m)).numpy())
    return np.concatenate(out)


def train_one(name, X, M, idx_tr, Y_tr, args, seed, planes=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(name)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * int(np.ceil(len(idx_tr) / args.bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1,
                                                anneal_strategy="cos")
    Yt = torch.from_numpy(Y_tr)
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(idx_tr))
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            b = perm[i:i + args.bs]
            j = idx_tr[b]
            order = np.argsort(j)  # sorted reads are faster on the memmap
            x = torch.from_numpy(np.asarray(X[j[order]], dtype=np.float32))
            m = torch.from_numpy(M[j[order]].copy())
            y = Yt[b[order]]
            if planes is not None:
                m = m & planes
                if (m.sum(1) == 0).any():
                    m[m.sum(1) == 0] = planes
            x, m = augment(x, m)
            loss = F.binary_cross_entropy_with_logits(model(x, m), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        if args.verbose:
            print(f"    ep{ep} loss {tot / len(perm):.4f}", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="/home/user/data/feats")
    ap.add_argument("--meta", default="/home/user/data/meta")
    ap.add_argument("--out", default="results")
    ap.add_argument("--models", default="mvmor")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--planes", default="Sagittal,Coronal,Axial")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--save-models", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    X, M, pool, ip, Ys, gold, ig, Yg = load_data(args.feats, args.meta)
    print(f"pool={len(pool)} gold={len(gold)} tokens/study={np.prod(X.shape[1:4])}", flush=True)
    X = np.ascontiguousarray(X)  # load into RAM (float16)
    os.makedirs(os.path.join(args.out, "preds"), exist_ok=True)
    fold = make_folds(pool, Ys, args.folds, os.path.join(args.out, "folds.csv"))
    np.save(os.path.join(args.out, "preds", "pool_labels.npy"), Ys)
    np.save(os.path.join(args.out, "preds", "gold_labels.npy"), Yg)
    pl = torch.tensor([p in args.planes.split(",") for p in PLANES])
    planes = None if pl.all() else pl
    run_folds = [int(f) for f in args.only_folds.split(",")] if args.only_folds else range(args.folds)

    for name in args.models.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            key = f"{name}{args.tag}_s{seed}"
            oof = np.full(Ys.shape, np.nan, np.float32)
            gp = []
            t0 = time.time()
            for f in run_folds:
                tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
                model = train_one(name, X, M, ip[tr], Ys[tr], args, seed * 100 + f, planes)
                oof[te] = predict(model, X, M, ip[te], planes=planes)
                gp.append(predict(model, X, M, ig, planes=planes))
                if args.save_models:
                    os.makedirs(os.path.join(args.out, "models"), exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(args.out, "models", f"{key}_f{f}.pt"))
                print(f"  {key} fold{f} done {time.time() - t0:.0f}s", flush=True)
            np.save(os.path.join(args.out, "preds", f"{key}_oof.npy"), oof)
            np.save(os.path.join(args.out, "preds", f"{key}_gold.npy"), np.mean(gp, 0))
            info = dict(model=name, tag=args.tag, seed=seed, params=n_params(build(name)),
                        train_seconds=time.time() - t0, epochs=args.epochs, lr=args.lr, wd=args.wd,
                        bs=args.bs, planes=args.planes, folds=list(run_folds))
            json.dump(info, open(os.path.join(args.out, "preds", f"{key}_info.json"), "w"), indent=1)
            print(f"== {key} finished in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
