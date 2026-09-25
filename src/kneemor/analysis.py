"""Explainability, robustness and error analysis for the proposed MV-MoR head.

1. Label-query attention -> share of attention mass per MRI plane for each label
   (averaged over gold studies): does each finding look at the clinically
   expected plane?
2. Router depth -> mean number of recursions per plane / slice position.
3. Robustness: (a) missing-plane at inference, (b) image perturbations on the
   gold studies (Gaussian noise, gamma, low resolution) re-encoded through the
   frozen backbone.
4. Error analysis: OOF AUC by report language and by scanner vendor.
Models are the 5 fold checkpoints saved by train.py --save-models.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.evaluate import macro_auc  # noqa: E402
from kneemor.features import build_backbone, encode, load_study  # noqa: E402
from kneemor.models import build  # noqa: E402
from kneemor.report_labeler import LABELS  # noqa: E402
from kneemor.train import PLANES, load_data, make_folds, predict  # noqa: E402

RES = "results"


def load_fold_models(key):
    ms = []
    for f in sorted(glob.glob(os.path.join(RES, "models", f"{key}_f*.pt"))):
        m = build(key.rsplit("_s", 1)[0])
        m.load_state_dict(torch.load(f, map_location="cpu"))
        ms.append(m.eval())
    return ms


def ens_predict(models, X, M, idx, planes=None):
    return np.mean([predict(m, X, M, idx, planes=planes) for m in models], 0)


def attention_and_depth(models, X, M, idx):
    P, D, G = X.shape[1:4]
    att = np.zeros((len(LABELS), P))
    depth_pd = np.zeros((P, D))
    n = 0
    with torch.no_grad():
        for m in models:
            x = torch.from_numpy(np.asarray(X[idx], np.float32))
            mk = torch.from_numpy(M[idx].copy())
            m(x, mk)
            w = m.dec.last_attn.numpy().reshape(len(idx), len(LABELS), P, D, G)  # [B, L, P, D, G]
            att += w.sum((3, 4)).mean(0)
            depth_pd += m.last_depth.numpy().reshape(len(idx), P, D, G).mean((0, 3))
            n += 1
    return att / n, depth_pd / n


def perturb(vol, kind, rng):
    v = vol.astype(np.float32) / 255
    if kind == "noise":
        v = v + rng.normal(0, 0.05, v.shape)
    elif kind == "gamma_0.7":
        v = np.clip(v, 0, 1) ** 0.7
    elif kind == "gamma_1.5":
        v = np.clip(v, 0, 1) ** 1.5
    elif kind == "lowres":
        s = v.shape[-1]
        v = np.stack([[cv2.resize(cv2.resize(sl, (s // 2, s // 2), interpolation=cv2.INTER_AREA), (s, s))
                       for sl in p] for p in v])
    return (np.clip(v, 0, 1) * 255).round().astype(np.uint8)


def main(key="mvmor_s0"):
    out = {}
    X, M, pool, ip, Ys, gold, ig, Yg = load_data("/home/user/data/feats", "/home/user/data/meta")
    X = np.ascontiguousarray(X)
    models = load_fold_models(key)
    assert len(models) == 5, "run train.py --save-models for the reference key first"
    fold = make_folds(pool, Ys, 5, os.path.join(RES, "folds.csv"))

    # 1-2. explainability
    att, depth = attention_and_depth(models, X, M, ig)
    out["attention_plane_share"] = {LABELS[i]: dict(zip(PLANES, att[i].round(4).tolist())) for i in range(len(LABELS))}
    out["router_depth_by_plane"] = dict(zip(PLANES, depth.mean(1).round(4).tolist()))
    out["router_depth_by_slice"] = {p: depth[i].round(3).tolist() for i, p in enumerate(PLANES)}

    # 3a. missing plane at inference (gold ensemble + OOF of each fold model on its own held-out fold)
    base_g = ens_predict(models, X, M, ig)
    out["gold_auc_full"] = macro_auc(Yg, base_g)[0]
    ys, vs = (Ys >= 1).astype(int), Ys != 0.5
    for drop in PLANES:
        keep = torch.tensor([p != drop for p in PLANES])
        g = ens_predict(models, X, M, ig, planes=keep)
        oof = np.zeros(Ys.shape, np.float32)
        for f, m in enumerate(models):
            te = np.where(fold == f)[0]
            oof[te] = predict(m, X, M, ip[te], planes=keep)
        out[f"drop_{drop}"] = dict(gold_auc=macro_auc(Yg, g)[0], oof_auc=macro_auc(ys, oof, vs)[0])

    # 3b. image perturbations on gold studies
    bb = build_backbone("/home/user/data/r18a1.pth")
    rng = np.random.default_rng(0)
    ids = open("/home/user/data/feats/train_ids.txt").read().split()
    for kind in ["noise", "gamma_0.7", "gamma_1.5", "lowres"]:
        Xp = np.zeros((len(ig),) + X.shape[1:], np.float16)
        for n, i in enumerate(ig):
            vol, _ = load_study(f"/home/user/data/vols/train/{ids[i]}.npz", 24, 160)
            Xp[n] = encode(bb, perturb(vol, kind, rng), 2)
        g = np.mean([predict(m, Xp, M[ig], np.arange(len(ig))) for m in models], 0)
        out[f"perturb_{kind}"] = dict(gold_auc=macro_auc(Yg, g)[0])
        print(kind, out[f"perturb_{kind}"], flush=True)

    # 4. error analysis by language / vendor (OOF, silver)
    oof = np.load(os.path.join(RES, "preds", f"{key}_oof.npy"))
    lang = pd.read_csv("/home/user/data/meta/lang.csv").set_index("StudyInstanceUID").loc[pool, "lang"].values
    out["oof_auc_by_language"] = {}
    for l in pd.Series(lang).value_counts().index:
        s = lang == l
        out["oof_auc_by_language"][l] = dict(n=int(s.sum()), auc=macro_auc(ys[s], oof[s], vs[s])[0])
    meta = pd.read_csv("/home/user/data/vols/train/series_meta.csv")
    vend = meta.groupby("StudyInstanceUID").manufacturer.first().str.upper().str.extract(
        r"(SIEMENS|GE|PHILIPS|TOSHIBA|CANON|HITACHI)")[0]
    v = vend.reindex(pool).values
    out["oof_auc_by_vendor"] = {}
    for name in ["SIEMENS", "GE", "PHILIPS", "TOSHIBA"]:
        s = v == name
        if s.sum() > 50:
            out["oof_auc_by_vendor"][name] = dict(n=int(s.sum()), auc=macro_auc(ys[s], oof[s], vs[s])[0])
    json.dump(out, open(os.path.join(RES, "analysis.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main(*sys.argv[1:])
