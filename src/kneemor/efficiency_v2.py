"""Backbone-efficiency study + honest accuracy, for the hybrid head (fold models of hybrid_s0).

The backbone is ~99.9% of pipeline FLOPs, so for the Efficiency track it is the only lever. Measured here:
  * slice budget: encode only every k-th slice (k=2 -> 12 slices/plane) and repeat each token to fill the
    24-slice layout the head expects (nearest-neighbour in depth). No retraining.
  * input size: 128 px instead of 160 px.
  * INT8 post-training static quantisation of ResNet-18 (FX graph mode, fbgemm), calibrated on 20
    NON-gold training studies.
Each variant re-encodes the 58 gold studies and reports gold macro AUC with the same 5 fold heads,
plus measured backbone GFLOPs and single-thread CPU latency per study.

Accuracy: per-label thresholds chosen by leave-one-out on gold (threshold fitted on 57 studies, applied
to the 58th) -> an unbiased accuracy estimate, compared with the all-negative baseline.
"""
from __future__ import annotations

import copy
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.analysis import load_fold_models  # noqa: E402
from kneemor.evaluate import macro_auc  # noqa: E402
from kneemor.features import MEAN, STD, build_backbone, load_study  # noqa: E402
from kneemor.train import load_data, predict  # noqa: E402

VOLS = "/home/user/data/vols/train"


@torch.no_grad()
def encode_variant(model, vol, step=1, size=160, grid=2):
    P, D, H, W = vol.shape
    idx = np.arange(0, D, step)
    v = vol[:, idx]
    if size != H:
        v = np.stack([[cv2.resize(sl, (size, size), interpolation=cv2.INTER_AREA) for sl in p] for p in v])
    x = torch.from_numpy(v.reshape(-1, 1, size, size)).float().div_(255).expand(-1, 3, -1, -1)
    f = model((x - MEAN) / STD)
    f = torch.nn.functional.adaptive_avg_pool2d(f, grid).flatten(2).transpose(1, 2)
    f = f.reshape(P, len(idx), grid * grid, -1)
    fill = np.minimum(np.arange(D) // step, len(idx) - 1)      # nearest-neighbour back to D slices
    return f[:, fill].half().numpy(), len(idx) * P


def quantize_int8(model, calib_ids):
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    torch.backends.quantized.engine = "fbgemm"
    ex = torch.randn(8, 3, 160, 160)
    prep = prepare_fx(copy.deepcopy(model).eval(), get_default_qconfig_mapping("fbgemm"), (ex,))
    with torch.no_grad():
        for sid in calib_ids:
            vol, _ = load_study(f"{VOLS}/{sid}.npz", 24, 160)
            x = torch.from_numpy(vol.reshape(-1, 1, 160, 160)).float().div_(255).expand(-1, 3, -1, -1)
            prep((x - MEAN) / STD)
    return convert_fx(prep)


def latency(fn, reps=3):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def loo_threshold_accuracy(y, p):
    """per label: threshold maximising accuracy on the other n-1 studies, applied to the held-out one."""
    n, L = y.shape
    hits = np.zeros((n, L), bool)
    for k in range(L):
        cand = np.unique(np.concatenate([p[:, k], [1.01]]))
        for i in range(n):
            m = np.arange(n) != i
            acc = [((p[m, k] >= t) == y[m, k]).mean() for t in cand]
            t = cand[int(np.argmax(acc))]
            hits[i, k] = (p[i, k] >= t) == y[i, k]
    return hits.mean(), hits.mean(0)


def main(key="hybrid_s0"):
    torch.set_num_threads(1)
    X, M, pool, ip, Ys, gold, ig, Yg = load_data("/home/user/data/feats", "/home/user/data/meta")
    heads = load_fold_models(key)
    ids = open("/home/user/data/feats/train_ids.txt").read().split()
    gold_ids = [ids[i] for i in ig]
    fp32 = build_backbone("/home/user/data/r18a1.pth")
    rng = np.random.default_rng(0)
    calib = list(rng.choice(pool, 20, replace=False))
    int8 = quantize_int8(fp32, calib)

    variants = {"fp32 · 24 sl · 160px": (fp32, 1, 160), "fp32 · 12 sl · 160px": (fp32, 2, 160),
                "fp32 · 8 sl · 160px": (fp32, 3, 160), "fp32 · 24 sl · 128px": (fp32, 1, 128),
                "fp32 · 12 sl · 128px": (fp32, 2, 128), "int8 · 24 sl · 160px": (int8, 1, 160),
                "int8 · 12 sl · 128px": (int8, 2, 128)}
    out = {}
    vol0, _ = load_study(f"{VOLS}/{gold_ids[0]}.npz", 24, 160)
    for name, (bb, step, size) in variants.items():
        Xg = np.zeros((len(ig),) + X.shape[1:], np.float16)
        for n, sid in enumerate(gold_ids):
            vol, _ = load_study(f"{VOLS}/{sid}.npz", 24, 160)
            Xg[n], _ = encode_variant(bb, vol, step, size)
        g = np.mean([predict(h, Xg, M[ig], np.arange(len(ig))) for h in heads], 0)
        n_img = 3 * len(range(0, 24, step))
        if bb is fp32:
            with FlopCounterMode(display=False) as fc:
                with torch.no_grad():
                    bb(torch.randn(n_img, 3, size, size))
            gfl = fc.get_total_flops() / 1e9
        else:
            gfl = None  # FlopCounter does not count quantized kernels; ~same op count as fp32 variant
        lat = latency(lambda: encode_variant(bb, vol0, step, size))
        out[name] = dict(gold_auc=macro_auc(Yg, g)[0], backbone_gflops=gfl, backbone_latency_ms=lat,
                         slices_encoded=n_img)
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out[name].items()}, flush=True)

    # honest accuracy on gold for the hybrid 3-seed ensemble
    g_ens = np.mean([np.load(f"results/preds/hybrid_s{s}_gold.npy") for s in range(3)], 0)
    acc, acc_lab = loo_threshold_accuracy(Yg.astype(int), g_ens)
    out["accuracy"] = dict(hybrid_loo_threshold_acc=float(acc), all_negative_acc=float((Yg == 0).mean()),
                           per_label=dict(zip(__import__("kneemor.report_labeler", fromlist=["LABELS"]).LABELS,
                                              acc_lab.round(4).tolist())))
    print(out["accuracy"])
    json.dump(out, open("results/efficiency_v2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
