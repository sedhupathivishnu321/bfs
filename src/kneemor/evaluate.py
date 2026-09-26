"""Metrics, confidence intervals and significance tests.

* OOF (silver) evaluation: per label, studies whose silver label is uncertain
  (0.5) are excluded; y = silver >= 1.
* Gold evaluation: 58 expert-labelled studies (ensemble of the 5 fold models).
* Threshold-dependent metrics (sens/spec/precision/F1) use a per-label
  threshold chosen by Youden's J on the OOF predictions (never on gold).
* 95% CIs: study-level bootstrap (2000 resamples). Model comparisons: paired
  bootstrap of the macro-AUC difference (two-sided p-value).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneemor.report_labeler import LABELS  # noqa: E402


def macro_auc(y, p, valid=None):
    aucs = []
    for k in range(y.shape[1]):
        v = np.ones(len(y), bool) if valid is None else valid[:, k]
        yy, pp = y[v, k], p[v, k]
        if 0 < yy.sum() < len(yy):
            aucs.append(roc_auc_score(yy, pp))
        else:
            aucs.append(np.nan)
    return np.nanmean(aucs), np.array(aucs)


def youden_thresholds(y, p, valid):
    th = []
    for k in range(y.shape[1]):
        v = valid[:, k]
        fpr, tpr, t = roc_curve(y[v, k], p[v, k])
        th.append(t[np.argmax(tpr - fpr)])
    return np.array(th)


def threshold_metrics(y, p, th):
    rows = []
    for k in range(y.shape[1]):
        yh = p[:, k] >= th[k]
        yy = y[:, k] == 1
        tp, fp = (yh & yy).sum(), (yh & ~yy).sum()
        fn, tn = (~yh & yy).sum(), (~yh & ~yy).sum()
        sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * sens / max(prec + sens, 1e-9)
        rows.append(dict(sens=sens, spec=spec, prec=prec, f1=f1, acc=(tp + tn) / len(yy)))
    return pd.DataFrame(rows, index=LABELS)


def bootstrap_ci(y, p, valid=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(macro_auc(y[i], p[i], None if valid is None else valid[i])[0])
    return np.nanpercentile(vals, [2.5, 97.5])


def paired_bootstrap(y, pa, pb, valid=None, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    obs = macro_auc(y, pa, valid)[0] - macro_auc(y, pb, valid)[0]
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = None if valid is None else valid[i]
        d.append(macro_auc(y[i], pa[i], v)[0] - macro_auc(y[i], pb[i], v)[0])
    d = np.array(d)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return obs, np.percentile(d, [2.5, 97.5]), min(p, 1.0)


def load_runs(pred_dir):
    runs = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, "*_oof.npy"))):
        key = os.path.basename(f)[:-8]
        name, seed = re.match(r"(.+)_s(\d+)$", key).groups()
        runs.setdefault(name, {})[int(seed)] = dict(
            oof=np.load(f), gold=np.load(f.replace("_oof.npy", "_gold.npy")),
            info=json.load(open(f.replace("_oof.npy", "_info.json"))))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--ref", default="mvmor")
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    pd_ = os.path.join(a.res, "preds")
    Ys = np.load(os.path.join(pd_, "pool_labels.npy"))
    Yg = np.load(os.path.join(pd_, "gold_labels.npy"))
    ys, vs = (Ys >= 1).astype(int), Ys != 0.5
    runs = load_runs(pd_)

    rows, per_label, thr_rows = [], [], []
    ens = {}
    for name, seeds in runs.items():
        full = {s: r for s, r in seeds.items() if not np.isnan(r["oof"]).any()}
        if not full:
            continue
        oof_aucs = [macro_auc(ys, r["oof"], vs)[0] for r in full.values()]
        gold_aucs = [macro_auc(Yg, r["gold"])[0] for r in full.values()]
        oof_e = np.mean([r["oof"] for r in full.values()], 0)
        gold_e = np.mean([r["gold"] for r in full.values()], 0)
        ens[name] = (oof_e, gold_e)
        info = next(iter(full.values()))["info"]
        g_auc, g_lab = macro_auc(Yg, gold_e)
        o_auc, o_lab = macro_auc(ys, oof_e, vs)
        th = youden_thresholds(ys, oof_e, vs)
        tm = threshold_metrics(Yg, gold_e, th)
        rows.append(dict(model=name, n_seeds=len(full), params=info["params"],
                         oof_auc_mean=np.mean(oof_aucs), oof_auc_sd=np.std(oof_aucs),
                         gold_auc_mean=np.mean(gold_aucs), gold_auc_sd=np.std(gold_aucs),
                         oof_auc_ens=o_auc, gold_auc_ens=g_auc,
                         gold_sens=tm.sens.mean(), gold_spec=tm.spec.mean(), gold_prec=tm.prec.mean(),
                         gold_f1=tm.f1.mean(), gold_acc=tm.acc.mean(),
                         train_min_per_seed=info["train_seconds"] / 60))
        per_label.append(pd.Series(g_lab, index=LABELS, name=f"{name}|gold"))
        per_label.append(pd.Series(o_lab, index=LABELS, name=f"{name}|oof"))
        thr_rows.append(tm.assign(model=name))
    df = pd.DataFrame(rows).sort_values("gold_auc_ens", ascending=False)

    # CIs and paired tests vs reference model
    ci_rows = []
    for name, (oof_e, gold_e) in ens.items():
        lo, hi = bootstrap_ci(Yg, gold_e, n=a.nboot)
        olo, ohi = bootstrap_ci(ys, oof_e, vs, n=max(200, a.nboot // 5))
        r = dict(model=name, gold_ci_lo=lo, gold_ci_hi=hi, oof_ci_lo=olo, oof_ci_hi=ohi)
        if a.ref in ens and name != a.ref:
            d, ci, p = paired_bootstrap(Yg, ens[a.ref][1], gold_e, n=a.nboot)
            r.update(gold_diff_vs_ref=d, gold_diff_p=p)
            d, ci, p = paired_bootstrap(ys, ens[a.ref][0], oof_e, vs, n=max(200, a.nboot // 5))
            r.update(oof_diff_vs_ref=d, oof_diff_p=p)
        ci_rows.append(r)
    df = df.merge(pd.DataFrame(ci_rows), on="model")

    # like-for-like: seed-0 run of every model vs seed-0 run of the reference (ablations have 1 seed)
    s0_rows = []
    if a.ref in runs and 0 in runs[a.ref]:
        ra = runs[a.ref][0]
        for name, seeds in runs.items():
            if name == a.ref or 0 not in seeds or np.isnan(seeds[0]["oof"]).any():
                continue
            rb = seeds[0]
            gd, gci, gp = paired_bootstrap(Yg, ra["gold"], rb["gold"], n=a.nboot)
            od, oci, op = paired_bootstrap(ys, ra["oof"], rb["oof"], vs, n=max(200, a.nboot // 5))
            s0_rows.append(dict(model=name, ref=a.ref, oof_auc_ref=macro_auc(ys, ra["oof"], vs)[0],
                                oof_auc=macro_auc(ys, rb["oof"], vs)[0], oof_diff=od, oof_ci_lo=oci[0],
                                oof_ci_hi=oci[1], oof_p=op, gold_auc_ref=macro_auc(Yg, ra["gold"])[0],
                                gold_auc=macro_auc(Yg, rb["gold"])[0], gold_diff=gd, gold_ci_lo=gci[0],
                                gold_ci_hi=gci[1], gold_p=gp))
        pd.DataFrame(s0_rows).to_csv(os.path.join(a.res, "paired_seed0_vs_ref.csv"), index=False)
    df.to_csv(os.path.join(a.res, "summary.csv"), index=False)
    pd.concat(per_label, axis=1).T.to_csv(os.path.join(a.res, "per_label_auc.csv"))
    pd.concat(thr_rows).to_csv(os.path.join(a.res, "gold_threshold_metrics.csv"))
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
