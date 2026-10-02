"""Aggregate, statistics, figures. Everything printed/plotted here is computed from the CSVs written by the benchmark runs."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1]; (R / "results/figures").mkdir(exist_ok=True, parents=True)
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

for proto, suite in (("A", "main"), ("B", "main"), ("A", "ablation"), ("B", "ablation")):
    f = R / f"results/protocol{proto}_{suite}.csv"
    if not f.exists(): continue
    d = pd.read_csv(f); P(f"\n## Protocol {proto} / {suite}  (folds={d.fold.nunique()}, seeds={d.seed.nunique()})")
    cols = ["MAE_gain_0.5s", "MAE_gain_1.0s", "MAE_gain_2.0s", "MAE_delay_1.0s", "ev_AUC", "ev_F1", "ev_recall_sens", "ev_specificity", "cov90"]
    cols = [c for c in cols if c in d]
    # fold-level mean over seeds, then mean±std across folds
    fm = d.groupby(["model", "fold"])[cols].mean().reset_index()
    agg = fm.groupby("model")[cols].agg(["mean", "std"])
    tab = pd.DataFrame({c: agg[c]["mean"].round(4).astype(str) + "±" + agg[c]["std"].round(4).astype(str) for c in cols})
    tab = tab.loc[fm.groupby("model")["MAE_gain_1.0s"].mean().sort_values().index]
    P(tab.to_markdown())
    tab.to_csv(R / f"results/summary_{proto}_{suite}.csv")
    ref = "PCT (proposed)" if suite == "main" else "PCT full"
    if ref in fm.model.values and fm.fold.nunique() >= 2:
        P(f"\nPaired comparison vs {ref} on MAE_gain_1.0s (fold-level means; Wilcoxon signed-rank + mean relative change):")
        piv = fm.pivot(index="fold", columns="model", values="MAE_gain_1.0s")
        for m in piv.columns:
            if m == ref: continue
            diff = piv[m] - piv[ref]
            try: p = stats.wilcoxon(diff).pvalue if len(diff) >= 5 else float("nan")
            except ValueError: p = float("nan")
            P(f"  {m:30s} Δ(other-ref)={diff.mean():+.4f} dB  rel={100*diff.mean()/piv[m].mean():+.1f}%  folds ref-better={int((diff>0).sum())}/{len(diff)}  p={p:.3f}")
    # window-level paired bootstrap (pooled over folds, seed 0) of PCT vs best baseline
    z = R / f"results/abs_err_{proto}_{suite}.npz"
    if z.exists() and suite == "main":
        E = np.load(z); folds = sorted({k.split("|")[0] for k in E.files})
        def pooled(m): return np.concatenate([E[f"{fo}|0|{m}"] for fo in folds])
        base = fm.groupby("model")["MAE_gain_1.0s"].mean().drop([ref]).idxmin()
        a_, b_ = pooled(ref), pooled(base); rng = np.random.default_rng(0)
        blk = 40; nb = len(a_) // blk; A = a_[:nb * blk].reshape(nb, blk).mean(1); Bm = b_[:nb * blk].reshape(nb, blk).mean(1)  # block bootstrap (temporal autocorr)
        bs = [(Bm[i] - A[i]).mean() for i in (rng.integers(0, nb, nb) for _ in range(2000))]
        P(f"\nBlock-bootstrap (block=40 windows) MAE(best baseline={base}) − MAE({ref}) = {np.mean(bs):+.4f} dB, 95% CI [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")
    # figure
    fig, ax = plt.subplots(figsize=(8, 4.2)); t = tab.copy(); mu = fm.groupby("model")["MAE_gain_1.0s"].mean().loc[t.index]; sd = fm.groupby("model")["MAE_gain_1.0s"].std().loc[t.index]
    ax.barh(range(len(mu)), mu, xerr=sd, color=["#c4562d" if m == ref else "#8aa" for m in mu.index]); ax.set_yticks(range(len(mu))); ax.set_yticklabels(mu.index); ax.invert_yaxis()
    ax.set_xlabel("MAE of 1 s-ahead gain change (dB), mean ± std over folds"); ax.set_title(f"Protocol {proto} – {suite}"); fig.tight_layout(); fig.savefig(R / f"results/figures/mae_{proto}_{suite}.png", dpi=150); plt.close(fig)
(R / "results/analysis_report.md").write_text("\n".join(out))
