"""Generate every figure that the REAL measured data and the trained models can support, and an INDEX.md that accounts for all 70 requested
figures (produced / produced-with-adaptation / not produced + reason). No synthetic or placeholder figures are generated."""
import sys, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import data as D

OUT = R / "results/plots"; ART = R / "results/artifacts"
DIRS = dict(A="A_prediction", B="B_uncertainty", C="C_link_optical", E="E_baselines", F="F_ablation", G="G_scalability", H="H_robustness")
for d in DIRS.values(): (OUT / d).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "font.size": 9,
                     "axes.titlesize": 10, "figure.autolayout": True})
OURS, GREY = "#c4562d", "#8a8f98"
PAL = {"PCT-E (ours)": OURS, "PCT-v2 (single)": "#e59a7c", "GBM": "#2f6f9f", "LSTM": "#6a9f58", "GRU": "#a67bb5", "TCN": "#d4a73a", "Transformer": "#3a9fa3", "Persistence": "#4d4d4d"}
NOM_RANGE_KM = dict(black=8.72, blue=5.0, purple=0.2, red=0.25, yellow=7.0)   # from recording descriptions (midpoint / representative)
status = {}
def save(fig, grp, num, slug, note=""):
    f = OUT / DIRS[grp] / f"{num:02d}_{slug}.png"; fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); status[num] = ("produced" + (f" (adapted: {note})" if note else ""), str(f.relative_to(R / 'results')))

# ---------- load artifacts ----------
folds, A = [], {}
for d in sorted(ART.glob("A_*")):
    if (d / "predictions.npz").exists(): z = np.load(d / "predictions.npz", allow_pickle=True); A[d.name[2:]] = {k: z[k] for k in z.files}; folds.append(d.name[2:])
print("folds with artifacts:", folds)
MODELS = ["PCT-E (ensemble x3)", "PCT-v2 (single)", "GBM", "LSTM", "GRU", "TCN", "Transformer", "Persistence"]
LBL = {"PCT-E (ensemble x3)": "PCT-E (ours)"}; lab = lambda m: LBL.get(m, m)
COL = 2   # gain @ 1 s
def _d(a, f): return a if "Yte" in a else a[f]
def phys(a, f, m, col=COL): d = _d(a, f); return d[f"te__{m}"][:, col] * d["Ste"][:, col]
def truth(a, f, col=COL): d = _d(a, f); return d["Yte"][:, col] * d["Ste"][:, col]

def ens_gauss(f):
    a = A[f]; mt, lv = a["member_te"], a["member_lv"]
    var = np.exp(lv).mean(0) + mt.var(0); return mt.mean(0), np.sqrt(var)            # normalised units
def gauss_for(f, m):
    """(mean, sd) in normalised units. PCT models: predicted; others: constant sd = std of residuals on the VALIDATION site."""
    a = A[f]
    if m == "PCT-E (ensemble x3)": return ens_gauss(f)
    if m == "PCT-v2 (single)": return a["member_te"][0], np.exp(0.5 * a["member_lv"][0])
    mu = a[f"te__{m}"]; sd = np.std(a["Yva"] - a[f"va__{m}"], axis=0) + 1e-6
    return mu, np.tile(sd, (len(mu), 1))

if folds:
    nf = len(folds)
    # ---- A1: actual vs predicted (acoustic channel gain, +1 s) ----
    fig, ax = plt.subplots(1, nf, figsize=(3.1 * nf, 3.3), squeeze=False)
    for i, f in enumerate(folds):
        a = A[f]; act = a["last"][:, 0] + truth(a, f); pr = a["last"][:, 0] + phys(a, f, "PCT-E (ensemble x3)")
        ax[0, i].scatter(act, pr, s=3, alpha=.35, color=OURS); lo, hi = min(act.min(), pr.min()), max(act.max(), pr.max()); ax[0, i].plot([lo, hi], [lo, hi], "k--", lw=.8)
        ax[0, i].set_title(f"{f} (held-out site)\nMAE(Δ)={np.abs(phys(a,f,'PCT-E (ensemble x3)')-truth(a,f)).mean():.3f} dB"); ax[0, i].set_xlabel("actual gain at +1 s (dB)"); ax[0, 0].set_ylabel("predicted (dB)")
    fig.suptitle("Actual vs predicted acoustic channel gain, +1 s (PCT-E; gain = SNR proxy, noise floor not in dataset)", y=1.03); save(fig, "A", 1, "actual_vs_pred_acoustic_gain", "channel gain dB instead of SNR")
    # ---- A1b time series + delay spread ----
    for col, nm, num_s in ((2, "gain (dB)", "01b_timeseries_gain"), (3, "rms delay spread (ms)", "01c_timeseries_delay")):
        fig, ax = plt.subplots(nf, 1, figsize=(8, 1.9 * nf), squeeze=False)
        for i, f in enumerate(folds):
            a = A[f]; sel = (a["rec"] == 0) & (a["r"] == 0); t = a["t"][sel] / D.RATE; o = np.argsort(t)
            ax[i, 0].plot(t[o], (a["last"][sel, col // 2] + truth(a, f, col)[sel])[o], color="k", lw=1, label="actual")
            ax[i, 0].plot(t[o], (a["last"][sel, col // 2] + phys(a, f, "PCT-E (ensemble x3)", col)[sel])[o], color=OURS, lw=1, label="PCT-E")
            ax[i, 0].set_ylabel(f"{f}\n{nm}"); 
        ax[0, 0].legend(ncol=2, frameon=False); ax[-1, 0].set_xlabel("time in recording (s); first recording, receiver 0")
        fig.savefig(OUT / DIRS["A"] / f"{num_s}.png", dpi=150); plt.close(fig)
    # ---- A6 error vs time ----
    fig, ax = plt.subplots(nf, 1, figsize=(8, 1.9 * nf), squeeze=False)
    for i, f in enumerate(folds):
        a = A[f]; sel = (a["rec"] == 0); t = a["t"][sel] / D.RATE
        for m in ("PCT-E (ensemble x3)", "Persistence", "GBM"):
            e = np.abs(phys(a, f, m) - truth(a, f))[sel]; o = np.argsort(t); es = pd.Series(e[o]).rolling(60, min_periods=10).mean()
            ax[i, 0].plot(t[o], es, color=PAL[lab(m)], lw=1.1, label=lab(m))
        ax[i, 0].set_ylabel(f"{f}\n|err| (dB)")
    ax[0, 0].legend(ncol=3, frameon=False); ax[-1, 0].set_xlabel("time in recording (s); rolling mean, all receivers of first recording"); save(fig, "A", 6, "error_vs_time")
    # ---- A7 error vs distance (site-level nominal range) ----
    rows = []
    for f in folds:
        for m in ("PCT-E (ensemble x3)", "Persistence", "GBM", "LSTM"): rows.append(dict(site=f, km=NOM_RANGE_KM[f], model=lab(m), mae=np.abs(phys(A, f, m) - truth(A, f)).mean() if False else np.abs(phys(A[f] and A, f, m) - truth(A, f)).mean()))
    d = pd.DataFrame(rows); fig, ax = plt.subplots(figsize=(5.2, 3.5))
    for m, g in d.groupby("model"): g = g.sort_values("km"); ax.plot(g.km, g.mae, "o-", color=PAL[m], label=m, lw=1.2)
    for _, r in d[d.model == "PCT-E (ours)"].iterrows(): ax.annotate(r.site, (r.km, r.mae), textcoords="offset points", xytext=(3, 4), fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("nominal link range of held-out site (km, from recording metadata)"); ax.set_ylabel("MAE of +1 s gain change (dB)")
    ax.set_title("Error vs distance (5 sites only – confounded with environment)"); ax.legend(frameon=False, fontsize=7); save(fig, "A", 7, "error_vs_distance", "site-level range, n=5 sites")
    # ---- A8 error vs channel condition ----
    pool = {m: np.concatenate([np.abs(phys(A, f, m) - truth(A, f)) for f in folds]) for m in ("PCT-E (ensemble x3)", "Persistence", "GBM", "LSTM", "Transformer")}
    vol = np.concatenate([A[f]["Ste"][:, 0] for f in folds]); dly = np.concatenate([A[f]["last"][:, 1] for f in folds])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    for k, (v, nm) in enumerate(((vol, "recent gain volatility (dB/step)"), (dly, "rms delay spread (ms)"))):
        q = pd.qcut(v, 4, duplicates="drop"); cats = q.categories
        for m, e in pool.items(): ax[k].plot(range(len(cats)), pd.Series(e).groupby(q.codes).mean().values, "o-", color=PAL[lab(m + (" (ensemble x3)" if False else ""))] if lab(m) in PAL else GREY, label=lab(m))
        ax[k].set_xticks(range(len(cats))); ax[k].set_xticklabels([f"Q{i+1}" for i in range(len(cats))]); ax[k].set_xlabel(f"{nm}, quartile (pooled windows)"); ax[k].set_ylabel("MAE +1 s (dB)")
    ax[0].legend(frameon=False, fontsize=7); fig.suptitle("Error vs channel condition (quartiles of window volatility / delay spread)", y=1.02); save(fig, "A", 8, "error_vs_channel_condition")
    # ---- B: uncertainty ----
    UM = ["PCT-E (ensemble x3)", "PCT-v2 (single)", "GBM", "LSTM", "GRU", "TCN", "Transformer", "Persistence"]
    def pooled(m, col=COL):
        mu, sd, y, S_ = [], [], [], []
        for f in folds:
            a = A[f]; mm, ss = gauss_for(f, m); mu.append(mm[:, col]); sd.append(ss[:, col]); y.append(a["Yte"][:, col]); S_.append(a["Ste"][:, col])
        return [np.concatenate(z) for z in (mu, sd, y, S_)]
    # 11 interval over time
    fig, ax = plt.subplots(nf, 1, figsize=(8, 2.0 * nf), squeeze=False)
    for i, f in enumerate(folds):
        a = A[f]; mu, sd = ens_gauss(f); sel = (a["rec"] == 0) & (a["r"] == 0); t = a["t"][sel] / D.RATE; o = np.argsort(t); S_ = a["Ste"][sel, COL]
        y = (a["Yte"][sel, COL] * S_)[o]; m_ = (mu[sel, COL] * S_)[o]; s_ = (sd[sel, COL] * S_)[o]
        ax[i, 0].fill_between(t[o], m_ - 1.645 * s_, m_ + 1.645 * s_, color=OURS, alpha=.25, label="90% interval"); ax[i, 0].plot(t[o], m_, color=OURS, lw=1, label="PCT-E mean"); ax[i, 0].plot(t[o], y, "k", lw=.8, label="actual")
        ax[i, 0].set_ylabel(f"{f}\nΔgain +1 s (dB)")
    ax[0, 0].legend(ncol=3, frameon=False); ax[-1, 0].set_xlabel("time (s), receiver 0 of first test recording"); save(fig, "B", 11, "prediction_interval_over_time")
    # 12 reliability (PIT) / 13 coverage vs nominal
    lv_ = np.linspace(.05, .95, 19); fig, ax = plt.subplots(figsize=(4.4, 4.2)); fig2, ax2 = plt.subplots(figsize=(4.4, 4.2))
    for m in UM:
        mu, sd, y, S_ = pooled(m); z = (y - mu) / sd; pit = norm.cdf(z)
        ax.plot(lv_, [(pit <= p).mean() for p in lv_], color=PAL[lab(m)], label=lab(m), lw=1.4 if "PCT" in m else 1)
        ax2.plot(lv_, [(np.abs(z) <= norm.ppf(.5 + p / 2)).mean() for p in lv_], color=PAL[lab(m)], label=lab(m), lw=1.4 if "PCT" in m else 1)
    for a_, t_ in ((ax, "Reliability diagram (PIT / quantile calibration)"), (ax2, "Central-interval coverage vs nominal")):
        a_.plot([0, 1], [0, 1], "k--", lw=.8); a_.set_xlabel("nominal probability"); a_.set_ylabel("empirical frequency"); a_.set_title(t_ + "\n(+1 s gain, pooled held-out sites)", fontsize=8); a_.legend(frameon=False, fontsize=6)
    save(fig, "B", 12, "reliability_diagram"); save(fig2, "B", 13, "coverage_vs_nominal")
    # 14 NLL / 15 CRPS / 16 sharpness vs coverage
    res = []
    for m in UM:
        for col, hz in ((2, "1 s"), (4, "2 s")):
            mu, sd, y, S_ = pooled(m, col); z = (y - mu) / sd; sdp = sd * S_
            nll = np.mean(0.5 * np.log(2 * np.pi * sdp ** 2) + (z * sdp) ** 2 / (2 * sdp ** 2)); crps = np.mean(sdp * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi)))
            res.append(dict(model=lab(m), horizon=hz, NLL=nll, CRPS=crps, cov90=(np.abs(z) <= 1.645).mean(), width90=(2 * 1.645 * sdp).mean()))
    U = pd.DataFrame(res); U.to_csv(R / "results/uncertainty_metrics.csv", index=False)
    for num, key, slug in ((14, "NLL", "nll_comparison"), (15, "CRPS", "crps_comparison")):
        fig, ax = plt.subplots(figsize=(6, 3.4)); p = U.pivot(index="model", columns="horizon", values=key).loc[[lab(m) for m in UM]]
        p.plot.bar(ax=ax, color=["#8aa4c0", OURS], width=.75); ax.set_ylabel(f"{key} (physical units, lower is better)"); ax.set_xlabel(""); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(f"{key}: non-PCT models use a constant Gaussian σ fitted on the validation site"); save(fig, "B", num, slug)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for m in UM:
        r = U[(U.model == lab(m)) & (U.horizon == "1 s")].iloc[0]; ax.scatter(r.width90, r.cov90, s=60, color=PAL[lab(m)], label=lab(m), zorder=3)
    ax.axhline(.9, color="k", ls="--", lw=.8); ax.set_xlabel("sharpness: mean width of 90% interval (dB, lower = sharper)"); ax.set_ylabel("empirical coverage of 90% interval"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("Sharpness vs coverage (+1 s gain; ideal = on dashed line, far left)"); save(fig, "B", 16, "sharpness_vs_coverage")
    # 17 OOD vs ID uncertainty
    bart = ART / "B_within" / "predictions.npz"
    if bart.exists():
        zb = np.load(bart); mt, lv = zb["member_te"], zb["member_lv"]; sd_id = np.sqrt(np.exp(lv).mean(0) + mt.var(0))[:, COL]
        fig, ax = plt.subplots(figsize=(5.6, 3.4)); data = [sd_id] + [ens_gauss(f)[1][:, COL] for f in folds]
        ax.boxplot(data, showfliers=False, patch_artist=True, boxprops=dict(facecolor="#e8c9bb")); ax.set_xticklabels(["ID: within-site\n(Protocol B)"] + [f"OOD: {f}" for f in folds], rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("predicted σ (normalised units)"); ax.set_title("Predicted uncertainty: in-distribution vs unseen-site (OOD)"); save(fig, "B", 17, "ood_vs_id_uncertainty")
    # ---- A9/A10 MAE / RMSE across models; E39-42 ----
    mA, fA = R / "results/protocolA_main.csv", R / "results/protocolA_final.csv"
    if mA.exists() and fA.exists():
        dm = pd.concat([pd.read_csv(f) for f in sorted(R.glob("results/protocolA_main*.csv"))] + [pd.read_csv(fA)]); dm["model"] = dm.model.replace({"PCT (proposed)": "PCT (single, base)", "PCT-E (ensemble x3)": "PCT-E (ours)"})
        fm = dm.groupby(["model", "fold"]).mean(numeric_only=True).reset_index()
        for num, kind in ((9, "MAE"), (10, "RMSE")):
            cols = [f"{kind}_gain_0.5s", f"{kind}_gain_1.0s", f"{kind}_gain_2.0s"]; g = fm.groupby("model")[cols].agg(["mean", "std"]); order = g[(cols[1], "mean")].sort_values().index
            fig, ax = plt.subplots(figsize=(9, 3.8)); w = .27
            for k, c in enumerate(cols): ax.bar(np.arange(len(order)) + (k - 1) * w, g.loc[order, (c, "mean")], w, yerr=g.loc[order, (c, "std")], capsize=2, label=c.split("_")[-1], color=["#9db7d1", "#5d89b4", "#2b5a8a"][k])
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right"); ax.set_ylabel(f"{kind} (dB), mean ± std over sites"); ax.legend(title="horizon", frameon=False); ax.set_title(f"{kind} comparison across models (Protocol A, leave-one-site-out)")
            save(fig, "A", num, f"{kind.lower()}_comparison_across_models")
        for num, base in ((39, "LSTM"), (40, "GRU"), (41, "TCN"), (42, "Transformer")):
            fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.3))
            for k, c in enumerate(("MAE_gain_1.0s", "MAE_gain_2.0s")):
                p = fm[fm.model.isin(["PCT-E (ours)", base])].pivot(index="fold", columns="model", values=c); p.plot.bar(ax=ax[k], color=[OURS, GREY] if list(p.columns)[0].startswith("PCT") else [GREY, OURS], width=.75)
                ax[k].set_ylabel(c.replace("_", " ") + " (dB)"); ax[k].set_xlabel("held-out site"); ax[k].set_title(f"{c.split('_')[-1]}: mean PCT-E {p['PCT-E (ours)'].mean():.3f} vs {base} {p[base].mean():.3f} dB", fontsize=8)
            fig.suptitle(f"PCT-E (ours) vs {base} – per held-out site (3 seeds)", y=1.02); save(fig, "E", num, f"pct_vs_{base.lower()}", "our model is PCT-E, not PIGT-DT")
    # ---- F52 ensemble size ----
    fig, ax = plt.subplots(figsize=(4.8, 3.3)); allm = {k: [] for k in (1, 2, 3)}
    for f in folds:
        a = A[f]; S_ = a["Ste"][:, COL]; y = a["Yte"][:, COL] * S_
        for k in (1, 2, 3): allm[k].append(np.abs(a["member_te"][:k].mean(0)[:, COL] * S_ - y).mean())
    arr = np.array([allm[k] for k in (1, 2, 3)]); ax.errorbar([1, 2, 3], arr.mean(1), yerr=arr.std(1), fmt="o-", color=OURS, capsize=3); ax.set_xticks([1, 2, 3])
    ax.set_xlabel("ensemble size (members)"); ax.set_ylabel("MAE +1 s gain (dB); mean ± std over sites"); ax.set_title("Effect of ensemble size (seed 0)"); save(fig, "F", 52, "ensemble_size")

# ---- F47/48/50 ablations (Protocol B) ----
ab = R / "results/protocolB_ablation.csv"
if ab.exists():
    d = pd.read_csv(ab); g = d.groupby("model")[["MAE_gain_1.0s", "MAE_gain_2.0s", "MAE_delay_1.0s"]].agg(["mean", "std"])
    for num, var, slug in ((47, "- AR prior", "remove_physics_prior"), (48, "- cross-receiver ctx", "remove_cross_receiver_context"), (50, "- heteroscedastic head", "remove_uncertainty_head")):
        if var not in g.index: continue
        fig, ax = plt.subplots(figsize=(5.4, 3.3)); cols = ["MAE_gain_1.0s", "MAE_gain_2.0s", "MAE_delay_1.0s"]; w = .35
        for k, m in enumerate(("PCT full", var)): ax.bar(np.arange(3) + (k - .5) * w, [g.loc[m, (c, "mean")] for c in cols], w, yerr=[g.loc[m, (c, "std")] for c in cols], capsize=2, color=[OURS, GREY][k], label=m)
        ax.set_xticks(range(3)); ax.set_xticklabels(["gain +1 s", "gain +2 s", "delay +1 s"]); ax.set_ylabel("MAE (dB / ms), mean ± std over 3 seeds"); ax.legend(frameon=False); ax.set_title(f"Ablation (Protocol B, chronological split): {var}", fontsize=9)
        save(fig, "F", num, slug, {47: "AR/linear-extrapolation prior is the 'physics' prior here", 48: "cross-receiver context stands in for graph structure", 50: "heteroscedastic head"}[num])
    fig, ax = plt.subplots(figsize=(7, 4)); o = g[("MAE_gain_1.0s", "mean")].sort_values()
    ax.barh(range(len(o)), o.values, xerr=g.loc[o.index, ("MAE_gain_1.0s", "std")], color=[OURS if m == "PCT full" else GREY for m in o.index], capsize=2); ax.set_yticks(range(len(o))); ax.set_yticklabels(o.index); ax.invert_yaxis()
    ax.set_xlabel("MAE gain +1 s (dB), mean ± std over 3 seeds"); ax.set_title("All ablation variants (Protocol B)"); fig.savefig(OUT / DIRS["F"] / "00_all_ablation_variants.png", dpi=150); plt.close(fig)

# ---- H robustness ----
rb = R / "results/robustness.csv"
if rb.exists():
    d = pd.read_csv(rb); xl = dict(dropout="sensor dropout probability per time step", noise="observation noise σ (× window volatility)", stale="DT staleness: newest k seconds missing (s)")
    for num, kind, slug in ((63, "dropout", "sensor_dropout"), (66, "noise", "noisy_observations"), (67, "stale", "stale_dt")):
        g = d[d.kind == kind].groupby(["model", "level"]).MAE_g1.mean().reset_index(); fig, ax = plt.subplots(figsize=(5.4, 3.5))
        for m, gg in g.groupby("model"): ax.plot(gg.level, gg.MAE_g1, "o-", color=PAL.get(m, GREY), label=m, lw=1.6 if "ours" in m else 1)
        ax.set_xlabel(xl[kind]); ax.set_ylabel("MAE +1 s gain (dB), mean over folds"); ax.legend(frameon=False, fontsize=6); ax.set_title(f"Robustness: {slug.replace('_',' ')}"); save(fig, "H", num, slug, "'stale DT' = stale observations" if kind == "stale" else "")
mA = R / "results/protocolA_final.csv"
if mA.exists() and (R / "results/protocolA_main.csv").exists():
    dm = pd.concat([pd.read_csv(f) for f in sorted(R.glob("results/protocolA_main*.csv"))] + [pd.read_csv(mA)]); fm = dm.groupby(["model", "fold"])["MAE_gain_1.0s"].mean().unstack(0)
    if "PCT-E (ensemble x3)" in fm:
        fig, ax = plt.subplots(figsize=(5.4, 3.3)); rel = 100 * (fm["Persistence"] - fm["PCT-E (ensemble x3)"]) / fm["Persistence"]
        ax.bar(rel.index, rel.values, color=[OURS if v > 0 else GREY for v in rel.values]); ax.axhline(0, color="k", lw=.8); ax.set_ylabel("MAE reduction vs persistence (%)"); ax.set_xlabel("held-out (unseen) site")
        ax.set_title("Unseen environment / acoustic condition: PCT-E skill vs persistence"); save(fig, "H", 65, "unseen_environment", "unseen site = unseen environment"); status[70] = ("produced (same figure as 65)", status[65][1])
# ---- optical (real measured BER) ----
op = R / "data/processed/optical_ber_table.csv"; opp = R / "results/optical_predictions.csv"
if op.exists():
    t = pd.read_csv(op)
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for (med, b), g in t[t.rate == 2.0].groupby(["medium", "bits"]): ax.plot(g.dist, np.maximum(g.ber, 1e-5), "o-", ls="-" if med == "clean" else "--", label=f"{med}, {2**b}-QAM")
    ax.set_yscale("log"); ax.set_xlabel("distance (cm)"); ax.set_ylabel("measured BER (floor 1e-5)"); ax.legend(frameon=False, fontsize=7); ax.set_title("Optical BER vs distance @ IQ rate 2.0 MS/s (measured, Dratnal et al.)"); save(fig, "C", 19, "optical_ber_vs_distance_measured", "BER instead of PDR (PDR not measured)")
    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    for (med, b), g in t[(t.dist == 30)].groupby(["medium", "bits"]): ax.plot(g.rate, np.maximum(g.ber, 1e-5), "o-", ls="-" if med == "clean" else "--", label=f"{med}, {2**b}-QAM")
    ax.set_yscale("log"); ax.set_xlabel("IQ rate (MS/s)"); ax.set_ylabel("measured BER (floor 1e-5)"); ax.legend(frameon=False, fontsize=7); ax.set_title("Optical BER vs IQ rate @ 30 cm (measured; SNR not recorded)"); save(fig, "C", 22, "optical_ber_vs_rate_measured", "BER vs IQ rate instead of vs SNR")
    r6 = R / "data/raw/ofdm_uwvc/Dataset_6_transmission_rate_comparison.csv"
    if r6.exists():
        q = pd.read_csv(r6, sep=";", encoding="utf-8-sig", dtype=str).map(lambda v: float(str(v).replace(",", ".").rstrip("k")) if v else v); fig, ax = plt.subplots(figsize=(5.6, 3.5))
        for c in q.columns[1:]: ax.plot(q["IQ rate"] / 1e3 if q["IQ rate"].max() > 1e4 else q["IQ rate"], q[c], "o-", label=c, lw=1)
        ax.set_xlabel("IQ rate (k)"); ax.set_ylabel("measured throughput (Dataset_6 units: Mbit/s)"); ax.legend(frameon=False, fontsize=6, ncol=2); ax.set_title("Optical throughput vs IQ rate by modulation (measured)"); save(fig, "C", 23, "optical_throughput_vs_rate_measured", "throughput vs IQ rate, not vs distance")
if opp.exists():
    p = pd.read_csv(opp); g = p[(p.model == "GBM") & (p.split == "LODO")]
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6)); ax[0].scatter(g.y_true, g.y_pred, s=8, alpha=.5, color=OURS); ax[0].plot([-5, -.5], [-5, -.5], "k--", lw=.8)
    ax[0].set_xlabel("measured log10 BER"); ax[0].set_ylabel("predicted log10 BER (held-out distance)"); ax[0].set_title("Actual vs predicted optical BER (leave-one-distance-out, GBM)", fontsize=8)
    e = g.assign(err=(g.y_pred - g.y_true).abs()).groupby("dist").err.mean(); ax[1].bar(e.index, e.values, width=3, color=OURS); ax[1].set_xlabel("held-out distance (cm)"); ax[1].set_ylabel("MAE of log10 BER")
    ax[1].set_title("Prediction error vs distance (optical)", fontsize=8); save(fig, "A", 3, "actual_vs_pred_optical_ber", "optical BER, log10"); status[68] = ("produced (optical only; right panel of figure 3)", status[3][1])
    fig, ax = plt.subplots(figsize=(4.8, 3.3)); s = p[p.split == "LOMO"].assign(err=lambda x: (x.y_pred - x.y_true).abs()).groupby(["model", "held_out"]).err.mean().unstack(); s.plot.bar(ax=ax, width=.75); ax.set_ylabel("MAE of log10 BER")
    ax.set_xlabel("model (columns: held-out medium)"); ax.set_title("Unseen medium (clean↔pump-circulated murky water)", fontsize=9); plt.setp(ax.get_xticklabels(), rotation=0); save(fig, "H", 69, "unseen_medium_optical", "unseen water medium, optical BER")
sc = R / "results/scalability.csv"
if sc.exists():
    d = pd.read_csv(sc)
    for num, col, yl, slug in ((57, "infer_ms_total", "inference latency, all nodes (ms, CPU)", "inference_latency_vs_nodes"), (58, "act_alloc_MB", "tensor memory allocated per forward pass (MB, CPU)", "memory_vs_nodes"), (59, "train_s_per_epoch", "training time per epoch (s, CPU)", "training_time_vs_nodes")):
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        for m, g in d.groupby("model"): ax.plot(g.nodes, g[col], "o-", label=m, lw=1.6 if m == "PCT" else 1, color=OURS if m == "PCT" else None)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("number of nodes N (one forecast per node)"); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=7); ax.set_title("Compute scaling (timing only; random inputs)"); save(fig, "G", num, slug, "CPU only, no GPU; per-node forecasters")

# ---------- INDEX ----------
REQ = {1: "Actual vs predicted acoustic SNR", 2: "Actual vs predicted optical SNR", 3: "Actual vs predicted BER", 4: "Actual vs predicted PDR", 5: "Actual vs predicted throughput", 6: "Prediction error vs time", 7: "Prediction error vs distance", 8: "Prediction error vs channel condition", 9: "MAE comparison across models", 10: "RMSE comparison across models",
 11: "Prediction interval / uncertainty over time", 12: "Calibration/reliability diagram", 13: "Coverage vs nominal confidence", 14: "NLL comparison", 15: "CRPS comparison", 16: "Sharpness vs coverage", 17: "OOD uncertainty vs ID uncertainty",
 18: "Acoustic PDR vs distance", 19: "Optical PDR vs distance", 20: "Hybrid PDR vs distance", 21: "Acoustic BER vs SNR", 22: "Optical BER vs SNR", 23: "Throughput vs distance", 24: "Energy vs distance", 25: "Latency vs distance", 26: "AC/OP/HYB mode-selection probability",
 27: "Reward vs training episode", 28: "PDR vs training episode", 29: "Throughput vs training episode", 30: "Energy vs training episode", 31: "BER vs training episode", 32: "Latency vs training episode", 33: "Constraint violations vs episode", 34: "Cumulative energy", 35: "Cumulative throughput", 36: "Mode switching over time",
 37: "PIGT-DT vs Physics-only", 38: "PIGT-DT vs Data-only", 39: "vs LSTM", 40: "vs GRU", 41: "vs TCN", 42: "vs Transformer", 43: "vs GNN", 44: "vs MAPPO", 45: "vs heuristic controller", 46: "vs oracle",
 47: "Remove physics prior", 48: "Remove graph structure", 49: "Remove temporal module", 50: "Remove uncertainty", 51: "Remove risk constraints", 52: "Ensemble size", 53: "Real/DT data ratio", 54: "Model mismatch", 55: "DT staleness", 56: "Node failure",
 57: "Inference latency vs nodes", 58: "CPU/GPU memory vs nodes", 59: "Training time vs nodes", 60: "Throughput vs nodes", 61: "PDR vs nodes", 62: "Energy vs nodes",
 63: "Sensor dropout", 64: "Channel-model mismatch", 65: "Environmental mismatch", 66: "Noisy observations", 67: "Stale DT", 68: "Unseen distances", 69: "Unseen turbidity", 70: "Unseen acoustic conditions"}
WHY = {2: "no optical SNR in the public optical dataset (BER only)", 4: "PDR/packet outcomes are not in either dataset", 5: "throughput is not measured over time/distance in the acoustic repository", 18: "no PDR data", 19: "no PDR data (see 19 adaptation: measured BER)", 20: "no hybrid link data", 21: "no acoustic BER/SNR measurements (channel impulse responses only)", 22: "optical dataset has no SNR",
 24: "no energy measurements", 25: "no latency measurements", 26: "no mode-selection experiment on real data", **{n: "no RL/controller was trained: requires a link simulator, which would be synthetic" for n in (27, 28, 29, 30, 31, 32, 33, 34, 35, 36)},
 37: "no physics-only (BELLHOP/optical-physics) model implemented; the 'AR prior' ablation (47) is the closest", 38: "no separate data-only variant of this model; see ablations", 43: "no graph model implemented (cross-receiver context ablation is 48)", 44: "no MAPPO/controller", 45: "no controller", 46: "no controller/oracle policy",
 49: "temporal-module ablation was not run", 51: "no risk constraints in this forecaster", 53: "no digital-twin/simulated data in this project", 54: "no simulator to mismatch (cross-site tests are in 65)", 55: "see 67 (stale observations)", 56: "node-failure experiment not run (sensor dropout in 63)",
 60: "no link-level throughput model", 61: "no PDR data", 62: "no energy data", 64: "no channel-model simulator", 70: "see 65", 8: "", 1: "", 3: ""}
lines = ["# (superseded by 16_sim_plots.py) Figure index (70 requested)", "", "Status per requested figure. **Produced** = computed from real measured data / trained models in this repo. **Not produced** = the real datasets cannot support it (no fabrication). Titles marked *adapted* differ from the request as stated.", "",
         "| # | Requested | Status | File / reason |", "|---|---|---|---|"]
for n in range(1, 71):
    if n in status: lines.append(f"| {n} | {REQ[n]} | {status[n][0]} | `{status[n][1]}` |")
    else: lines.append(f"| {n} | {REQ[n]} | **not produced** | {WHY.get(n, 'prerequisite results missing at generation time')} |")
json.dump({str(k): v for k, v in status.items()}, open(OUT / "status_real.json", "w")); print(f"produced {len(status)} / 70 (real-data figures)")

