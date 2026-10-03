"""Figures that need link-level quantities (SNR, BER, PDR, throughput, energy, latency, controllers, scalability, robustness).
ALL are from the SIMULATED hybrid link (real measured acoustic gain dynamics + assumed link budget) -> titles carry '[SIMULATION]'. Also writes the combined INDEX.md."""
import sys, json, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import norm
R = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(R / "src"))
from uwfc import simexp as X, dt as DT, linksim as S, ctrl as C
OUT = R / "results/plots"; SIM = X.SIM; P = X.P
DIRS = dict(A="A_prediction", B="B_uncertainty", C="C_link_optical", D="D_controller", E="E_baselines", F="F_ablation", G="G_scalability", H="H_robustness")
for d in DIRS.values(): (OUT / d).mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "font.size": 9, "axes.titlesize": 9.5, "figure.autolayout": True})
OURS = "#c4562d"; TAG = "[SIMULATION] "
CC = {"PIGT-DT": OURS, "MAPPO": "#2f6f9f", "MAPPO+DT": "#3a9fa3", "Reactive": "#8a8f98", "Heuristic": "#d4a73a", "Oracle": "#222222", "AC-only": "#a67bb5", "OP-only": "#6a9f58", "HYB-low": "#7f6a4f"}
simstatus = {}
def save(fig, grp, num, slug, note=""):
    f = OUT / DIRS[grp] / f"{num:02d}_sim_{slug}.png"; fig.savefig(f, dpi=150, bbox_inches="tight"); plt.close(fig); simstatus.setdefault(num, []).append(("simulated" + (f" ({note})" if note else ""), str(f.relative_to(R / "results"))))
def have(*fs): return all((SIM / f).exists() for f in fs)

# ---------------------------------------------------------------- DT forecast figures (A, B, E)
if have("dt_preds_main.npz", "dt_metrics_main.csv"):
    Z = np.load(SIM / "dt_preds_main.npz"); D = X.datasets(64); M = pd.read_csv(SIM / "dt_metrics_main.csv"); ENS = "PIGT-DT ensemble x3"
    def getp(ts, lab): return Z[f"{ts}__{lab}__mu"], (Z[f"{ts}__{lab}__sd"] if Z[f"{ts}__{lab}__sd"].size else None)
    def scat(ax, t, pr, title, lo=None, hi=None, log=False, mask=None):
        if mask is not None: t, pr = t[mask], pr[mask]
        i = np.random.default_rng(0).choice(len(t), min(4000, len(t)), replace=False); ax.scatter(t[i], pr[i], s=2, alpha=.3, color=OURS)
        a_, b_ = (min(t.min(), pr.min()), max(t.max(), pr.max())) if lo is None else (lo, hi); ax.plot([a_, b_], [a_, b_], "k--", lw=.8); ax.set_title(title)
        if log: ax.set_xscale("log"); ax.set_yscale("log")
    for num, slug, what in ((1, "actual_vs_pred_acoustic_snr", "ac"), (2, "actual_vs_pred_optical_snr", "op"), (3, "actual_vs_pred_ber", "ber"), (4, "actual_vs_pred_pdr", "pdr"), (5, "actual_vs_pred_throughput", "thr")):
        fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.6))
        for k, ts in enumerate(("test_id", "test_ood")):
            d = D[ts]; mu, sd = getp(ts, ENS); tg = d["tgt"]; act = d["near"] & (tg[..., 1] > -29); dp, dtr = DT.derived(mu[..., 0], mu[..., 1]), DT.derived(tg[..., 0], tg[..., 1])
            if what == "ac": scat(ax[k], tg[..., 0].ravel(), mu[..., 0].ravel(), f"{ts}: acoustic SNR (dB), decision step"); ax[k].set_xlabel("true"); ax[k].set_ylabel("PIGT-DT predicted")
            elif what == "op":
                if act.sum() > 10: scat(ax[k], tg[..., 1][act], mu[..., 1][act], f"{ts}: optical SNR (dB), active links"); ax[k].set_xlabel("true"); ax[k].set_ylabel("predicted")
                else: ax[k].text(.5, .5, "optical link inactive\n(SNR < -29 dB everywhere)", ha="center", transform=ax[k].transAxes); ax[k].set_title(ts)
            elif what == "ber":
                scat(ax[k], dtr["ber_ac"].ravel() + 1e-9, dp["ber_ac"].ravel() + 1e-9, f"{ts}: acoustic BER", lo=1e-9, hi=.5, log=True); ax[k].set_xlabel("true"); ax[k].set_ylabel("predicted")
            elif what == "pdr": scat(ax[k], dtr["pdr_ac"].ravel(), dp["pdr_ac"].ravel(), f"{ts}: acoustic PDR", lo=0, hi=1); ax[k].set_xlabel("true"); ax[k].set_ylabel("predicted")
            else:
                if act.sum() > 10: scat(ax[k], dtr["thr_op"][act], dp["thr_op"][act], f"{ts}: optical goodput (kbit/s)", mask=None); ax[k].set_xlabel("true"); ax[k].set_ylabel("predicted")
                else: scat(ax[k], dtr["thr_ac"].ravel(), dp["thr_ac"].ravel(), f"{ts}: acoustic goodput (kbit/s)"); ax[k].set_xlabel("true"); ax[k].set_ylabel("predicted")
        fig.suptitle(TAG + "PIGT-DT (3-member ensemble), forecast for the decision step from 1 s-stale observations", y=1.02, fontsize=9); save(fig, "A", num, slug, "BER panel is acoustic" if what == "ber" else "")
    d = D["test_id"]; mu, _ = getp("test_id", ENS); tg = d["tgt"]; B_ = 48; Tn = len(tg) // B_
    e_ac = np.abs(mu[..., 0] - tg[..., 0]).reshape(Tn, B_, 8); p_ac = np.abs(d["persist"][..., 0] - tg[..., 0]).reshape(Tn, B_, 8)
    act = (d["near"] & (tg[..., 1] > -29)).reshape(Tn, B_, 8); e_op = np.where(act, np.abs(mu[..., 1] - tg[..., 1]).reshape(Tn, B_, 8), np.nan); p_op = np.where(act, np.abs(d["persist"][..., 1] - tg[..., 1]).reshape(Tn, B_, 8), np.nan)
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2)); tt = np.arange(Tn) * S.STEP_S
    ax[0].plot(tt, e_ac.mean((1, 2)), color=OURS, label="PIGT-DT"); ax[0].plot(tt, p_ac.mean((1, 2)), color="grey", label="persistence"); ax[1].plot(tt, np.nanmean(e_op, (1, 2)), color=OURS); ax[1].plot(tt, np.nanmean(p_op, (1, 2)), color="grey")
    ax[0].set_title("acoustic SNR |error| (dB)"); ax[1].set_title("optical SNR |error| (dB), active links"); [a_.set_xlabel("time in episode (s)") for a_ in ax]; ax[0].legend(frameon=False); fig.suptitle(TAG + "Prediction error vs time (test_id)", y=1.02); save(fig, "A", 6, "error_vs_time")
    dist = d["d_tgt"][..., 0]; fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2)); bins = np.exp(np.linspace(np.log(3), np.log(2500), 9))
    for k, (e, pe, lab) in enumerate(((np.abs(mu[..., 0] - tg[..., 0]), np.abs(d["persist"][..., 0] - tg[..., 0]), "acoustic"), (np.abs(mu[..., 1] - tg[..., 1]), np.abs(d["persist"][..., 1] - tg[..., 1]), "optical"))):
        m = np.ones_like(e, bool) if k == 0 else (d["near"] & (tg[..., 1] > -29)); idx = np.digitize(dist, bins)
        for arr, c, l in ((e, OURS, "PIGT-DT"), (pe, "grey", "persistence")): ax[k].plot([np.sqrt(bins[i - 1] * bins[i]) for i in range(1, len(bins)) if (m & (idx == i)).sum() > 20], [arr[m & (idx == i)].mean() for i in range(1, len(bins)) if (m & (idx == i)).sum() > 20], "o-", color=c, label=l)
        ax[k].set_xscale("log"); ax[k].set_xlabel("link distance (m)"); ax[k].set_ylabel("MAE (dB)"); ax[k].set_title(f"{lab} SNR")
    ax[0].legend(frameon=False); fig.suptitle(TAG + "Prediction error vs distance (test_id)", y=1.02); save(fig, "A", 7, "error_vs_distance")
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2)); sn = d["last_ac"].ravel(); q = pd.qcut(sn, 5, duplicates="drop"); ec = pd.Series(np.abs(mu[..., 0] - tg[..., 0]).ravel()).groupby(q.codes).mean(); pc = pd.Series(np.abs(d["persist"][..., 0] - tg[..., 0]).ravel()).groupby(q.codes).mean()
    ax[0].plot(ec.index + 1, ec.values, "o-", color=OURS, label="PIGT-DT"); ax[0].plot(pc.index + 1, pc.values, "o-", color="grey", label="persistence"); ax[0].set_xlabel("observed acoustic SNR quintile (Q1=poor link)"); ax[0].set_ylabel("MAE (dB)"); ax[0].legend(frameon=False)
    ch = d["X"][:, :, -1, 5]; m = d["near"] & (tg[..., 1] > -29); q2 = pd.qcut(ch[m], 4, duplicates="drop"); eo = pd.Series(np.abs(mu[..., 1] - tg[..., 1])[m]).groupby(q2.codes).mean(); po = pd.Series(np.abs(d["persist"][..., 1] - tg[..., 1])[m]).groupby(q2.codes).mean()
    ax[1].plot(eo.index + 1, eo.values, "o-", color=OURS); ax[1].plot(po.index + 1, po.values, "o-", color="grey"); ax[1].set_xlabel("estimated optical attenuation quartile (Q4=turbid)"); ax[1].set_ylabel("MAE (dB)")
    fig.suptitle(TAG + "Prediction error vs channel condition (test_id)", y=1.02); save(fig, "A", 8, "error_vs_channel_condition")
    S1 = M[(M.testset == "test_id") & (M.kind == "single")]; order = ["Physics-only", "Data-only", "LSTM", "GRU", "TCN", "Transformer", "GNN", "PIGT-DT"]
    for num, kind in ((9, "MAE"), (10, "RMSE")):
        cols = [f"{kind}_ac_h1", f"{kind}_ac_h2", f"{kind}_op_h1", f"{kind}_op_h2"]; g = S1.groupby("model")[cols].agg(["mean", "std"]); fig, ax = plt.subplots(figsize=(9, 3.6)); w = .2
        for k, c in enumerate(cols): ax.bar(np.arange(len(order)) + (k - 1.5) * w, [g.loc[m, (c, "mean")] if m in g.index else np.nan for m in order], w, yerr=[g.loc[m, (c, "std")] if m in g.index else 0 for m in order], capsize=2, label=c.replace(f"{kind}_", ""), color=["#9db7d1", "#5d89b4", "#e8b49c", OURS][k])
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20, ha="right"); ax.set_ylabel(f"{kind} (dB), mean ± std over 3 seeds"); ax.legend(title="target (h1=decision step, h2=+0.75 s)", frameon=False, fontsize=7); ax.set_title(TAG + f"{kind} of SNR forecast across digital-twin models (test_id)"); save(fig, "A", num, f"{kind.lower()}_across_models")
    # E37-43 PIGT-DT vs each DT baseline
    for num, base in ((37, "Physics-only"), (38, "Data-only"), (39, "LSTM"), (40, "GRU"), (41, "TCN"), (42, "Transformer"), (43, "GNN")):
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        for k, ts in enumerate(("test_id", "test_ood")):
            sub = M[(M.testset == ts) & (M.kind == "single") & (M.model.isin(["PIGT-DT", base]))]; cols = ["MAE_ac_h1", "MAE_ac_h2", "MAE_op_h1", "MAE_op_h2"]; g = sub.groupby("model")[cols].agg(["mean", "std"]); w = .35
            for j, m in enumerate(("PIGT-DT", base)): ax[k].bar(np.arange(4) + (j - .5) * w, [g.loc[m, (c, "mean")] if m in g.index else np.nan for c in cols], w, yerr=[g.loc[m, (c, "std")] if m in g.index and not np.isnan(g.loc[m, (c, "std")]) else 0 for c in cols], capsize=2, color=[OURS, "grey"][j], label=m)
            ax[k].set_xticks(range(4)); ax[k].set_xticklabels(["ac h1", "ac h2", "op h1", "op h2"]); ax[k].set_title(ts); ax[k].set_ylabel("MAE (dB)")
        ax[0].legend(frameon=False); fig.suptitle(TAG + f"PIGT-DT vs {base}: SNR forecast error (single models, 3 seeds)", y=1.02); save(fig, "E", num, f"pigtdt_vs_{base.lower().replace('-', '_')}")
    # ---- B: uncertainty
    ens_sd = getp("test_id", ENS)[1]; mu_e = getp("test_id", ENS)[0]; sdm = ens_sd
    z = (mu_e - tg) / np.maximum(sdm, 1e-3); sel = np.ones_like(z, bool); sel[..., 1] = d["near"] & (tg[..., 1] > -29); sel[..., 3] = d["near"] & (tg[..., 3] > -29)
    fig, ax = plt.subplots(figsize=(8, 3.2)); i0 = 0; e0 = 3; ts_ = np.arange(Tn) * S.STEP_S; sl = slice(e0 * 0, None)
    mm = mu_e[..., 0].reshape(Tn, B_, 8)[:, 0, 0]; ss = sdm[..., 0].reshape(Tn, B_, 8)[:, 0, 0]; tt_ = tg[..., 0].reshape(Tn, B_, 8)[:, 0, 0]
    ax.fill_between(ts_, mm - 1.645 * ss, mm + 1.645 * ss, color=OURS, alpha=.25, label="90% interval"); ax.plot(ts_, mm, color=OURS, label="mean"); ax.plot(ts_, tt_, "k", lw=.8, label="true"); ax.set_xlabel("time (s)"); ax.set_ylabel("acoustic SNR (dB)"); ax.legend(frameon=False, ncol=3); ax.set_title(TAG + "Prediction interval over time (one node, test_id)"); save(fig, "B", 11, "prediction_interval_over_time")
    lv = np.linspace(.05, .95, 19); fig, ax = plt.subplots(figsize=(4.4, 4.2)); fig2, ax2 = plt.subplots(figsize=(4.4, 4.2))
    cand = {"PIGT-DT ensemble x3": (mu_e, sdm)}
    for nm in ("PIGT-DT", ):
        mu1, sd1 = getp("test_id", "PIGT-DT"); cand["PIGT-DT single member"] = (mu1, sd1)
    for nm in ("LSTM", "TCN"):   # constant-sigma baselines (sigma from validation residuals)
        key = f"test_id__{nm}__mu"
        if key in Z.files:
            from uwfc import simexp as X_; m_ = X_.get_model(nm, 0, None, None); mv, _ = DT.predict(m_, D["val"], DT.SPECS[nm][1]); s_c = np.std(mv - D["val"]["tgt"], axis=(0, 1)); cand[f"{nm} (const. sigma)"] = (Z[key], np.broadcast_to(s_c, Z[key].shape))
    ucol = ["#c4562d", "#e59a7c", "#6a9f58", "#d4a73a"]
    for (nm, (m_, s_)), c in zip(cand.items(), ucol):
        zz = (m_ - tg) / np.maximum(s_, 1e-3); pit = norm.cdf(zz)[sel]; ax.plot(lv, [(pit <= q_).mean() for q_ in lv], color=c, label=nm); ax2.plot(lv, [(np.abs(zz[sel]) <= norm.ppf(.5 + q_ / 2)).mean() for q_ in lv], color=c, label=nm)
    for a_, t_ in ((ax, "Reliability diagram (PIT)"), (ax2, "Central-interval coverage vs nominal")): a_.plot([0, 1], [0, 1], "k--", lw=.8); a_.set_xlabel("nominal"); a_.set_ylabel("empirical"); a_.set_title(TAG + t_, fontsize=8); a_.legend(frameon=False, fontsize=6)
    save(fig, "B", 12, "reliability_diagram"); save(fig2, "B", 13, "coverage_vs_nominal")
    U = []
    for nm, (m_, s_) in cand.items():
        for ts, dd in (("test_id", D["test_id"]), ("test_ood", D["test_ood"])):
            if ts == "test_ood" and nm.endswith("(const. sigma)"): m2 = Z[f"test_ood__{nm.split(' ')[0]}__mu"]; s2 = s_[:len(m2)] if s_.shape[0] >= len(m2) else s_
            elif ts == "test_ood" and nm == "PIGT-DT ensemble x3": m2, s2 = getp("test_ood", ENS)
            elif ts == "test_ood": m2, s2 = getp("test_ood", "PIGT-DT")
            else: m2, s2 = m_, s_
            tg2 = dd["tgt"]; sel2 = np.ones_like(tg2, bool); sel2[..., 1] = dd["near"] & (tg2[..., 1] > -29); sel2[..., 3] = dd["near"] & (tg2[..., 3] > -29); s2 = np.maximum(s2, 1e-3); z2 = (m2 - tg2) / s2
            U.append(dict(model=nm, testset=ts, NLL=float((0.5 * np.log(2 * np.pi * s2 ** 2) + .5 * z2 ** 2)[sel2].mean()), CRPS=float((s2 * (z2 * (2 * norm.cdf(z2) - 1) + 2 * norm.pdf(z2) - 1 / np.sqrt(np.pi)))[sel2].mean()), cov90=float((np.abs(z2) <= 1.645)[sel2].mean()), width90=float((2 * 1.645 * s2)[sel2].mean()), mean_sigma=float(s2[sel2].mean())))
    U = pd.DataFrame(U); U.to_csv(SIM / "dt_uncertainty.csv", index=False)
    for num, key, slug in ((14, "NLL", "nll_comparison"), (15, "CRPS", "crps_comparison")):
        fig, ax = plt.subplots(figsize=(6, 3.3)); U.pivot(index="model", columns="testset", values=key).plot.bar(ax=ax, color=["#8aa4c0", OURS], width=.7); ax.set_ylabel(f"{key} (dB units, lower is better)"); plt.setp(ax.get_xticklabels(), rotation=15, ha="right"); ax.set_title(TAG + f"{key} comparison"); save(fig, "B", num, slug)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for nm, c in zip(U.model.unique(), ucol):
        for ts, mk in (("test_id", "o"), ("test_ood", "s")): r = U[(U.model == nm) & (U.testset == ts)].iloc[0]; ax.scatter(r.width90, r.cov90, s=55, marker=mk, color=c, label=f"{nm} ({ts})" if ts == "test_id" else None)
    ax.axhline(.9, color="k", ls="--", lw=.8); ax.set_xlabel("mean width of 90% interval (dB)"); ax.set_ylabel("empirical coverage"); ax.legend(frameon=False, fontsize=6); ax.set_title(TAG + "Sharpness vs coverage (circle=ID, square=OOD)"); save(fig, "B", 16, "sharpness_vs_coverage")
    fig, ax = plt.subplots(figsize=(4.8, 3.3)); ax.boxplot([getp("test_id", ENS)[1][..., 0].ravel(), getp("test_ood", ENS)[1][..., 0].ravel()], showfliers=False, patch_artist=True, boxprops=dict(facecolor="#e8c9bb")); ax.set_xticklabels(["ID", "OOD (unseen turbidity,\ndistance, noise)"]); ax.set_ylabel("predicted σ of acoustic SNR (dB)"); ax.set_title(TAG + "Uncertainty ID vs OOD"); save(fig, "B", 17, "ood_vs_id_uncertainty")

# ---------------------------------------------------------------- link-level sweeps (C)
tr = X.traces(); rng = np.random.default_rng(5); sc0 = S.make_scenarios(tr, 24, 8, rng, sites=X.TEST_SITES)
dgrid = np.exp(np.linspace(np.log(2), np.log(4000), 28)); cs = (0.15, 0.4, 0.75)
def sweep(c):
    out = {a: {k: [] for k in ("pdr", "ber", "goodput", "energy", "latency")} for a in range(6)}
    for dd in dgrid:
        sc = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sc0.items()}; sc["dist"][:] = dd; sc["logc"][:] = np.log(c); sa, so = S.snr_db(sc, P)
        for a in range(6):
            m = S.link_metrics(sa, so, np.full(sa.shape, a), p=P, dist=sc["dist"])
            for k in out[a]: out[a][k].append(float(m[k].mean()))
    return out
SW = {c: sweep(c) for c in cs}; ACT = {0: "AC low", 1: "AC high", 2: "OP low", 3: "OP high", 4: "HYB low", 5: "HYB high"}
def sweep_fig(num, slug, key, ylab, logy=False, note="", acts=(0, 1, 2, 3, 4, 5)):
    only_ac = set(acts) <= {0, 1}; cs_ = cs[:1] if only_ac else cs
    fig, ax = plt.subplots(1, len(cs_), figsize=(4.0 * len(cs_) + 1, 3.2), sharey=True, squeeze=False); ax = ax[0]
    for k, c in enumerate(cs_):
        for a in acts: ax[k].plot(dgrid, np.maximum(SW[c][a][key], 1e-12) if logy else SW[c][a][key], label=ACT[a], ls="-" if a % 2 else "--", color=["#a67bb5", "#a67bb5", "#6a9f58", "#6a9f58", "#c4562d", "#c4562d"][a])
        ax[k].set_xscale("log"); ax[k].set_xlabel("distance (m)"); ax[k].set_title("acoustic modes (independent of optical attenuation)" if only_ac else f"optical attenuation c={c} /m"); logy and ax[k].set_yscale("log")
    ax[0].set_ylabel(ylab); ax[0].legend(frameon=False, fontsize=6); fig.suptitle(TAG + f"{ylab} vs distance (acoustic gain from real traces; optical c fixed per panel)", y=1.03, fontsize=9); save(fig, "C", num, slug, note)
sweep_fig(18, "acoustic_pdr_vs_distance", "pdr", "PDR (acoustic modes)", acts=(0, 1)); sweep_fig(19, "optical_pdr_vs_distance", "pdr", "PDR (optical modes)", acts=(2, 3)); sweep_fig(20, "hybrid_pdr_vs_distance", "pdr", "PDR (hybrid modes)", acts=(4, 5))
sweep_fig(23, "throughput_vs_distance", "goodput", "goodput (kbit/s)", logy=True); sweep_fig(24, "energy_vs_distance", "energy", "energy per step (J)"); sweep_fig(25, "latency_vs_distance", "latency", "latency (s)", logy=True)
snr = np.linspace(-5, 30, 100); fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.2)); ax[0].semilogy(snr, S.ber_bpsk(snr), color=OURS); ax[0].set_xlabel("acoustic SNR (dB)"); ax[0].set_ylabel("BER"); ax[0].set_title("acoustic: coherent BPSK (Eb/N0 = SNR assumed)")
sa_, so_ = S.snr_db(sc0, P); ax[0].scatter(sa_.ravel()[::40], S.ber_bpsk(sa_.ravel()[::40]), s=3, color="k", alpha=.3, label="simulated link states"); ax[0].set_ylim(1e-9, .5); ax[0].legend(frameon=False)
ax[1].semilogy(snr + 5, S.ber_qam64(snr + 5), color="#2f6f9f"); ax[1].set_xlabel("optical electrical SNR (dB)"); ax[1].set_title("optical: 64-QAM (theory)"); ax[1].set_ylim(1e-9, .5); save(fig, "C", 21, "acoustic_ber_vs_snr", "theory curves + simulated states")
fig, ax = plt.subplots(figsize=(4.6, 3.2)); ax.semilogy(snr + 5, S.ber_qam64(snr + 5), color="#2f6f9f"); ms = np.clip(so_.ravel(), -5, 45)[::40]; ax.scatter(ms, S.ber_qam64(ms), s=3, color="k", alpha=.3); ax.set_xlabel("optical electrical SNR (dB)"); ax.set_ylabel("BER"); ax.set_ylim(1e-9, .5); ax.set_title(TAG + "Optical BER vs SNR (64-QAM; compare measured lab BER in plots/C_link_optical)", fontsize=8); save(fig, "C", 22, "optical_ber_vs_snr")

# ---------------------------------------------------------------- controllers (C26, D, E44-46)
cb = SIM / "controllers_eval_all.csv"
mp = []
if cb.exists():
    CT = pd.concat([pd.read_csv(cb)] + mp, ignore_index=True)
    def bar(num, base, slug, metrics=("reward", "goodput", "pdr", "energy", "viol"), ts="test_id"):
        order = [base, "PIGT-DT"] if base != "Oracle" else ["PIGT-DT", "Oracle"]
        fig, ax = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 3.0))
        for k, m in enumerate(metrics):
            g = CT[(CT.testset == ts) & CT.controller.isin(order)].groupby("controller")[m].agg(["mean", "std"]); vals = [g.loc[o, "mean"] if o in g.index else np.nan for o in order]; ax[k].bar(range(2), vals, yerr=[g.loc[o, "std"] if o in g.index and not np.isnan(g.loc[o, "std"]) else 0 for o in order], color=[CC.get(o, "grey") for o in order], capsize=2); ax[k].set_xticks(range(2)); ax[k].set_xticklabels(order, rotation=15, fontsize=7); ax[k].set_title(m)
        fig.suptitle(TAG + f"PIGT-DT controller vs {base} ({ts}; MAPPO: mean±std over seeds)", y=1.03, fontsize=9); save(fig, "E", num, slug)
    bar(44, "MAPPO", "pigtdt_vs_mappo"); bar(45, "Heuristic", "pigtdt_vs_heuristic"); bar(46, "Oracle", "pigtdt_vs_oracle")
    fig, ax = plt.subplots(figsize=(8, 3.4)); tg_ = CT[CT.testset == "test_id"].groupby("controller")[["reward"]].mean(); order = [c for c in ["AC-only", "OP-only", "HYB-low", "Heuristic", "Reactive", "MAPPO", "MAPPO+DT", "PIGT-DT", "PIGT-DT (uncertainty MC)", "PIGT-DT (risk-averse)", "Oracle"] if c in tg_.index]
    gg = CT[CT.testset == "test_id"].groupby("controller")["reward"].agg(["mean", "std"]).loc[order]; ax.bar(range(len(order)), gg["mean"], yerr=gg["std"].fillna(0), color=[CC.get(o, "#bbbbbb") for o in order], capsize=2); ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=25, ha="right"); ax.set_ylabel("mean reward per node-step"); ax.set_title(TAG + "All controllers (test_id)")
    fig.savefig(OUT / DIRS["E"] / "00_sim_all_controllers.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    # C26 mode selection probability vs distance
    ep = C.Episode(X.gen(3000, X.TEST_SITES, 64), 64, P, None); bins = np.exp(np.linspace(np.log(3), np.log(2500), 9)); fig, ax = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for k, (nm, pol) in enumerate((("Reactive", C.Greedy("obs")), ("Oracle", C.Greedy("oracle")), ("Heuristic", C.Heuristic()))):
        out = C.rollout(ep, pol, P); mode = out["act"] // 2; dd = ep.dist; idx = np.digitize(dd, bins)
        for m, lab, c in ((0, "AC", "#a67bb5"), (1, "OP", "#6a9f58"), (2, "HYB", "#c4562d")): ax[k].plot([np.sqrt(bins[i - 1] * bins[i]) for i in range(1, len(bins))], [((mode == m) & (idx == i)).sum() / max((idx == i).sum(), 1) for i in range(1, len(bins))], "o-", label=lab, color=c)
        ax[k].set_xscale("log"); ax[k].set_xlabel("distance (m)"); ax[k].set_title(nm)
    ax[0].set_ylabel("selection probability"); ax[0].legend(frameon=False); fig.suptitle(TAG + "AC / OP / HYB mode-selection probability vs distance", y=1.03); save(fig, "C", 26, "mode_selection_probability")
if mp or True:
    for fam, nm in (("mappo", "MAPPO"), ("mappo_dt", "MAPPO+DT")): pass
hist = {fam: [pd.read_csv(f) for f in sorted(glob.glob(str(SIM / f"mappo_hist_{fam}_s*.csv")))] for fam in ("mappo", "mappo_dt")}
if hist["mappo"] and cb.exists():
    # reference lines: baselines evaluated on training-distribution episodes
    ref_ep = C.Episode(X.gen(5999, X.TRAIN_SITES, 64), 64, P, None); refs = {n: C.summarize(C.rollout(ref_ep, pol, P)) for n, pol in (("Reactive", C.Greedy("obs")), ("Oracle", C.Greedy("oracle")), ("Heuristic", C.Heuristic()))}
    for num, key, slug, yl in ((27, "reward", "reward_vs_episode", "reward / node-step"), (28, "pdr", "pdr_vs_episode", "PDR"), (29, "goodput", "throughput_vs_episode", "goodput (kbit/s)"), (30, "energy", "energy_vs_episode", "energy / node-step (J)"), (31, "ber", "ber_vs_episode", "BER"), (32, "latency", "latency_vs_episode", "latency (s)"), (33, "viol", "constraint_violations_vs_episode", "violation rate (PDR<0.9 or battery death)")):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for fam, nm in (("mappo", "MAPPO"), ("mappo_dt", "MAPPO+DT")):
            if not hist[fam]: continue
            Hh = pd.concat([h.assign(s=i) for i, h in enumerate(hist[fam])]); g = Hh.groupby("episodes")[key].agg(["mean", "std"]); sm = g.rolling(10, min_periods=1).mean(); ax.plot(sm.index, sm["mean"], color=CC[nm], label=nm); ax.fill_between(sm.index, sm["mean"] - sm["std"].fillna(0), sm["mean"] + sm["std"].fillna(0), color=CC[nm], alpha=.15)
        for n, ls in (("Reactive", ":"), ("Heuristic", "-."), ("Oracle", "--")): ax.axhline(refs[n][key], color=CC[n], ls=ls, lw=1, label=n)
        ax.set_xlabel("training episodes (16 envs × 8 nodes each)"); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=6); ax.set_title(TAG + yl + " vs training episode", fontsize=8); save(fig, "D", num, slug)
    ser = {}
    for f in [SIM / "series_eval_all.npz"]:
        if f.exists():
            z = np.load(f)
            for k in z.files: lab, m = k.split("__"); ser.setdefault(lab, {})[m] = z[k]
    show = [c for c in ["Heuristic", "Reactive", "PIGT-DT", "MAPPO", "MAPPO+DT", "Oracle"] if c in ser]
    for num, key, slug, yl in ((34, "energy", "cumulative_energy", "cumulative energy per node (J)"), (35, "goodput", "cumulative_throughput", "cumulative delivered data per node (kbit)")):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for c in show: v = ser[c][key]; ax.plot(np.arange(len(v)) * S.STEP_S, np.cumsum(v) * (S.STEP_S if key == "goodput" else 1.0), label=c, color=CC.get(c, "grey"), lw=2 if c == "PIGT-DT" else 1.1)
        ax.set_xlabel("time (s)"); ax.set_ylabel(yl); ax.legend(frameon=False, fontsize=6); ax.set_title(TAG + yl, fontsize=8); save(fig, "D", num, slug)
    fig, ax = plt.subplots(len(show), 1, figsize=(7.5, 1.3 * len(show) + .5), sharex=True)
    for a_, c in zip(np.atleast_1d(ax), show): a_.plot(np.arange(len(ser[c]["act0"])) * S.STEP_S, ser[c]["act0"], drawstyle="steps-post", color=CC.get(c, "grey")); a_.set_yticks(range(6)); a_.set_yticklabels(["AC-L", "AC-H", "OP-L", "OP-H", "HYB-L", "HYB-H"], fontsize=5); a_.set_ylabel(c, fontsize=7)
    np.atleast_1d(ax)[-1].set_xlabel("time (s)"); fig.suptitle(TAG + "Mode/power switching over time (the node whose mode changes most under the oracle; one test episode)", y=1.0, fontsize=9); save(fig, "D", 36, "mode_switching_over_time")

# ---------------------------------------------------------------- ablations (F)
ab = SIM / "ablation_controllers.csv"; dma, dmr = SIM / "dt_metrics_ablate.csv", SIM / "dt_metrics_ratio.csv"
if ab.exists() and dma.exists():
    A_ = pd.read_csv(ab); Dm = pd.concat([pd.read_csv(SIM / "dt_metrics_main.csv"), pd.read_csv(dma)] + ([pd.read_csv(dmr)] if dmr.exists() else []))
    comp = A_[A_.experiment == "component"].set_index("variant")
    def comp_fig(num, slug, var, title, ref="PIGT-DT (full)"):
        fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.1)); sub = Dm[(Dm.testset == "test_id") & Dm.model.isin(["PIGT-DT", var]) & (Dm.kind == "single")]; cols = ["MAE_ac_h1", "MAE_ac_h2", "MAE_op_h1", "MAE_op_h2"]; g = sub.groupby("model")[cols].agg(["mean", "std"]); w = .35
        for j, m in enumerate(("PIGT-DT", var)): ax[0].bar(np.arange(4) + (j - .5) * w, [g.loc[m, (c, "mean")] if m in g.index else np.nan for c in cols], w, yerr=[g.loc[m, (c, "std")] if m in g.index else 0 for c in cols], capsize=2, color=[OURS, "grey"][j], label=m)
        ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["ac h1", "ac h2", "op h1", "op h2"]); ax[0].set_ylabel("forecast MAE (dB)"); ax[0].legend(frameon=False, fontsize=7); ax[0].set_title("digital-twin forecast error")
        rv = [comp.loc[ref, "reward"], comp.loc[var, "reward"] if var in comp.index else np.nan]; ax[1].bar(range(2), rv, color=[OURS, "grey"]); ax[1].set_xticks(range(2)); ax[1].set_xticklabels([ref.replace(" (full)", ""), var], rotation=10, fontsize=7); ax[1].set_ylabel("controller reward (test_id)"); ax[1].set_ylim(min(rv) - .1, max(rv) + .05); ax[1].set_title("DT-greedy controller reward")
        fig.suptitle(TAG + title, y=1.03, fontsize=9); save(fig, "F", num, slug)
    comp_fig(47, "remove_physics_prior", "- no physics prior", "Effect of removing the physics prior"); comp_fig(48, "remove_graph_structure", "- no graph", "Effect of removing graph structure"); comp_fig(49, "remove_temporal_module", "- no temporal", "Effect of removing the temporal module"); comp_fig(50, "remove_uncertainty", "- no uncertainty", "Effect of removing the uncertainty head")
    rk = A_[A_.experiment == "risk"]; fig, ax = plt.subplots(1, 3, figsize=(9.5, 3)); 
    for k, m in enumerate(("reward", "viol", "energy")): ax[k].bar(range(len(rk)), rk[m], color=[OURS, "grey", "#8aa4c0"][:len(rk)]); ax[k].set_xticks(range(len(rk))); ax[k].set_xticklabels([v.replace("risk constraint", "risk c.").replace(" (+50% violation weight, uncertainty MC)", "\n(risk-averse)") for v in rk.variant], rotation=12, fontsize=6); ax[k].set_title(m)
    fig.suptitle(TAG + "Effect of removing the risk (reliability-violation) term from the planner", y=1.03, fontsize=9); save(fig, "F", 51, "remove_risk_constraints")
    en = A_[A_.experiment == "ensemble"]; fig, ax = plt.subplots(figsize=(4.6, 3.1)); ax.plot(en.level, en.reward, "o-", color=OURS); ax.set_xlabel("ensemble size M"); ax.set_ylabel("controller reward (test_id)"); ax.set_title(TAG + "Effect of ensemble size"); save(fig, "F", 52, "ensemble_size")
    rr = A_[A_.experiment == "real_sim_ratio"]
    if len(rr): fig, ax = plt.subplots(figsize=(4.6, 3.1)); ax.plot(rr.level, rr.reward, "o-", color=OURS); ax.set_xlabel("% of DT training episodes using REAL measured acoustic traces (rest: synthetic AR traces)"); ax.set_ylabel("controller reward (test sites, real traces)"); ax.set_title(TAG + "Effect of real/synthetic data ratio", fontsize=8); save(fig, "F", 53, "real_vs_dt_data_ratio")
    mm = A_[A_.experiment == "mismatch"]; fig, ax = plt.subplots(figsize=(7.6, 3.2)); lv_ = list(dict.fromkeys(mm.level)); w = .27
    for j, lab in enumerate(("PIGT-DT", "Reactive", "Oracle")): g = mm[mm.variant == lab].set_index("level").loc[lv_]; ax.bar(np.arange(len(lv_)) + (j - 1) * w, g.reward, w, label=lab, color=CC[lab])
    ax.set_xticks(range(len(lv_))); ax.set_xticklabels(lv_, rotation=20, ha="right"); ax.set_ylabel("reward"); ax.legend(frameon=False); ax.set_title(TAG + "Effect of model mismatch (true link budget ≠ DT's nominal)"); save(fig, "F", 54, "model_mismatch")
    st = A_[A_.experiment == "staleness"]; fig, ax = plt.subplots(figsize=(5, 3.2))
    for lab in ("PIGT-DT", "Reactive", "Oracle"): g = st[st.variant == lab]; ax.plot(g.level, g.reward, "o-", label=lab, color=CC[lab])
    ax.set_xlabel("feedback delay / DT staleness (s)"); ax.set_ylabel("reward"); ax.legend(frameon=False); ax.set_title(TAG + "Effect of DT staleness"); save(fig, "F", 55, "dt_staleness")
    nf = A_[A_.experiment == "node_failure"]; fig, ax = plt.subplots(figsize=(5, 3.2))
    for lab, c in (("PIGT-DT (graph)", OURS), ("no graph", "#8aa4c0"), ("Reactive", "grey")): g = nf[nf.variant == lab]; ax.plot(g.level, g.reward, "o-", label=lab, color=c)
    ax.set_xlabel("fraction of failed nodes"); ax.set_ylabel("reward (surviving nodes)"); ax.legend(frameon=False); ax.set_title(TAG + "Effect of node failure"); save(fig, "F", 56, "node_failure")

# ---------------------------------------------------------------- scalability (G)
sc_f = SIM / "scalability_sim.csv"
if sc_f.exists():
    Sc = pd.read_csv(sc_f); dt_ = Sc[Sc.controller == "_DT"]; cc = Sc[Sc.controller != "_DT"]
    for num, col, slug, yl in ((57, "dt_infer_ms", "inference_latency_vs_nodes", "DT inference latency, whole network (ms, CPU)"), (58, "dt_mem_MB", "memory_vs_nodes", "tensor memory per forward pass (MB, CPU)"), (59, "dt_train_s_per_epoch", "training_time_vs_nodes", "DT training time per epoch (s, 16 envs)")):
        fig, ax = plt.subplots(figsize=(4.8, 3.2)); ax.plot(dt_.N, dt_[col], "o-", color=OURS); ax.set_xscale("log", base=2); ax.set_xlabel("number of nodes N"); ax.set_ylabel(yl, fontsize=8); ax.set_title(TAG + "single-member PIGT-DT, 4 vCPU, no GPU", fontsize=8); save(fig, "G", num, slug, "CPU only; no GPU measurement available")
    for num, col, slug, yl in ((60, "goodput_total", "throughput_vs_nodes", "network goodput (kbit/s, sum over nodes)"), (61, "pdr", "pdr_vs_nodes", "mean PDR"), (62, "energy_total", "energy_vs_nodes", "network energy per step (J)")):
        fig, ax = plt.subplots(figsize=(5.2, 3.3))
        for c in ("Heuristic", "Reactive", "PIGT-DT", "MAPPO", "MAPPO+DT", "Oracle"): g = cc[cc.controller == c]; ax.plot(g.N, g[col], "o-", label=c, color=CC[c], lw=2 if c == "PIGT-DT" else 1)
        ax.set_xscale("log", base=2); ax.set_xlabel("number of nodes N"); ax.set_ylabel(yl, fontsize=8); ax.legend(frameon=False, fontsize=6); ax.set_title(TAG + yl, fontsize=8); save(fig, "G", num, slug)

# ---------------------------------------------------------------- robustness (H)
rb = SIM / "robustness_controllers.csv"
if rb.exists():
    Rb = pd.read_csv(rb)
    def line(num, kind, slug, xl, metric="reward", cat=False, note=""):
        g = Rb[Rb.kind == kind]; fig, ax = plt.subplots(figsize=(7.2 if cat else 5.2, 3.3)); lv_ = list(dict.fromkeys(g.level)); w = .13
        for j, c in enumerate(("Heuristic", "Reactive", "PIGT-DT", "MAPPO", "MAPPO+DT", "Oracle")):
            gg = g[g.controller == c].set_index("level").loc[lv_]
            if cat: ax.bar(np.arange(len(lv_)) + (j - 2.5) * w, gg[metric], w, label=c, color=CC[c])
            else: ax.plot([float(x) for x in lv_], gg[metric], "o-", label=c, color=CC[c], lw=2 if c == "PIGT-DT" else 1)
        if cat: ax.set_xticks(range(len(lv_))); ax.set_xticklabels(lv_, rotation=18, ha="right", fontsize=7)
        ax.set_xlabel(xl); ax.set_ylabel(metric); ax.legend(frameon=False, fontsize=6, ncol=2); ax.set_title(TAG + f"Robustness: {slug.replace('_', ' ')}", fontsize=9); save(fig, "H", num, slug, note)
    line(63, "sensor_dropout", "sensor_dropout", "probability an observation is missing"); line(64, "channel_model_mismatch", "channel_model_mismatch", "true link parameters vs DT nominal", cat=True)
    line(65, "environment_mismatch", "environment_mismatch", "ambient-noise shift", cat=True); line(66, "noisy_observations", "noisy_observations", "extra observation noise std (dB)"); line(67, "stale_dt", "stale_dt", "feedback delay (s)")
    line(68, "unseen_distances", "unseen_distances", "distance regime", cat=True); line(69, "unseen_turbidity", "unseen_turbidity", "turbidity regime", cat=True); line(70, "unseen_acoustic_site", "unseen_acoustic_conditions", "acoustic recording site (blue/red seen in training)", cat=True)

# ---------------------------------------------------------------- INDEX
REQ = json.load(open(R / "scripts/figure_titles.json"))
real = json.load(open(OUT / "status_real.json")) if (OUT / "status_real.json").exists() else {}
lines = ["# Figure index (70 requested)", "", "Every requested figure now has at least one version. **[MEASURED]** = real measured data (acoustic channel recordings / optical BER tables). **[SIMULATION]** = produced from the hybrid-link simulator (`src/uwfc/linksim.py`): acoustic gain *dynamics* are real, but the link budget, optical attenuation, modem power, packet size, rewards, battery, and plume/turbidity dynamics are ASSUMED (`docs/SIMULATION_ASSUMPTIONS.md`). Simulation results are NOT evidence about real systems.", "",
         "| # | Requested | Version(s) |", "|---|---|---|"]
for n in range(1, 71):
    v = []
    if str(n) in real: v.append(f"[MEASURED] {real[str(n)][0]} → `{real[str(n)][1]}`")
    for st, path in simstatus.get(n, []): v.append(f"[SIMULATION] {st} → `{path}`")
    lines.append(f"| {n} | {REQ[str(n)]} | " + ("<br>".join(v) if v else "**not produced (missing prerequisite result)**") + " |")
(OUT / "INDEX.md").write_text("\n".join(lines)); print("sim figures:", sum(len(v) for v in simstatus.values()), "| numbers covered:", len(set(simstatus) | set(map(int, real))), "/ 70")
