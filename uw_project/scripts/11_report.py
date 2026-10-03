"""Builds results/FINAL_REPORT.md entirely from measured result files (no hand-typed numbers)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
R = Path(__file__).resolve().parents[1]; res = R / "results"
san = lambda m: "".join(ch if ch.isalnum() else "_" for ch in m)
L = []; P = L.append
def md(df): return df.to_markdown()
INV = {"PCT-E (ours)": "PCT-E (ensemble x3)", "PCT (single, base)": "PCT (proposed)"}
REN = {"PCT (proposed)": "PCT (single, base)", "PCT-E (ensemble x3)": "PCT-E (ours)"}

def load(proto):
    parts = [pd.read_csv(f) for s in ("main", "final") for f in sorted(res.glob(f"protocol{proto}_{s}*.csv"))]
    d = pd.concat(parts, ignore_index=True); d["model"] = d.model.replace(REN); return d

def table(d, cols, unit="fold"):
    fm = d.groupby(["model", unit])[cols].mean().reset_index(); g = fm.groupby("model")[cols].agg(["mean", "std"])
    t = pd.DataFrame({c: g[c]["mean"].round(4).astype(str) + " ± " + g[c]["std"].round(4).astype(str) for c in cols}); return t.loc[fm.groupby("model")[cols[1]].mean().sort_values().index], fm

def block_boot(proto, a, b, folds, blk=40, n=3000, seed=0):
    """paired block bootstrap over pooled held-out windows (seed-0 runs): mean(|err_b|-|err_a|). positive => a better."""
    ea, eb = [], []
    for sfx in ("final", "main"):
        pass
    def find(m):
        out = []
        for fo in folds:
            for suite in ("final", "main"):
                cands = sorted(res.glob(f"errs/{proto}_{suite}*/{fo}__0__{san(INV.get(m, m))}.npy"))
                if cands: out.append(np.load(cands[0])); break
            else: return None
        return np.concatenate(out)
    A_, B_ = find(a), find(b)
    if A_ is None or B_ is None or len(A_) != len(B_): return None
    nb = len(A_) // blk; d = (B_[:nb * blk] - A_[:nb * blk]).reshape(nb, blk).mean(1); rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, nb, nb)].mean() for _ in range(n)]); return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

P("# Final report – real-data underwater acoustic channel forecasting with a compact causal network (PCT / PCT-E)\n")
P("_All numbers below are produced by the scripts in this repository from measured data; nothing is hand-entered. See `docs/` for methodology and limitations._\n")
# ---- data
if (res / "data_audit.csv").exists():
    a = pd.read_csv(res / "data_audit.csv"); P("## 1. Data (real, CC-BY-4.0)\n")
    P(md(a.groupby("site").agg(recordings=("code", "count"), receivers=("receivers", "first"), delay_taps=("taps", "first"), duration_s=("duration_s", "first"), fc_kHz=("fc_hz", lambda x: x.iloc[0] / 1e3), nonfinite=("nonfinite_features", "sum"), site_description=("description", "first")).round(1)))
    P("\nSource: Underwater Acoustic Channel Repository (Zenodo 10.5281/zenodo.21287414) – measured time-varying channel impulse responses. Optical: Dratnal et al., Zenodo 10.5281/zenodo.17256508. The ALOMEX 2015 record (IMDEA) hosts the paper PDF only; no measurement files were available, so ALOMEX could not be used.\n")
# ---- protocols
dcols = ["MAE_gain_0.5s", "MAE_gain_1.0s", "MAE_gain_2.0s", "MAE_delay_1.0s", "RMSE_gain_1.0s", "ev_AUC", "ev_F1", "ev_recall_sens", "ev_specificity", "cov90"]
verdict = {}
for proto, title in (("A", "Protocol A – leave-one-site-out (cross-environment; primary)"), ("B", "Protocol B – chronological split inside every recording (deployment-like)")):
    try: d = load(proto)
    except ValueError: continue
    unit = "fold" if d.fold.nunique() > 1 else "seed"; cols = [c for c in dcols if c in d]; t, fm = table(d, cols, unit)
    P(f"## {title}\n"); P(f"folds: {sorted(d.fold.unique())}; seeds: {d.seed.nunique()}; values = mean ± std across {unit}s (fold-level means of seed-averaged values for A; seeds for B; MAE in dB / ms).\n"); P(md(t)); P("")
    ref = "PCT-E (ours)"
    if ref in fm.model.values:
        piv = fm.pivot(index=unit, columns="model", values="MAE_gain_1.0s"); piv2 = fm.pivot(index=unit, columns="model", values="MAE_gain_2.0s")
        P(f"\n**Paired comparison of {ref} (paired by {unit}).** Δ = MAE(other) − MAE(PCT-E); positive = PCT-E better. With only {len(piv)} paired units, counts are descriptive; see the bootstrap for uncertainty.\n")
        rows = []
        for m in piv.columns:
            if m == ref: continue
            r = {"model": m}
            for nm, pv in (("g1s", piv), ("g2s", piv2)):
                dd = pv[m] - pv[ref]; r[f"Δ MAE {nm} (dB)"] = round(dd.mean(), 4); r[f"PCT-E better in folds {nm}"] = f"{int((dd > 0).sum())}/{len(dd)}"
            rows.append(r)
        P(md(pd.DataFrame(rows).set_index("model").sort_values("Δ MAE g1s (dB)"))); P("")
        best = piv.drop(columns=[ref]).mean().idxmin(); best2 = piv2.drop(columns=[ref]).mean().idxmin()
        indep = [m for m in piv.columns if not m.startswith("PCT") and "shrink" not in m]
        def art_err(model, col, folds):
            out = []
            for fo in folds:
                z = np.load(res / f"artifacts/{proto}_{fo}/predictions.npz", allow_pickle=True); nm = INV.get(model, model)
                pred = z[f"te__{nm}"][:, col]; out.append(np.abs(pred - z["Yte"][:, col]) * z["Ste"][:, col])
            return np.concatenate(out)
        def boot(a, b, col, blk=40, n=3000):
            fo = sorted(d.fold.unique()); A_, B_ = art_err(a, col, fo), art_err(b, col, fo); nb = len(A_) // blk
            dd = (B_[:nb * blk] - A_[:nb * blk]).reshape(nb, blk).mean(1); rng = np.random.default_rng(0)
            bs = np.array([dd[rng.integers(0, nb, nb)].mean() for _ in range(n)]); return dd.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
        P(f"\n**Pooled-window paired block bootstrap** (seed-0 models, all held-out windows, blocks of 40 consecutive windows, 3000 resamples). Value = MAE(other) − MAE(PCT-E); positive = PCT-E better; 95% CI.\n")
        rows = []
        for hz, col, mm in (("+1 s", 2, piv), ("+2 s", 4, piv2)):
            b_ = mm[indep].mean().idxmin()
            for nm_, b2 in (("best independent baseline", b_), ("Persistence", "Persistence"), ("PCT (single, base)", "PCT (single, base)")):
                try: m_, lo, hi = boot(ref, b2, col)
                except Exception as e: continue
                rows.append({"horizon": hz, "vs": f"{b2} ({nm_})" if nm_.startswith("best") else nm_, "Δ MAE (dB)": round(m_, 4), "95% CI": f"[{lo:+.4f}, {hi:+.4f}]", "verdict": "PCT-E better" if lo > 0 else ("PCT-E worse" if hi < 0 else "inconclusive")})
                verdict[(proto, hz, nm_)] = (b2, m_, lo, hi)
        if rows: P(md(pd.DataFrame(rows).set_index("horizon")))
        P("")
        verdict[(proto, "rank1")] = (fm.groupby("model")["MAE_gain_1.0s"].mean().sort_values().index.tolist(), fm.groupby("model")["MAE_gain_2.0s"].mean().sort_values().index.tolist())
        P("")
# ---- ablation
if (res / "protocolB_ablation.csv").exists():
    a = pd.read_csv(res / "protocolB_ablation.csv"); g = a.groupby("model")[["MAE_gain_1.0s", "MAE_gain_2.0s", "MAE_delay_1.0s", "cov90"]].agg(["mean", "std"])
    t = pd.DataFrame({c: g[c]["mean"].round(4).astype(str) + " ± " + g[c]["std"].round(4).astype(str) for c in g.columns.levels[0]}); t = t.loc[g[("MAE_gain_1.0s", "mean")].sort_values().index]
    P("## Ablation (Protocol B, 3 seeds; ± = seed std)\n"); P("Changes of ≲0.02 dB are inside seed noise (compare std). Ablations were run on the Protocol-B test segment for analysis only; they were NOT used to choose the final configuration (validation-only, see §Selection).\n"); P(md(t)); P("")
# ---- selection
if (res / "dev_selected_config.json").exists():
    j = json.load(open(res / "dev_selected_config.json")); P("## Selection (validation sites only)\n"); P(f"Candidates scored on each fold's validation site; selected `{j['best']}` (cfg {j['cfg']}).\n")
    P(md(pd.DataFrame(j["ranking"]).round(4))); P("")
# ---- complexity
if (res / "complexity.csv").exists(): P("## Computational complexity (CPU, batch=1 / 256; no GPU)\n"); P(md(pd.read_csv(res / "complexity.csv").set_index("model"))); P("\nPCT-E uses 3 PCT members → 3× the PCT parameters/FLOPs/latency.\n")
# ---- optical
if (res / "optical_ber.csv").exists():
    o = pd.read_csv(res / "optical_ber.csv"); P("## Optical BER surrogate (real measured OFDM-UWVC tables; n=540 points; leave-one-distance-out / leave-one-medium-out)\n"); P(md(o.groupby(["split", "model"])[["MAE_log10", "RMSE_log10", "feas_acc"]].mean().round(3))); P("")
# ---- uncertainty
if (res / "uncertainty_metrics.csv").exists(): P("## Uncertainty (Protocol A, pooled held-out windows; non-PCT models use constant σ from validation residuals)\n"); P(md(pd.read_csv(res / "uncertainty_metrics.csv").set_index(["model", "horizon"]).round(4))); P("")
# ---- robustness
if (res / "robustness.csv").exists():
    r = pd.read_csv(res / "robustness.csv"); P("## Robustness (MAE gain +1 s, dB; mean over folds; seed-0 models)\n")
    for k in r.kind.unique(): P(f"**{k}**\n"); P(md(r[r.kind == k].pivot_table(index="model", columns="level", values="MAE_g1").round(3))); P("")
if (res / "scalability.csv").exists(): P("## Scalability (timing only)\n"); P(md(pd.read_csv(res / "scalability.csv").pivot_table(index="model", columns="nodes", values="infer_ms_total").round(2))); P("\n(inference ms for N nodes on 4 vCPU)\n")
# ---- verdict
# ======================================================================= SIMULATION STUDY
sim = res / "sim"
P("\n---\n\n# Part II – Simulated hybrid acoustic–optical link: digital twin, controllers, ablations, scalability, robustness\n")
P("> **Everything in Part II comes from a simulator** (`src/uwfc/linksim.py`). Only the acoustic channel-gain *dynamics* are real (measured recordings); link budgets, optical attenuation (with assumed advected turbidity plumes), power, energy, PDR, latency, reward, battery and 1 s feedback delay are **assumptions** (`docs/SIMULATION_ASSUMPTIONS.md`). Results describe behaviour inside this simulator, **not** real systems. Train sites: blue, red; validation: yellow; test: black, purple; `test_ood` = unseen turbidity (c 0.55–1.1 /m), distances (22–45 m, 2.2–3.5 km) and +6 dB noise.\n")
def dtm(part): return pd.read_csv(sim / f"dt_metrics_{part}.csv") if (sim / f"dt_metrics_{part}.csv").exists() else None
parts = [x for x in (dtm("main"), dtm("ablate"), dtm("ratio")) if x is not None]
if parts:
    M = pd.concat(parts, ignore_index=True); cols = ["MAE_ac_h1", "MAE_ac_h2", "MAE_op_h1", "MAE_op_h2", "MAE_pdr_ac", "MAE_pdr_op", "MAE_log10ber_ac", "MAE_thr_op_kbps"]
    for ts in ("test_id", "test_ood"):
        sub = M[(M.testset == ts) & M.kind.isin(["single", "ens"]) & ~M.tag.fillna("").str.startswith("ratio")]
        g = sub.groupby("model")[cols].agg(["mean", "std"]); t = pd.DataFrame({c: g[c]["mean"].round(3).astype(str) + (" ± " + g[c]["std"].round(3).astype(str)).where(g[c]["std"].notna(), "") for c in cols}).loc[g[("MAE_ac_h2", "mean")].sort_values().index]
        P(f"## Digital-twin forecasting – {ts} (mean ± std over 3 seeds; SNR errors in dB; h1 = decision step from 1 s-stale observations, h2 = +0.75 s; optical metrics only on active optical links, empty on OOD)\n"); P(md(t)); P("")
    ens = M[(M.testset == "test_id") & (M.kind == "ens_k")]
    if len(ens): P("Ensemble size (test_id):\n"); P(md(ens.set_index("model")[["MAE_ac_h1", "MAE_ac_h2", "MAE_op_h1", "MAE_op_h2", "NLL", "CRPS", "cov90"]].round(3))); P("")
if (sim / "dt_uncertainty.csv").exists(): P("## Digital-twin uncertainty (PIGT-DT ensemble vs constant-σ baselines)\n"); P(md(pd.read_csv(sim / "dt_uncertainty.csv").set_index(["model", "testset"]).round(3))); P("")
cf = sim / "controllers_eval_all.csv"
if cf.exists():
    C_ = pd.read_csv(cf); cols = ["reward", "goodput", "pdr", "ber", "energy", "latency", "viol", "switch_rate"]
    for ts in ("test_id", "test_ood"):
        g = C_[C_.testset == ts].groupby("controller")[cols].agg(["mean", "std"]); t = pd.DataFrame({c: g[c]["mean"].round(3).astype(str) + (" ± " + g[c]["std"].round(3).astype(str)).where(g[c]["std"].notna(), "") for c in cols}).loc[g[("reward", "mean")].sort_values(ascending=False).index]
        P(f"## Controllers – {ts} (64 episodes × 8 nodes; ± = std over 3 training seeds for MAPPO variants only; goodput kbit/s, energy J/step)\n"); P(md(t)); P("")
    if (sim / "per_env_rewards.npz").exists():
        Z = np.load(sim / "per_env_rewards.npz"); rng = np.random.default_rng(0); rowsb = []
        def pe(ts, c):
            ks = sorted(k for k in Z.files if k.startswith(f"{ts}__{c}__")); return np.mean([Z[k] for k in ks], 0) if ks else None
        for ts in ("test_id", "test_ood"):
            a = pe(ts, "PIGT-DT")
            for other in ("Reactive", "Heuristic", "MAPPO", "MAPPO+DT", "Oracle"):
                b = pe(ts, other)
                if a is None or b is None: continue
                d = a - b; bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]); lo, hi = np.percentile(bs, [2.5, 97.5])
                rowsb.append({"testset": ts, "PIGT-DT minus": other, "Δ reward": round(d.mean(), 4), "95% CI (paired bootstrap over 64 episodes)": f"[{lo:+.4f}, {hi:+.4f}]", "verdict": "PIGT-DT better" if lo > 0 else ("PIGT-DT worse" if hi < 0 else "inconclusive")})
        if rowsb: P("### PIGT-DT controller vs others – paired bootstrap over episodes\n"); P(md(pd.DataFrame(rowsb).set_index("testset"))); P(""); simverd = rowsb
if (sim / "ablation_controllers.csv").exists():
    A_ = pd.read_csv(sim / "ablation_controllers.csv"); P("## Controller-level ablations / stress tests (DT-greedy controller, test_id unless noted)\n")
    for e in A_.experiment.unique():
        P(f"**{e}**\n"); sub = A_[A_.experiment == e]
        if e in ("component", "risk", "ensemble", "real_sim_ratio"): P(md(sub.set_index("variant")[["reward", "goodput", "pdr", "energy", "latency", "viol"]].round(3)))
        else: P(md(sub.pivot_table(index="variant", columns="level", values="reward", sort=False).round(3)))
        P("")
if (sim / "robustness_controllers.csv").exists():
    Rb = pd.read_csv(sim / "robustness_controllers.csv"); P("## Robustness of controllers (reward)\n")
    for k in Rb.kind.unique(): P(f"**{k}**\n"); P(md(Rb[Rb.kind == k].pivot_table(index="controller", columns="level", values="reward", sort=False).round(2))); P("")
if (sim / "scalability_sim.csv").exists():
    Sc = pd.read_csv(sim / "scalability_sim.csv"); P("## Scalability (CPU, 4 vCPU, no GPU)\n"); P(md(Sc[Sc.controller == "_DT"].set_index("N")[["dt_infer_ms", "dt_infer_ms_per_node", "dt_ensemble3_ms", "dt_mem_MB", "dt_train_s_per_epoch"]].round(3))); P("")
    P(md(Sc[Sc.controller != "_DT"].pivot_table(index="controller", columns="N", values="goodput_total").round(0))); P("(network goodput, kbit/s)\n")
hh = {f: [pd.read_csv(x) for x in sorted(sim.glob(f"mappo_hist_{f}_s*.csv"))] for f in ("mappo", "mappo_dt")}
if hh["mappo"]: P("## MAPPO training (last 20 iterations of the training batch, mean over seeds)\n"); P(md(pd.DataFrame({n: pd.concat(v).query("iter >= iter.max()-19")[["reward", "pdr", "goodput", "energy", "viol"]].mean() for n, v in (("MAPPO", hh["mappo"]), ("MAPPO+DT", hh["mappo_dt"])) if v}).T.round(3))); P("")
P("## Verdict (computed from the numbers above)\n")
for proto in ("A", "B"):
    kk = (proto, "rank1")
    if kk not in verdict: continue
    r1, r2 = verdict[kk]; P(f"- **Protocol {proto}**: PCT-E ranks {r1.index('PCT-E (ours)')+1}/{len(r1)} at +1 s and {r2.index('PCT-E (ours)')+1}/{len(r2)} at +2 s by mean MAE (the list includes PCT variants and shrink controls).")
    for hz in ("+1 s", "+2 s"):
        for nm_ in ("best independent baseline", "Persistence"):
            v = verdict.get((proto, hz, nm_))
            if v: b2, m_, lo, hi = v; P(f"  - {hz} vs {b2}: Δ={m_:+.4f} dB, 95% CI [{lo:+.4f}, {hi:+.4f}] → " + ("**PCT-E significantly better**" if lo > 0 else ("PCT-E significantly worse" if hi < 0 else "inconclusive (CI includes 0)")))
P("\nSee `docs/LIMITATIONS.md` for caveats (5 sites, persistence-dominated task, validation reuse, scale-normalisation pilot).\n")
(res / "FINAL_REPORT.md").write_text("\n".join(L)); print("wrote FINAL_REPORT.md")
