"""Optical-link BER surrogate on the real OFDM-UWVC measurements (Dratnal et al., Zenodo 10.5281/zenodo.17256508).
Target: log10 BER (floor 1e-5, the smallest resolvable non-zero value in the tables).
Features: IQ rate, bits/symbol, distance, medium (clean / pump-circulated murky water).
Protocols: (1) leave-one-DISTANCE-out, (2) leave-one-MEDIUM-out (train clean -> test pump and vice versa). n is small -> CIs by bootstrap.
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
R = Path(__file__).resolve().parents[1]; raw = R / "data/raw/ofdm_uwvc"
FLOOR = 1e-5


def read(p):
    df = pd.read_csv(p, sep=";", encoding="utf-8-sig", dtype=str)
    return df.map(lambda v: float(str(v).replace(",", ".")) if re.fullmatch(r"[\d,.eE+-]+", str(v)) else v)


rows = []
for medium, tag in (("clean", "Dataset_2_clean_water_measurement_{}_cm.csv"), ("pump", "Dataset_4_water_pump_measurement_{}_cm.csv")):
    for d in range(15, 60, 5):
        df = read(raw / tag.format(d))
        for _, r in df.iterrows():
            rate = float(str(r["IQ rate"]).rstrip("k")) * 1e3
            for q in ("64-QAM", "128-QAM", "256-QAM") if "64-QAM" in df.columns else ("64QAM", "128QAM", "256QAM"):
                rows.append(dict(medium=medium, dist=d, rate=rate / 1e6, bits=int(np.log2(int(q.replace("-", "").replace("QAM", "")))),
                                 ber=float(r[q])))
D = pd.DataFrame(rows); D["y"] = np.log10(np.maximum(D.ber, FLOOR)); D["pump"] = (D.medium == "pump").astype(int)
D.to_csv(R / "data/processed/optical_ber_table.csv", index=False)
print("rows", len(D), "zero-BER rows", int((D.ber == 0).sum()))
X = lambda d: np.c_[d.rate, d.bits, d.dist, d.pump, d.rate * d.bits, d.rate ** 2, d.rate * d.dist, d.bits * d.dist]
models = {"Mean": None, "Ridge(poly)": lambda: make_pipeline(StandardScaler(), Ridge(1.0)),
          "GBM": lambda: GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=.05, random_state=0),
          "MLP": lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-2, max_iter=3000, random_state=0))}
res = []; preds = []
def run(split_name, folds):
    for te_name, tr, te in folds:
        for m, mk in models.items():
            p = np.full(len(te), D.y.iloc[tr].mean()) if mk is None else mk().fit(X(D.iloc[tr]), D.y.iloc[tr]).predict(X(D.iloc[te]))
            e = p - D.y.iloc[te].values
            preds.append(pd.DataFrame(dict(split=split_name, held_out=te_name, model=m, medium=D.medium.iloc[te].values, dist=D.dist.iloc[te].values, rate=D.rate.iloc[te].values, bits=D.bits.iloc[te].values, y_true=D.y.iloc[te].values, y_pred=p)))
            # mode-selection utility: pick highest-rate*bits config meeting BER<=1e-3 -> does the predicted-feasible set match?
            feas_t = D.ber.iloc[te].values <= 1e-3; feas_p = 10 ** p <= 1e-3
            res.append(dict(split=split_name, held_out=te_name, model=m, MAE_log10=np.abs(e).mean(), RMSE_log10=np.sqrt((e ** 2).mean()),
                            feas_acc=(feas_t == feas_p).mean(), n=len(te)))
idx = np.arange(len(D))
run("LODO", [(f"{d}cm", idx[D.dist != d], idx[D.dist == d]) for d in sorted(D.dist.unique())])
run("LOMO", [(m, idx[D.medium != m], idx[D.medium == m]) for m in ("clean", "pump")])
pd.concat(preds).to_csv(R / 'results/optical_predictions.csv', index=False)
Rs = pd.DataFrame(res); Rs.to_csv(R / "results/optical_ber.csv", index=False)
print(Rs.groupby(["split", "model"])[["MAE_log10", "RMSE_log10", "feas_acc"]].mean().round(3))
