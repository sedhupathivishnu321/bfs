"""Verify + featurise every downloaded recording; writes data/processed/features/<code>.npz and a data-audit CSV."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uwfc.features import load_cir, describe, FEATURES

R = Path(__file__).resolve().parents[1]
out = R / "data/processed/features"; out.mkdir(parents=True, exist_ok=True)
rows = []
for p in sorted((R / "data/raw/uwa").glob("*.mat")):
    if "noise" in p.name:
        continue
    try:
        h, info = load_cir(p)
    except Exception as e:  # corrupt / partial download
        print("SKIP", p.name, e); continue
    F = describe(h, info["fs_delay"])
    bad = int((~np.isfinite(F)).sum())
    np.savez_compressed(out / f"{info['codename']}.npz", F=F, fs_time=info["fs_time"], desc=info["description"])
    rows.append(dict(code=info["codename"], site=info["codename"].split("_")[0], frames=h.shape[0], receivers=h.shape[1],
                     taps=h.shape[2], duration_s=h.shape[0] / info["fs_time"], fs_time=info["fs_time"], fc_hz=info["fc"],
                     nonfinite_features=bad, gain_mean_db=float(F[..., 0].mean()), gain_std_db=float(F[..., 0].std()),
                     description=info["description"]))
    print(rows[-1]["code"], rows[-1]["description"], "T=", h.shape[0], "nonfinite=", bad)
pd.DataFrame(rows).to_csv(R / "results/data_audit.csv", index=False)
