#!/usr/bin/env bash
# Full reproduction (≈ 3–4 h on 4 CPU cores, no GPU needed). Re-running resumes from existing results/*.csv.
set -e
export PYTHONPATH=src UWFC_THREADS=${UWFC_THREADS:-4} OMP_NUM_THREADS=${UWFC_THREADS:-4}
pip install -r requirements.txt
# 1. data (CC-BY-4.0): optical BER tables + selected acoustic recordings (~3 GB)
python - <<'PY'
import json, urllib.request, pathlib
d = json.load(urllib.request.urlopen("https://zenodo.org/api/records/17256508")); p = pathlib.Path("data/raw/ofdm_uwvc"); p.mkdir(parents=True, exist_ok=True)
[urllib.request.urlretrieve(f["links"]["self"], p / f["key"]) for f in d["files"]]
PY
bash data/raw/fetch_uwa.sh red_1 red_2 red_3 red_4 black blue_1 blue_2 blue_3 blue_4 blue_5 blue_10 yellow_4 purple_5 purple_10
python scripts/01_extract_features.py               # verify + featurise
python scripts/03_optical_ber.py                    # optical BER module
python scripts/02_run_benchmark.py --protocol A --suite main --seeds 3
python scripts/02_run_benchmark.py --protocol B --suite main --seeds 3
python scripts/02_run_benchmark.py --protocol B --suite ablation --seeds 3
python scripts/06_dev_select.py                     # validation-only config selection
python scripts/07_final.py --protocol A --seeds 3   # PCT-E + fairness controls + artifacts
python scripts/07_final.py --protocol B --seeds 3
python scripts/09_robustness.py
python scripts/04_complexity.py
python scripts/10_scalability.py
python scripts/08_make_plots.py
# ---- Part II (simulation) ----
python scripts/12_sim_dt.py --part main & python scripts/12_sim_dt.py --part ablate; wait; python scripts/12_sim_dt.py --part ratio
python scripts/13_sim_controllers.py --mode baselines
for s in 0 1 2; do python scripts/13_sim_controllers.py --mode mappo --seed $s; python scripts/13_sim_controllers.py --mode mappo_dt --seed $s; done
python scripts/13_sim_controllers.py --mode eval_all
python scripts/14_sim_ablation.py
python scripts/15_sim_scale_robust.py --part robust
python scripts/15_sim_scale_robust.py --part scale      # run on an otherwise idle machine (timing)
python scripts/08_make_plots.py && python scripts/16_sim_plots.py
python scripts/11_report.py
pytest -q tests
