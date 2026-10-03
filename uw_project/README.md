# Underwater channel forecasting on real measured data – PCT / PCT-E

Complete, reproducible pipeline: real data download → verification → featurisation → leakage-safe splits → baselines → proposed compact network (PCT) and ensemble (PCT-E) → ablations → uncertainty → robustness → scalability → figures → report.

**Read first:** `results/FINAL_REPORT.md` (all measured numbers, generated from CSVs), `docs/LIMITATIONS.md`, `results/plots/INDEX.md` (status of each of the 70 requested figures).

## Headline (measured, 5 held-out sites, 3 seeds)
* Cross-site forecasting (Protocol A): PCT-E has the lowest mean error at +1 s (0.517 dB) and +2 s (0.582 dB) – **significantly better than all independent baselines at +2 s** (−0.036 dB vs Transformer, bootstrap CI excludes 0) and better than persistence at both horizons; at +1 s the gap to the best baseline (Transformer) is **inconclusive**.
* On a chronological split (Protocol B) it is **not** better than gradient boosting.
* Weaknesses: overconfident uncertainty on unseen sites; loses its edge with ≥ 1 s stale input; some components unjustified by ablation. See `docs/LIMITATIONS.md`.
* Real-data scope: acoustic channel-impulse-response forecasting + optical BER surrogate. The original PIGT-DT hybrid-controller experiments need data that are not publicly available (see limitations) and were not run.

## Layout
```
src/uwfc/        features, data/windowing, models (LSTM/GRU/TCN/Transformer/PCT), training, baselines, metrics
scripts/         01 extract · 02 benchmark · 03 optical · 04 complexity · 06 selection · 07 final · 08 plots · 09 robustness · 10 scalability · 11 report
results/         CSVs, FINAL_REPORT.md, plots/ (A_prediction … H_robustness + INDEX.md), artifacts/ (trained seed-0 weights + predictions)
data/            processed features (raw data are re-downloadable: data/raw/fetch_uwa.sh, run_all.sh)
docs/            METHODOLOGY, LITERATURE_AND_GAP, LIMITATIONS
tests/           unit tests (leakage, normalisation, shapes)
```
## Reproduce
`bash run_all.sh` (≈ 3–4 h on 4 CPU cores; resumable). Tests: `PYTHONPATH=src pytest -q tests`.

## Data & licences
Underwater Acoustic Channel Repository (Zenodo 10.5281/zenodo.21287414, CC-BY-4.0); OFDM-UWVC dataset (Dratnal et al., Zenodo 10.5281/zenodo.17256508, CC-BY-4.0). Raw files are not redistributed here.
