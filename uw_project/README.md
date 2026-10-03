# Underwater channel forecasting and hybrid acoustic–optical control

Two parts, kept strictly separate because their evidence is different:

| | Part I – **measured** | Part II – **simulated** |
|---|---|---|
| Data | 14 real acoustic channel recordings (5 sites) + 540 real optical BER points | simulator (`src/uwfc/linksim.py`): real acoustic gain dynamics + **assumed** link budget, optical attenuation/plumes, power, packets, rewards, battery, 1 s feedback delay |
| What | forecasting of channel gain/delay spread with a compact network (PCT / PCT-E) vs 7 baselines; ablations; uncertainty; robustness; optical BER model | graph-temporal digital twin (PIGT-DT) vs baselines; controllers (heuristic, reactive, DT-planner, oracle, MAPPO, MAPPO+DT); ablations; scalability to 128 nodes; robustness |
| Evidence about the real world? | yes (limited: 5 sites) | **no** – behaviour inside the simulator only |

**Read first:** `results/FINAL_REPORT.md` (all numbers, generated from CSVs) · `docs/LIMITATIONS.md` · `docs/SIMULATION_ASSUMPTIONS.md` · `results/plots/INDEX.md` (all 70 requested figures, each marked [MEASURED] or [SIMULATION]).

## Headline results
**Part I (measured, 5 held-out sites, 3 seeds).** PCT-E has the lowest cross-site error at +1 s (0.517 dB) and +2 s (0.582 dB) and is significantly better than every independent baseline at +2 s (paired block bootstrap); at +1 s it beats persistence but its gap to the best baseline is inconclusive; on a chronological split it is not better than gradient boosting. Its uncertainty is badly overconfident on unseen sites; it loses its edge when ≥ 1 s of input is stale.

**Part II (simulation, test sites black/purple unseen in training).** Reward per node-step: Oracle 1.208 > MAPPO+DT 1.168 > PIGT-DT planner 1.140 > Reactive 1.118 > Heuristic 1.081 > MAPPO 1.070. PIGT-DT planner is better than Reactive/Heuristic/MAPPO (paired bootstrap CIs exclude 0) but its difference to MAPPO+DT is inconclusive and it closes only ≈24 % of the reactive→oracle gap. Digital-twin component ablations barely move controller reward (≤ 0.013). Out-of-distribution, MAPPO variants fail to generalise. A simulator bug (hybrid latency) was found and fixed during the study; all results were regenerated.

## What is real and what is not
Real: acoustic channel-gain fluctuations, optical data rate/BER tables. **Assumed (simulation only):** PDR, energy, latency, optical SNR, attenuation field, hybrid link, controller actions, multi-node scenarios. 35 of the 70 requested figures exist only as simulation figures; none of the simulated numbers should be quoted as real-system performance.

## Layout
```
src/uwfc/   features, data, models, train, baselines, metrics (Part I) · linksim, dt, ctrl, simexp (Part II)
scripts/    01 extract · 02 benchmark · 03 optical · 04 complexity · 06 selection · 07 final · 08 plots(measured) · 09 robustness · 10 scalability(real-data models)
            11 report · 12 sim DT · 13 sim controllers · 14 sim ablation · 15 sim scale/robust · 16 sim plots + INDEX
results/    CSVs, FINAL_REPORT.md, plots/ (A…H + INDEX.md), artifacts/ (Part I weights/predictions), sim/ (Part II results, DT/MAPPO weights)
docs/       METHODOLOGY (Part I) · METHODOLOGY_SIM · SIMULATION_ASSUMPTIONS · LITERATURE_AND_GAP · LIMITATIONS
tests/      leakage/normalisation/shape tests (Part I) and simulator/controller regression tests (Part II)
```
## Reproduce
`bash run_all.sh` (≈ 8–10 h on 4 CPU cores, no GPU; resumable). Tests: `PYTHONPATH=src pytest -q tests`.

## Data & licences
Underwater Acoustic Channel Repository (Zenodo 10.5281/zenodo.21287414, CC-BY-4.0); OFDM-UWVC (Dratnal et al., Zenodo 10.5281/zenodo.17256508, CC-BY-4.0). Raw files are not redistributed.
