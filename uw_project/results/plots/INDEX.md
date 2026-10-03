# Figure index (70 requested)

Status per requested figure. **Produced** = computed from real measured data / trained models in this repo. **Not produced** = the real datasets cannot support it (no fabrication). Titles marked *adapted* differ from the request as stated.

| # | Requested | Status | File / reason |
|---|---|---|---|
| 1 | Actual vs predicted acoustic SNR | produced (adapted: channel gain dB instead of SNR) | `plots/A_prediction/01_actual_vs_pred_acoustic_gain.png` |
| 2 | Actual vs predicted optical SNR | **not produced** | no optical SNR in the public optical dataset (BER only) |
| 3 | Actual vs predicted BER | produced (adapted: optical BER, log10) | `plots/A_prediction/03_actual_vs_pred_optical_ber.png` |
| 4 | Actual vs predicted PDR | **not produced** | PDR/packet outcomes are not in either dataset |
| 5 | Actual vs predicted throughput | **not produced** | throughput is not measured over time/distance in the acoustic repository |
| 6 | Prediction error vs time | produced | `plots/A_prediction/06_error_vs_time.png` |
| 7 | Prediction error vs distance | produced (adapted: site-level range, n=5 sites) | `plots/A_prediction/07_error_vs_distance.png` |
| 8 | Prediction error vs channel condition | produced | `plots/A_prediction/08_error_vs_channel_condition.png` |
| 9 | MAE comparison across models | produced | `plots/A_prediction/09_mae_comparison_across_models.png` |
| 10 | RMSE comparison across models | produced | `plots/A_prediction/10_rmse_comparison_across_models.png` |
| 11 | Prediction interval / uncertainty over time | produced | `plots/B_uncertainty/11_prediction_interval_over_time.png` |
| 12 | Calibration/reliability diagram | produced | `plots/B_uncertainty/12_reliability_diagram.png` |
| 13 | Coverage vs nominal confidence | produced | `plots/B_uncertainty/13_coverage_vs_nominal.png` |
| 14 | NLL comparison | produced | `plots/B_uncertainty/14_nll_comparison.png` |
| 15 | CRPS comparison | produced | `plots/B_uncertainty/15_crps_comparison.png` |
| 16 | Sharpness vs coverage | produced | `plots/B_uncertainty/16_sharpness_vs_coverage.png` |
| 17 | OOD uncertainty vs ID uncertainty | produced | `plots/B_uncertainty/17_ood_vs_id_uncertainty.png` |
| 18 | Acoustic PDR vs distance | **not produced** | no PDR data |
| 19 | Optical PDR vs distance | produced (adapted: BER instead of PDR (PDR not measured)) | `plots/C_link_optical/19_optical_ber_vs_distance_measured.png` |
| 20 | Hybrid PDR vs distance | **not produced** | no hybrid link data |
| 21 | Acoustic BER vs SNR | **not produced** | no acoustic BER/SNR measurements (channel impulse responses only) |
| 22 | Optical BER vs SNR | produced (adapted: BER vs IQ rate instead of vs SNR) | `plots/C_link_optical/22_optical_ber_vs_rate_measured.png` |
| 23 | Throughput vs distance | produced (adapted: throughput vs IQ rate, not vs distance) | `plots/C_link_optical/23_optical_throughput_vs_rate_measured.png` |
| 24 | Energy vs distance | **not produced** | no energy measurements |
| 25 | Latency vs distance | **not produced** | no latency measurements |
| 26 | AC/OP/HYB mode-selection probability | **not produced** | no mode-selection experiment on real data |
| 27 | Reward vs training episode | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 28 | PDR vs training episode | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 29 | Throughput vs training episode | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 30 | Energy vs training episode | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 31 | BER vs training episode | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 32 | Latency vs training episode | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 33 | Constraint violations vs episode | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 34 | Cumulative energy | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 35 | Cumulative throughput | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 36 | Mode switching over time | **not produced** | no RL/controller was trained: requires a link simulator, which would be synthetic |
| 37 | PIGT-DT vs Physics-only | **not produced** | no physics-only (BELLHOP/optical-physics) model implemented; the 'AR prior' ablation (47) is the closest |
| 38 | PIGT-DT vs Data-only | **not produced** | no separate data-only variant of this model; see ablations |
| 39 | vs LSTM | produced (adapted: our model is PCT-E, not PIGT-DT) | `plots/E_baselines/39_pct_vs_lstm.png` |
| 40 | vs GRU | produced (adapted: our model is PCT-E, not PIGT-DT) | `plots/E_baselines/40_pct_vs_gru.png` |
| 41 | vs TCN | produced (adapted: our model is PCT-E, not PIGT-DT) | `plots/E_baselines/41_pct_vs_tcn.png` |
| 42 | vs Transformer | produced (adapted: our model is PCT-E, not PIGT-DT) | `plots/E_baselines/42_pct_vs_transformer.png` |
| 43 | vs GNN | **not produced** | no graph model implemented (cross-receiver context ablation is 48) |
| 44 | vs MAPPO | **not produced** | no MAPPO/controller |
| 45 | vs heuristic controller | **not produced** | no controller |
| 46 | vs oracle | **not produced** | no controller/oracle policy |
| 47 | Remove physics prior | produced (adapted: AR/linear-extrapolation prior is the 'physics' prior here) | `plots/F_ablation/47_remove_physics_prior.png` |
| 48 | Remove graph structure | produced (adapted: cross-receiver context stands in for graph structure) | `plots/F_ablation/48_remove_cross_receiver_context.png` |
| 49 | Remove temporal module | **not produced** | temporal-module ablation was not run |
| 50 | Remove uncertainty | produced (adapted: heteroscedastic head) | `plots/F_ablation/50_remove_uncertainty_head.png` |
| 51 | Remove risk constraints | **not produced** | no risk constraints in this forecaster |
| 52 | Ensemble size | produced | `plots/F_ablation/52_ensemble_size.png` |
| 53 | Real/DT data ratio | **not produced** | no digital-twin/simulated data in this project |
| 54 | Model mismatch | **not produced** | no simulator to mismatch (cross-site tests are in 65) |
| 55 | DT staleness | **not produced** | see 67 (stale observations) |
| 56 | Node failure | **not produced** | node-failure experiment not run (sensor dropout in 63) |
| 57 | Inference latency vs nodes | produced (adapted: CPU only, no GPU; per-node forecasters) | `plots/G_scalability/57_inference_latency_vs_nodes.png` |
| 58 | CPU/GPU memory vs nodes | produced (adapted: CPU only, no GPU; per-node forecasters) | `plots/G_scalability/58_memory_vs_nodes.png` |
| 59 | Training time vs nodes | produced (adapted: CPU only, no GPU; per-node forecasters) | `plots/G_scalability/59_training_time_vs_nodes.png` |
| 60 | Throughput vs nodes | **not produced** | no link-level throughput model |
| 61 | PDR vs nodes | **not produced** | no PDR data |
| 62 | Energy vs nodes | **not produced** | no energy data |
| 63 | Sensor dropout | produced | `plots/H_robustness/63_sensor_dropout.png` |
| 64 | Channel-model mismatch | **not produced** | no channel-model simulator |
| 65 | Environmental mismatch | produced (adapted: unseen site = unseen environment) | `plots/H_robustness/65_unseen_environment.png` |
| 66 | Noisy observations | produced | `plots/H_robustness/66_noisy_observations.png` |
| 67 | Stale DT | produced (adapted: 'stale DT' = stale observations) | `plots/H_robustness/67_stale_dt.png` |
| 68 | Unseen distances | produced (optical only; right panel of figure 3) | `plots/A_prediction/03_actual_vs_pred_optical_ber.png` |
| 69 | Unseen turbidity | produced (adapted: unseen water medium, optical BER) | `plots/H_robustness/69_unseen_medium_optical.png` |
| 70 | Unseen acoustic conditions | produced (same figure as 65) | `plots/H_robustness/65_unseen_environment.png` |