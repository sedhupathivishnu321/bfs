# Figure index (70 requested)

Every requested figure now has at least one version. **[MEASURED]** = real measured data (acoustic channel recordings / optical BER tables). **[SIMULATION]** = produced from the hybrid-link simulator (`src/uwfc/linksim.py`): acoustic gain *dynamics* are real, but the link budget, optical attenuation, modem power, packet size, rewards, battery, and plume/turbidity dynamics are ASSUMED (`docs/SIMULATION_ASSUMPTIONS.md`). Simulation results are NOT evidence about real systems.

| # | Requested | Version(s) |
|---|---|---|
| 1 | Actual vs predicted acoustic SNR | [MEASURED] produced (adapted: channel gain dB instead of SNR) → `plots/A_prediction/01_actual_vs_pred_acoustic_gain.png`<br>[SIMULATION] simulated → `plots/A_prediction/01_sim_actual_vs_pred_acoustic_snr.png` |
| 2 | Actual vs predicted optical SNR | [SIMULATION] simulated → `plots/A_prediction/02_sim_actual_vs_pred_optical_snr.png` |
| 3 | Actual vs predicted BER | [MEASURED] produced (adapted: optical BER, log10) → `plots/A_prediction/03_actual_vs_pred_optical_ber.png`<br>[SIMULATION] simulated (BER panel is acoustic) → `plots/A_prediction/03_sim_actual_vs_pred_ber.png` |
| 4 | Actual vs predicted PDR | [SIMULATION] simulated → `plots/A_prediction/04_sim_actual_vs_pred_pdr.png` |
| 5 | Actual vs predicted throughput | [SIMULATION] simulated → `plots/A_prediction/05_sim_actual_vs_pred_throughput.png` |
| 6 | Prediction error vs time | [MEASURED] produced → `plots/A_prediction/06_error_vs_time.png`<br>[SIMULATION] simulated → `plots/A_prediction/06_sim_error_vs_time.png` |
| 7 | Prediction error vs distance | [MEASURED] produced (adapted: site-level range, n=5 sites) → `plots/A_prediction/07_error_vs_distance.png`<br>[SIMULATION] simulated → `plots/A_prediction/07_sim_error_vs_distance.png` |
| 8 | Prediction error vs channel condition | [MEASURED] produced → `plots/A_prediction/08_error_vs_channel_condition.png`<br>[SIMULATION] simulated → `plots/A_prediction/08_sim_error_vs_channel_condition.png` |
| 9 | MAE comparison across models | [MEASURED] produced → `plots/A_prediction/09_mae_comparison_across_models.png`<br>[SIMULATION] simulated → `plots/A_prediction/09_sim_mae_across_models.png` |
| 10 | RMSE comparison across models | [MEASURED] produced → `plots/A_prediction/10_rmse_comparison_across_models.png`<br>[SIMULATION] simulated → `plots/A_prediction/10_sim_rmse_across_models.png` |
| 11 | Prediction interval / uncertainty over time | [MEASURED] produced → `plots/B_uncertainty/11_prediction_interval_over_time.png`<br>[SIMULATION] simulated → `plots/B_uncertainty/11_sim_prediction_interval_over_time.png` |
| 12 | Calibration/reliability diagram | [MEASURED] produced → `plots/B_uncertainty/12_reliability_diagram.png`<br>[SIMULATION] simulated → `plots/B_uncertainty/12_sim_reliability_diagram.png` |
| 13 | Coverage vs nominal confidence | [MEASURED] produced → `plots/B_uncertainty/13_coverage_vs_nominal.png`<br>[SIMULATION] simulated → `plots/B_uncertainty/13_sim_coverage_vs_nominal.png` |
| 14 | NLL comparison | [MEASURED] produced → `plots/B_uncertainty/14_nll_comparison.png`<br>[SIMULATION] simulated → `plots/B_uncertainty/14_sim_nll_comparison.png` |
| 15 | CRPS comparison | [MEASURED] produced → `plots/B_uncertainty/15_crps_comparison.png`<br>[SIMULATION] simulated → `plots/B_uncertainty/15_sim_crps_comparison.png` |
| 16 | Sharpness vs coverage | [MEASURED] produced → `plots/B_uncertainty/16_sharpness_vs_coverage.png`<br>[SIMULATION] simulated → `plots/B_uncertainty/16_sim_sharpness_vs_coverage.png` |
| 17 | OOD uncertainty vs ID uncertainty | [MEASURED] produced → `plots/B_uncertainty/17_ood_vs_id_uncertainty.png`<br>[SIMULATION] simulated → `plots/B_uncertainty/17_sim_ood_vs_id_uncertainty.png` |
| 18 | Acoustic PDR vs distance | [SIMULATION] simulated → `plots/C_link_optical/18_sim_acoustic_pdr_vs_distance.png` |
| 19 | Optical PDR vs distance | [MEASURED] produced (adapted: BER instead of PDR (PDR not measured)) → `plots/C_link_optical/19_optical_ber_vs_distance_measured.png`<br>[SIMULATION] simulated → `plots/C_link_optical/19_sim_optical_pdr_vs_distance.png` |
| 20 | Hybrid PDR vs distance | [SIMULATION] simulated → `plots/C_link_optical/20_sim_hybrid_pdr_vs_distance.png` |
| 21 | Acoustic BER vs SNR | [SIMULATION] simulated (theory curves + simulated states) → `plots/C_link_optical/21_sim_acoustic_ber_vs_snr.png` |
| 22 | Optical BER vs SNR | [MEASURED] produced (adapted: BER vs IQ rate instead of vs SNR) → `plots/C_link_optical/22_optical_ber_vs_rate_measured.png`<br>[SIMULATION] simulated → `plots/C_link_optical/22_sim_optical_ber_vs_snr.png` |
| 23 | Throughput vs distance | [MEASURED] produced (adapted: throughput vs IQ rate, not vs distance) → `plots/C_link_optical/23_optical_throughput_vs_rate_measured.png`<br>[SIMULATION] simulated → `plots/C_link_optical/23_sim_throughput_vs_distance.png` |
| 24 | Energy vs distance | [SIMULATION] simulated → `plots/C_link_optical/24_sim_energy_vs_distance.png` |
| 25 | Latency vs distance | [SIMULATION] simulated → `plots/C_link_optical/25_sim_latency_vs_distance.png` |
| 26 | AC / OP / HYB mode-selection probability | [SIMULATION] simulated → `plots/C_link_optical/26_sim_mode_selection_probability.png` |
| 27 | Reward vs training episode | [SIMULATION] simulated → `plots/D_controller/27_sim_reward_vs_episode.png` |
| 28 | PDR vs training episode | [SIMULATION] simulated → `plots/D_controller/28_sim_pdr_vs_episode.png` |
| 29 | Throughput vs training episode | [SIMULATION] simulated → `plots/D_controller/29_sim_throughput_vs_episode.png` |
| 30 | Energy consumption vs training episode | [SIMULATION] simulated → `plots/D_controller/30_sim_energy_vs_episode.png` |
| 31 | BER vs training episode | [SIMULATION] simulated → `plots/D_controller/31_sim_ber_vs_episode.png` |
| 32 | Latency vs training episode | [SIMULATION] simulated → `plots/D_controller/32_sim_latency_vs_episode.png` |
| 33 | Constraint violations vs episode | [SIMULATION] simulated → `plots/D_controller/33_sim_constraint_violations_vs_episode.png` |
| 34 | Cumulative energy consumption | [SIMULATION] simulated → `plots/D_controller/34_sim_cumulative_energy.png` |
| 35 | Cumulative throughput | [SIMULATION] simulated → `plots/D_controller/35_sim_cumulative_throughput.png` |
| 36 | Communication-mode switching over time | [SIMULATION] simulated → `plots/D_controller/36_sim_mode_switching_over_time.png` |
| 37 | PIGT-DT vs Physics-only | [SIMULATION] simulated → `plots/E_baselines/37_sim_pigtdt_vs_physics_only.png` |
| 38 | PIGT-DT vs Data-only | [SIMULATION] simulated → `plots/E_baselines/38_sim_pigtdt_vs_data_only.png` |
| 39 | PIGT-DT vs LSTM | [MEASURED] produced (adapted: our model is PCT-E, not PIGT-DT) → `plots/E_baselines/39_pct_vs_lstm.png`<br>[SIMULATION] simulated → `plots/E_baselines/39_sim_pigtdt_vs_lstm.png` |
| 40 | PIGT-DT vs GRU | [MEASURED] produced (adapted: our model is PCT-E, not PIGT-DT) → `plots/E_baselines/40_pct_vs_gru.png`<br>[SIMULATION] simulated → `plots/E_baselines/40_sim_pigtdt_vs_gru.png` |
| 41 | PIGT-DT vs TCN | [MEASURED] produced (adapted: our model is PCT-E, not PIGT-DT) → `plots/E_baselines/41_pct_vs_tcn.png`<br>[SIMULATION] simulated → `plots/E_baselines/41_sim_pigtdt_vs_tcn.png` |
| 42 | PIGT-DT vs Transformer | [MEASURED] produced (adapted: our model is PCT-E, not PIGT-DT) → `plots/E_baselines/42_pct_vs_transformer.png`<br>[SIMULATION] simulated → `plots/E_baselines/42_sim_pigtdt_vs_transformer.png` |
| 43 | PIGT-DT vs GNN | [SIMULATION] simulated → `plots/E_baselines/43_sim_pigtdt_vs_gnn.png` |
| 44 | PIGT-DT vs MAPPO | [SIMULATION] simulated → `plots/E_baselines/44_sim_pigtdt_vs_mappo.png` |
| 45 | PIGT-DT vs heuristic controller | [SIMULATION] simulated → `plots/E_baselines/45_sim_pigtdt_vs_heuristic.png` |
| 46 | PIGT-DT vs oracle | [SIMULATION] simulated → `plots/E_baselines/46_sim_pigtdt_vs_oracle.png` |
| 47 | Effect of removing physics prior | [MEASURED] produced (adapted: AR/linear-extrapolation prior is the 'physics' prior here) → `plots/F_ablation/47_remove_physics_prior.png`<br>[SIMULATION] simulated → `plots/F_ablation/47_sim_remove_physics_prior.png` |
| 48 | Effect of removing graph structure | [MEASURED] produced (adapted: cross-receiver context stands in for graph structure) → `plots/F_ablation/48_remove_cross_receiver_context.png`<br>[SIMULATION] simulated → `plots/F_ablation/48_sim_remove_graph_structure.png` |
| 49 | Effect of removing temporal module | [SIMULATION] simulated → `plots/F_ablation/49_sim_remove_temporal_module.png` |
| 50 | Effect of removing uncertainty | [MEASURED] produced (adapted: heteroscedastic head) → `plots/F_ablation/50_remove_uncertainty_head.png`<br>[SIMULATION] simulated → `plots/F_ablation/50_sim_remove_uncertainty.png` |
| 51 | Effect of removing risk constraints | [SIMULATION] simulated → `plots/F_ablation/51_sim_remove_risk_constraints.png` |
| 52 | Effect of ensemble size | [MEASURED] produced → `plots/F_ablation/52_ensemble_size.png`<br>[SIMULATION] simulated → `plots/F_ablation/52_sim_ensemble_size.png` |
| 53 | Effect of real/DT data ratio | [SIMULATION] simulated → `plots/F_ablation/53_sim_real_vs_dt_data_ratio.png` |
| 54 | Effect of model mismatch | [SIMULATION] simulated → `plots/F_ablation/54_sim_model_mismatch.png` |
| 55 | Effect of DT staleness | [SIMULATION] simulated → `plots/F_ablation/55_sim_dt_staleness.png` |
| 56 | Effect of node failure | [SIMULATION] simulated → `plots/F_ablation/56_sim_node_failure.png` |
| 57 | Inference latency vs number of nodes | [MEASURED] produced (adapted: CPU only, no GPU; per-node forecasters) → `plots/G_scalability/57_inference_latency_vs_nodes.png`<br>[SIMULATION] simulated (CPU only; no GPU measurement available) → `plots/G_scalability/57_sim_inference_latency_vs_nodes.png` |
| 58 | GPU/CPU memory vs number of nodes | [MEASURED] produced (adapted: CPU only, no GPU; per-node forecasters) → `plots/G_scalability/58_memory_vs_nodes.png`<br>[SIMULATION] simulated (CPU only; no GPU measurement available) → `plots/G_scalability/58_sim_memory_vs_nodes.png` |
| 59 | Training time vs number of nodes | [MEASURED] produced (adapted: CPU only, no GPU; per-node forecasters) → `plots/G_scalability/59_training_time_vs_nodes.png`<br>[SIMULATION] simulated (CPU only; no GPU measurement available) → `plots/G_scalability/59_sim_training_time_vs_nodes.png` |
| 60 | Throughput vs number of nodes | [SIMULATION] simulated → `plots/G_scalability/60_sim_throughput_vs_nodes.png` |
| 61 | PDR vs number of nodes | [SIMULATION] simulated → `plots/G_scalability/61_sim_pdr_vs_nodes.png` |
| 62 | Energy vs number of nodes | [SIMULATION] simulated → `plots/G_scalability/62_sim_energy_vs_nodes.png` |
| 63 | Performance under sensor dropout | [MEASURED] produced → `plots/H_robustness/63_sensor_dropout.png`<br>[SIMULATION] simulated → `plots/H_robustness/63_sim_sensor_dropout.png` |
| 64 | Performance under channel-model mismatch | [SIMULATION] simulated → `plots/H_robustness/64_sim_channel_model_mismatch.png` |
| 65 | Performance under environmental mismatch | [MEASURED] produced (adapted: unseen site = unseen environment) → `plots/H_robustness/65_unseen_environment.png`<br>[SIMULATION] simulated → `plots/H_robustness/65_sim_environment_mismatch.png` |
| 66 | Performance under noisy observations | [MEASURED] produced → `plots/H_robustness/66_noisy_observations.png`<br>[SIMULATION] simulated → `plots/H_robustness/66_sim_noisy_observations.png` |
| 67 | Performance under stale DT | [MEASURED] produced (adapted: 'stale DT' = stale observations) → `plots/H_robustness/67_stale_dt.png`<br>[SIMULATION] simulated → `plots/H_robustness/67_sim_stale_dt.png` |
| 68 | Performance under unseen distances | [MEASURED] produced (optical only; right panel of figure 3) → `plots/A_prediction/03_actual_vs_pred_optical_ber.png`<br>[SIMULATION] simulated → `plots/H_robustness/68_sim_unseen_distances.png` |
| 69 | Performance under unseen turbidity | [MEASURED] produced (adapted: unseen water medium, optical BER) → `plots/H_robustness/69_unseen_medium_optical.png`<br>[SIMULATION] simulated → `plots/H_robustness/69_sim_unseen_turbidity.png` |
| 70 | Performance under unseen acoustic conditions | [MEASURED] produced (same figure as 65) → `plots/H_robustness/65_unseen_environment.png`<br>[SIMULATION] simulated → `plots/H_robustness/70_sim_unseen_acoustic_conditions.png` |