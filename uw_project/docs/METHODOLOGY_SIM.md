# Methodology – simulated hybrid link, digital twin and controllers (Part II)

All of Part II is **simulation** (see `SIMULATION_ASSUMPTIONS.md`); only the acoustic gain dynamics are measured.

## Environment (`src/uwfc/linksim.py`)
`N` nodes (default 8) talk to a hub. Per node and 0.25 s step: distance (slow drift), acoustic gain (a REAL measured trace), optical attenuation (OU + advected plumes), ambient noise. Link SNRs → BER (BPSK / 64-QAM) → PDR → goodput, energy, latency. Action per node: {AC, OP, HYB} × {low, high power}. Reward = 0.6·log2(1+goodput kbit/s) − 0.5·energy[J] − 2·latency[s] − 2·1[PDR<0.9] − 0.3·1[switch]; a battery (45 % of always-high-power acoustic energy) kills a node when exhausted. Observations are noisy (0.7 dB) and 1 s stale; distances are known.

## Digital twin (`src/uwfc/dt.py`) – "PIGT-DT"
* Task: from 6 s of stale observations of every node, forecast each node's acoustic and optical SNR at the decision step and 0.75 s later.
* **Physics prior**: acoustic – last observation corrected by the exactly known change of spreading+absorption loss with distance; optical – attenuation coefficient estimated from the last observation, then Beer–Lambert/geometric update for the new distance. The network predicts the **residual** to this prior.
* **Temporal module**: gated depthwise-separable dilated causal conv encoder (same blocks as PCT, Part I). **Graph module**: one message-passing layer over the kNN spatial graph (mean of neighbour embeddings). **Uncertainty**: heteroscedastic head + 3-member ensemble (variance = mean aleatoric + member disagreement).
* Baselines (same data/targets/optimiser, persistence prior): Physics-only (prior alone), Data-only (temporal, no graph, no physics), LSTM, GRU, TCN, Transformer, GNN (graph + last-step MLP, no temporal conv, no physics). 3 seeds each (PIGT-DT: 5 members, seeds 0–4).
* Ablations: no physics prior, no graph, no temporal module, no uncertainty head; ensemble size 1/2/3/5; 0/50/100 % synthetic AR acoustic traces instead of real ones in DT training.
* Splits: train sites {blue, red}, validation {yellow}, test {black, purple} (real acoustic traces unseen in training) + `test_ood` (unseen turbidity, distances, +6 dB noise). Scenario seeds differ across splits; normalisation constants are fixed (no fitting on test).

## Controllers (`src/uwfc/ctrl.py`)
| Controller | What it does |
|---|---|
| AC-only / OP-only / HYB-low | fixed action |
| Heuristic | thresholds on observed SNRs; the 3 thresholds are tuned on validation-site episodes (27-point grid) |
| Reactive | argmax of one-step expected reward computed from the stale *observed* SNRs (persistence model) |
| **PIGT-DT controller** | same planner, but expected reward is computed from the DT's forecast of the decision-step SNRs (ensemble mean); battery-aware energy price. Variants: Monte-Carlo over the predictive distribution, risk-averse (+50 % violation weight) |
| Oracle | same planner with the true decision-step SNRs (upper bound for one-step planners) |
| MAPPO / MAPPO+DT | shared-parameter actor, mean-field centralised critic, PPO (γ=0.95, λ=0.95, clip 0.2), 250 iterations × 16 episodes × 8 nodes, 3 seeds; MAPPO+DT additionally observes the DT forecast, its σ and implied PDRs |

Evaluation: 64 episodes × 8 nodes per test set (fixed seeds); results are expected values (no packet sampling). Paired bootstrap over episodes for PIGT-DT vs each controller.

## Stress tests / scalability
Sensor dropout (hold-last), extra observation noise, feedback delay 0.25–4 s, link-budget mismatch (SL ±3 dB, absorption ×1.5, optical c ×0.7/×1.5, noise ±6 dB), unseen distance/turbidity ranges, per-site results (blue/red seen; black/purple/yellow unseen), node failure (zeroed observations), N = 4…128 nodes (DT inference latency, tensor memory, training time per epoch on CPU; controller goodput/PDR/energy vs N with MAPPO trained at N = 8). No GPU was available, so no GPU memory/latency figures exist.

## Known limitations of the design
* Channel dynamics beyond the real acoustic gain are my assumptions; the graph-temporal DT is evaluated in a world where plumes propagate between neighbouring nodes **by construction**.
* Controllers are one-step planners + PPO; no multi-step MPC, no packet-level retransmission/queueing, no MAC contention, no interference between nodes.
* The "oracle" is an upper bound only for one-step greedy planning with a battery price, not for the true sequential optimum.
