# Simulation assumptions (hybrid acoustic–optical link, digital twin, controllers)

**Read this before interpreting anything under `results/sim/`, plots marked `_sim_`, or the controller/DT/ablation/scalability/robustness sections of the report.**

No public dataset in this project contains PDR, energy, latency, optical SNR, a hybrid link, controller actions or multi-node data. Those quantities are therefore produced by a **simulator** (`src/uwfc/linksim.py`). Results from it show how methods behave *inside this simulator*; they are **not** evidence about real underwater systems.

## What is real
| Element | Source |
|---|---|
| Acoustic channel-gain fluctuations per node (dB, relative to each trace's mean) | **Measured** channel recordings (Zenodo 10.5281/zenodo.21287414); one real receiver trace per node; azimuth-neighbouring nodes get neighbouring receivers of the same recording (real spatial correlation). Train sites: blue, red; validation: yellow; test: black, purple. |
| Optical data rate (64-QAM, 2 MS/s) = 4490 kbit/s | **Measured** throughput table (OFDM-UWVC Dataset_6) |
| Optical BER-vs-SNR *shape* | standard 64-QAM theory; **not** fitted to the lab data (lab distances are 15–55 cm; extrapolating attenuation to metres from that range is not defensible) |

## What is assumed (all in `linksim.Params` / `make_scenarios`)
| Item | Value | Note |
|---|---|---|
| Control interval | 0.25 s (4 Hz) | 96-step episodes + 24-step history |
| Acoustic link budget | SL = 170.8 + 10log10(η·P) dB re 1 µPa (η = 0.5); TL = 15 log10 d + Thorp absorption at 12 kHz; in-band noise 118 dB + OU wind-noise ±1.5 dB; modulation BPSK, Eb/N0 = SNR | noise level was raised from an initial 111 dB because the first calibration made the acoustic link trivially perfect |
| Acoustic power / rate | 2 W (low), 10 W (high) electrical; 4 kbit/s | |
| Optical link budget | SNR_el = 90 dB − 20·c·d·log10(e) − 40 log10 d (+12 dB at high power), 64-QAM, 3 W / 8 W | c = optical attenuation coefficient (1/m) |
| Optical turbidity field | log c = μ + spatially-correlated OU (σ 0.2, τ 15 s) + **1–2 advected Gaussian plumes** (amplitude 0.6–1.4 in log c, speed 0.02–0.045 area-units/step) | plumes make upstream neighbours informative (graph) and trends predictable (temporal); **this is a design assumption that favours graph-temporal models** |
| Training regime | c = exp(−1.9…−0.9) = 0.15–0.41 /m; near nodes 4–22 m (optical-capable), far nodes 150–1800 m | OOD tests use c 0.55–1.1 /m, near 22–45 m, far 2200–3500 m, +6 dB noise |
| Distances | slow sinusoidal drift (AUV motion), assumed known to the controller | |
| Packet | 512 bit; PDR = (1−BER)^512; expected values, no packet sampling | |
| Feedback | observed SNRs are noisy (σ 0.7 dB) and **4 control steps (1 s) stale** | acoustic propagation + processing; without delay the reactive controller was within 2 % of the oracle and nothing could be learned |
| Reward per node-step | 0.6·log2(1+goodput kbit/s) − 0.5·E[J] − 2·latency[s] − 2·1[PDR<0.9] − 0.3·1[mode/power switch] | weights are assumptions |
| Battery | 45 % of the energy of always transmitting acoustically at high power; exhausted ⇒ node dead (reward −2, no goodput) | couples time steps |
| Actions | {AC, OP, HYB(parallel on both)} × {low, high power} | HYB goodput = sum of both, energy = sum |
| Latency | d/1500 s (acoustic) or d/2.25e8 (optical) + airtime | |
| Spatial graph | kNN (k = 3) on random node positions | |

## Calibration disclosure
Simulator constants were adjusted three times **before any controller/DT comparison was run**, using only (i) whether acoustic/optical PDR spanned (0,1) and (ii) whether the *oracle's* action choices were diverse and beat the best fixed policy. A fourth change (adding 1 s feedback delay) was made after observing that reactive ≈ oracle, i.e. that the problem left no room for forecasting; it is stated here because it is a design choice that creates the regime in which a forecasting digital twin can matter. Neither choice was tuned on, or to favour, any specific controller.

## What the simulation can and cannot support
* **Can**: relative behaviour of controllers/forecasters *given these assumptions*; effect of staleness, mismatch, dropout, node failure, ensemble size, data mix, scale — as sensitivity analyses of the assumed model.
* **Cannot**: absolute PDR / energy / latency / throughput of any real modem; whether hybrid switching helps in the field; whether MAPPO or the DT would transfer to real hardware. Optical results beyond the 15–55 cm lab distances are pure extrapolation of an assumed physical model.
