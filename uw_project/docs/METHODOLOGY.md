# Methodology

## Task
Forecast, from the last 3 s of **measured** channel descriptors at one receiver (plus the mean of the other receivers), the **change** of acoustic channel gain (dB) and RMS delay spread (ms) at horizons 0.5, 1 and 2 s. Gain is a noise-independent SNR proxy (the repository does not contain noise power, so absolute SNR is unavailable).
Downstream relevance: a link-adaptation controller (modulation / power / mode) can act on the forecast before degradation. Target platform: a low-power modem / AUV processor → ≲ 20k parameters.

## Data (all real, CC-BY-4.0, verified in `results/data_audit.csv`)
* **Acoustic**: 14 recordings, 5 sites (Mariana Trench `black`, North Atlantic `blue`, North Atlantic `purple`, Singapore `red`, Hawaii `yellow`); time-varying channel impulse responses `h_hat(time, receiver, delay)`; 3–24 receivers, 12.5–25 kHz carriers, ≈ 37–54 s per recording. Only a subset of the repository (≈3 GB) was downloaded.
* **Optical**: OFDM-based underwater visible-light BER tables (Dratnal et al.): 540 measured points (BER vs IQ rate × {64,128,256}-QAM × 9 distances × {clean, pump-circulated murky water}).
* **ALOMEX 2015** (IMDEA): only the paper PDF is public; no measurement files → not used. Consequently **no simultaneous acoustic/optical measurements** exist in this study and hybrid-mode control could not be evaluated on real data.

## Features (`src/uwfc/features.py`)
From each CIR frame and receiver: gain (dB, 10 log10 Σ|h|²), RMS delay spread, peak-tap energy ratio, spectral flatness, peak-tap delay. Recordings are block-averaged to a common 16 Hz grid; at most 6 evenly spaced receivers per recording are kept (site balance).

## Leakage-safe protocols
* **Protocol A (primary)** – leave-one-**site**-out: the test site never appears in training; a different site is the validation set (early stopping, shrinkage fitting, model selection). All windows (input *and* targets) of a split lie inside that split's recordings.
* **Protocol B** – chronological 60/15/25 % split inside every recording; windows never straddle a boundary (`make_windows` enforces input+target ⊂ segment, unit-tested).
* Normalisation constants are fitted on training windows only (unit-tested). Every model gets identical windows, inputs, splits, seeds (3) and early-stopping rule.
* **Scale invariance (applies to ALL models)**: inputs are window-relative and divided by the window's own volatility (std of first differences); targets are divided by the same volatility; log-volatility and the normalised absolute level are extra inputs. This was introduced after a debugging pilot on the `blue` fold showed cross-site amplitude shift breaking every learned model (this pilot used a test fold; the fix is generic and applied identically to all models).

## Models
Baselines: Persistence, ridge AR, gradient boosting (HistGB), LSTM, GRU, TCN, Transformer (all ≈ 16–29k parameters except GBM).
**Proposed – PCT (Physics-prior Causal Temporal network, 12.0k parameters)**: input 1×1 conv → 4 gated depthwise-separable dilated causal conv blocks (dilations 1,2,4,8) → last-step + learned-attention pooling → heads. Mean forecast = `AR_prior(own relative history) + tanh(α+1)·residual` where the AR prior is a zero-initialised linear extrapolator (the "physics" prior: smooth channel dynamics), plus a heteroscedastic log-variance head trained with a Gaussian NLL on the detached mean. Loss: L1 on the mean (+0.3·NLL).
**Final – PCT-E**: mean of 3 PCT members (seeds differ), L1 loss, dropout 0.2 — configuration chosen **only on validation sites** (`scripts/06_dev_select.py`; 4 candidates; differences were ≈0.002 dB, i.e. selection was essentially a tie). Predictive variance = mean member variance + member disagreement.
**Controls**: single member, "+ shrink" (per-output factor λ∈[0,1.5] fitted on the validation site), and GBM + the same shrink, so that no gain can be attributed to post-processing alone.

## Metrics & statistics
MAE/RMSE of the forecast change in physical units (dB, ms); degradation-event detection (gain drop below the train-set 15th percentile at +1 s): AUC, precision, recall (sensitivity), specificity, F1, accuracy; uncertainty: 90 % coverage, interval width, Gaussian NLL and CRPS (non-PCT models are given a constant σ fitted on the validation site); complexity: parameters, FLOPs, CPU latency, tensor memory. Significance: **paired block bootstrap** (blocks of 40 consecutive windows to respect temporal autocorrelation, 3000 resamples) on pooled held-out windows; with only 5 sites, fold-level tests have little power and are reported descriptively. Seeds: 3 per configuration (ablation std over seeds is reported).

## Reproducibility
`run_all.sh` re-creates everything (resumable). Seeds fixed; absolute numbers vary slightly with thread count / BLAS because CPU training is not bit-deterministic.
