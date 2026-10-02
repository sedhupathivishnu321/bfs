# Literature positioning and research gap

> Scope note: this is a targeted positioning review written from the authors' knowledge of the cited works; it was **not** produced by a systematic database search and should be extended (and each citation re-verified) before journal submission.

## Problem
Forecasting the short-horizon evolution of a *measured* underwater acoustic channel (gain, delay spread) so that an adaptive link (modulation / power / mode selection, including hybrid acoustic–optical switching) can act before degradation occurs. Target platform: low-power embedded modem / AUV processor, hence tiny models (≲ 20k parameters, ≲ 5 MFLOPs, ms latency) are required.

## Related work (verify before publication)
- **Data**: Underwater Acoustic Channel Repository (Zenodo 10.5281/zenodo.21287414, CC-BY-4.0): measured time-varying channel impulse responses from several oceans/sites. OFDM-based underwater visible-light dataset (Dratnal et al., Zenodo 10.5281/zenodo.17256508, CC-BY-4.0): measured BER vs IQ rate/QAM order/distance/medium. Diamant et al., "On the relationship between the underwater acoustic and optical channels," IEEE TWC 2017 (ALOMEX; the public record contains the paper only, not the raw measurements).
- **Sequence models**: LSTM (Hochreiter & Schmidhuber 1997), GRU (Cho et al. 2014), TCN (Bai et al. 2018), Transformer (Vaswani et al. 2017); long-horizon time-series Transformers (Informer, Zhou et al. 2021; PatchTST, Nie et al. 2023) and the simple-linear-baseline critique (DLinear, Zeng et al. AAAI 2023).
- **Distribution shift in forecasting**: reversible instance normalisation (RevIN, Kim et al. ICLR 2022) — motivates our per-window volatility normalisation, which turned out to be essential for cross-site transfer.
- **Efficient architectures**: depthwise-separable convolutions (Xception, Chollet 2017; MobileNets, Howard et al. 2017); gated convolutions (Dauphin et al. 2017).
- **Uncertainty**: heteroscedastic Gaussian likelihood and β-NLL (Seitzer et al. ICLR 2022); deep ensembles (Lakshminarayanan et al. 2017).
- **Compression** (not applied here, listed as future work): magnitude pruning (Han et al. 2015), post-training int8 quantisation, knowledge distillation (Hinton et al. 2015).

## Gap
1. Most underwater-channel prediction papers evaluate within one environment / one random split; **cross-site generalisation on public measured channels is rarely reported**, and windows from the same recording leak across random splits.
2. Neural forecasters are rarely compared with the strong *persistence* and gradient-boosting baselines under identical protocols; on slowly varying channels persistence is hard to beat.
3. Parameter/latency budgets for embedded deployment are seldom reported alongside accuracy.

## Hypothesis
H1. A scale-invariant, causal, depthwise-separable temporal network with an explicit linear extrapolation prior and a heteroscedastic head forecasts channel-gain/delay-spread changes with lower error than comparable-size recurrent/attention/TCN baselines under leave-one-site-out evaluation, at ≤ 15k parameters.
H2. Each component (AR prior, gating, depthwise-separable convs, cross-receiver context, absolute-level input, uncertainty head) contributes measurable accuracy or calibration.
(Whether H1/H2 are supported is decided only by the measured results in `results/`.)
