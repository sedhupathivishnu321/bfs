# Final report – real-data underwater acoustic channel forecasting with a compact causal network (PCT / PCT-E)

_All numbers below are produced by the scripts in this repository from measured data; nothing is hand-entered. See `docs/` for methodology and limitations._

## 1. Data (real, CC-BY-4.0)

| site   |   recordings |   receivers |   delay_taps |   duration_s |   fc_kHz |   nonfinite | site_description                                                                                                                                                                                            |
|:-------|-------------:|------------:|-------------:|-------------:|---------:|------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| black  |            1 |           8 |          176 |         36.6 |     18   |           0 | Mariana Trench; dT/dR/dW: 6m/8717m/8720m; d: 8.72km; fc: 18kHz; l: circular                                                                                                                                 |
| blue   |            6 |          12 |          320 |         52.3 |     13   |           0 | North Atlantic; dT/dR/dW: 30m-60m/50m/100m; d: 3-7km; fc: 13kHz; l: 12cm                                                                                                                                    |
| purple |            2 |          12 |          430 |         53.4 |     12.5 |           0 | North Atlantic; dT/dR/dW: 11m/10m/15m; d: 60m (cross, 32 elements, SE/SW), 200m (vertical 24 elements, SE/SW), 1km (vertical 12 elements, SE); fc: 12.5kHz; 3.75cm (cross), 5cm (vertical), 12cm (vertical) |
| red    |            4 |           3 |          768 |         47.8 |     25   |           0 | Singapore; dT/dR/dW: 6m/4.6m/8-15m; d: 0.1-0.4; fc: 25kHz; l: 80cm                                                                                                                                          |
| yellow |            1 |          24 |          188 |         52.1 |     13   |           0 | Hawaii; dT/dR/dW: 50m/50m/100m; d: 7km; fc: 13kHz; l: 20cm                                                                                                                                                  |

Source: Underwater Acoustic Channel Repository (Zenodo 10.5281/zenodo.21287414) – measured time-varying channel impulse responses. Optical: Dratnal et al., Zenodo 10.5281/zenodo.17256508. The ALOMEX 2015 record (IMDEA) hosts the paper PDF only; no measurement files were available, so ALOMEX could not be used.

## Protocol A – leave-one-site-out (cross-environment; primary)

folds: ['black', 'blue', 'purple', 'red', 'yellow']; seeds: 3; values = mean ± std across folds (fold-level means of seed-averaged values for A; seeds for B; MAE in dB / ms).

| model                        | MAE_gain_0.5s   | MAE_gain_1.0s   | MAE_gain_2.0s   | MAE_delay_1.0s   | RMSE_gain_1.0s   | ev_AUC          | ev_F1           | ev_recall_sens   | ev_specificity   | cov90           |
|:-----------------------------|:----------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|
| PCT-E (ours)                 | 0.3778 ± 0.2506 | 0.5171 ± 0.3422 | 0.5816 ± 0.3769 | 0.2305 ± 0.2069  | 0.6784 ± 0.4565  | 0.6565 ± 0.0642 | 0.0702 ± 0.0809 | 0.0426 ± 0.0535  | 0.9867 ± 0.0199  | 0.5918 ± 0.2697 |
| PCT-E (ensemble x3) + shrink | 0.379 ± 0.2559  | 0.5197 ± 0.3508 | 0.5887 ± 0.3844 | 0.2231 ± 0.2045  | 0.682 ± 0.466    | 0.6506 ± 0.0702 | 0.0822 ± 0.0844 | 0.0514 ± 0.054   | 0.9777 ± 0.0265  | 0.5891 ± 0.2662 |
| PCT-v2 (single) + shrink     | 0.3784 ± 0.2537 | 0.5213 ± 0.3544 | 0.5975 ± 0.3927 | 0.2303 ± 0.2145  | 0.6838 ± 0.47    | 0.6354 ± 0.0707 | 0.0745 ± 0.078  | 0.046 ± 0.0489   | 0.9806 ± 0.0223  | 0.5865 ± 0.2706 |
| PCT-v2 (single)              | 0.3806 ± 0.2516 | 0.5232 ± 0.3508 | 0.5926 ± 0.3892 | 0.2348 ± 0.21    | 0.6859 ± 0.4666  | 0.651 ± 0.0568  | 0.0643 ± 0.0798 | 0.0388 ± 0.0507  | 0.9861 ± 0.0186  | 0.5896 ± 0.2738 |
| PCT (single, base)           | 0.3784 ± 0.2503 | 0.5241 ± 0.3438 | 0.5865 ± 0.3778 | 0.2356 ± 0.2113  | 0.6867 ± 0.4595  | 0.6357 ± 0.0684 | 0.0645 ± 0.0555 | 0.0369 ± 0.0325  | 0.9886 ± 0.0099  | 0.5648 ± 0.2959 |
| GBM + shrink                 | 0.3735 ± 0.2596 | 0.5271 ± 0.37   | 0.641 ± 0.4053  | 0.223 ± 0.2062   | 0.6933 ± 0.4901  | 0.6375 ± 0.0982 | 0.0413 ± 0.0528 | 0.0228 ± 0.0294  | 0.9935 ± 0.0082  | nan             |
| Transformer                  | 0.3832 ± 0.2469 | 0.532 ± 0.3293  | 0.6237 ± 0.3773 | 0.2544 ± 0.2079  | 0.6934 ± 0.4455  | 0.6682 ± 0.0701 | 0.0349 ± 0.049  | 0.0239 ± 0.036   | 0.9952 ± 0.0056  | nan             |
| Persistence                  | 0.3784 ± 0.2665 | 0.5386 ± 0.3844 | 0.6436 ± 0.4476 | 0.2394 ± 0.2351  | 0.709 ± 0.5094   | 0.5 ± 0.0       | 0.0 ± 0.0       | 0.0 ± 0.0        | 1.0 ± 0.0        | nan             |
| LSTM                         | 0.3796 ± 0.2617 | 0.5404 ± 0.3796 | 0.6413 ± 0.4382 | 0.2463 ± 0.2283  | 0.7105 ± 0.504   | 0.5276 ± 0.0611 | 0.0007 ± 0.0014 | 0.0004 ± 0.0007  | 0.9998 ± 0.0003  | nan             |
| GBM                          | 0.3798 ± 0.2274 | 0.5521 ± 0.3058 | 0.807 ± 0.4283  | 0.2624 ± 0.2603  | 0.7224 ± 0.4055  | 0.5701 ± 0.2294 | 0.1575 ± 0.1145 | 0.1007 ± 0.0763  | 0.9735 ± 0.0274  | nan             |
| TCN                          | 0.3924 ± 0.2457 | 0.5548 ± 0.3411 | 0.6504 ± 0.3846 | 0.2513 ± 0.2072  | 0.7253 ± 0.4633  | 0.5947 ± 0.048  | 0.0355 ± 0.0513 | 0.0203 ± 0.0302  | 0.9928 ± 0.0094  | nan             |
| GRU                          | 0.3917 ± 0.2465 | 0.5604 ± 0.3525 | 0.6583 ± 0.4121 | 0.2543 ± 0.2158  | 0.7299 ± 0.4767  | 0.5135 ± 0.1044 | 0.0071 ± 0.0142 | 0.0037 ± 0.0075  | 0.9991 ± 0.0018  | nan             |
| AR(ridge)                    | 0.444 ± 0.2156  | 0.6956 ± 0.3171 | 0.9756 ± 0.53   | 0.3523 ± 0.3435  | 0.8953 ± 0.4357  | 0.6019 ± 0.1065 | 0.158 ± 0.1307  | 0.1834 ± 0.1254  | 0.9536 ± 0.0335  | nan             |


**Paired comparison of PCT-E (ours) (paired by fold).** Δ = MAE(other) − MAE(PCT-E); positive = PCT-E better. With only 5 paired units, counts are descriptive; see the bootstrap for uncertainty.

| model                        |   Δ MAE g1s (dB) | PCT-E better in folds g1s   |   Δ MAE g2s (dB) | PCT-E better in folds g2s   |
|:-----------------------------|-----------------:|:----------------------------|-----------------:|:----------------------------|
| PCT-E (ensemble x3) + shrink |           0.0026 | 3/5                         |           0.0072 | 3/5                         |
| PCT-v2 (single) + shrink     |           0.0042 | 3/5                         |           0.016  | 3/5                         |
| PCT-v2 (single)              |           0.0061 | 4/5                         |           0.011  | 3/5                         |
| PCT (single, base)           |           0.007  | 4/5                         |           0.0049 | 5/5                         |
| GBM + shrink                 |           0.01   | 3/5                         |           0.0594 | 5/5                         |
| Transformer                  |           0.0148 | 3/5                         |           0.0421 | 5/5                         |
| Persistence                  |           0.0215 | 4/5                         |           0.062  | 4/5                         |
| LSTM                         |           0.0232 | 4/5                         |           0.0597 | 4/5                         |
| GBM                          |           0.0349 | 3/5                         |           0.2254 | 4/5                         |
| TCN                          |           0.0377 | 5/5                         |           0.0689 | 5/5                         |
| GRU                          |           0.0433 | 5/5                         |           0.0767 | 5/5                         |
| AR(ridge)                    |           0.1784 | 5/5                         |           0.394  | 5/5                         |


**Pooled-window paired block bootstrap** (seed-0 models, all held-out windows, blocks of 40 consecutive windows, 3000 resamples). Value = MAE(other) − MAE(PCT-E); positive = PCT-E better; 95% CI.

| horizon   | vs                                      |   Δ MAE (dB) | 95% CI             | verdict      |
|:----------|:----------------------------------------|-------------:|:-------------------|:-------------|
| +1 s      | Transformer (best independent baseline) |       0.0044 | [-0.0074, +0.0156] | inconclusive |
| +1 s      | Persistence                             |       0.0199 | [+0.0059, +0.0341] | PCT-E better |
| +2 s      | Transformer (best independent baseline) |       0.0362 | [+0.0209, +0.0513] | PCT-E better |
| +2 s      | Persistence                             |       0.0747 | [+0.0534, +0.0955] | PCT-E better |


## Protocol B – chronological split inside every recording (deployment-like)

folds: ['within']; seeds: 3; values = mean ± std across seeds (fold-level means of seed-averaged values for A; seeds for B; MAE in dB / ms).

| model                        | MAE_gain_0.5s   | MAE_gain_1.0s   | MAE_gain_2.0s   | MAE_delay_1.0s   | RMSE_gain_1.0s   | ev_AUC          | ev_F1           | ev_recall_sens   | ev_specificity   | cov90           |
|:-----------------------------|:----------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|:----------------|:-----------------|:-----------------|:----------------|
| GBM                          | 0.3626 ± 0.0    | 0.5712 ± 0.0    | 0.7716 ± 0.0    | 0.1851 ± 0.0     | 0.7736 ± 0.0     | 0.7582 ± 0.0    | 0.3084 ± 0.0    | 0.2051 ± 0.0     | 0.977 ± 0.0      | nan             |
| GBM + shrink                 | 0.3625 ± 0.0    | 0.5725 ± 0.0    | 0.7705 ± 0.0    | 0.1844 ± 0.0     | 0.7736 ± 0.0     | 0.7582 ± 0.0    | 0.338 ± 0.0     | 0.2393 ± 0.0     | 0.9676 ± 0.0     | nan             |
| PCT-E (ensemble x3) + shrink | 0.3695 ± 0.0036 | 0.5796 ± 0.0036 | 0.7589 ± 0.0046 | 0.1866 ± 0.0014  | 0.782 ± 0.0049   | 0.7478 ± 0.0051 | 0.2769 ± 0.0313 | 0.1776 ± 0.0246  | 0.9812 ± 0.0018  | 0.9022 ± 0.0067 |
| PCT-E (ours)                 | 0.3688 ± 0.0035 | 0.5806 ± 0.0033 | 0.7603 ± 0.0046 | 0.1865 ± 0.0015  | 0.7816 ± 0.0047  | 0.7478 ± 0.0051 | 0.2901 ± 0.0303 | 0.1918 ± 0.0246  | 0.9765 ± 0.0019  | 0.9015 ± 0.0061 |
| TCN                          | 0.3744 ± 0.0029 | 0.5846 ± 0.0027 | 0.8021 ± 0.013  | 0.1944 ± 0.001   | 0.7988 ± 0.0037  | 0.7386 ± 0.006  | 0.3549 ± 0.0138 | 0.2626 ± 0.02    | 0.9604 ± 0.0062  | nan             |
| PCT-v2 (single) + shrink     | 0.3726 ± 0.0029 | 0.5873 ± 0.0038 | 0.773 ± 0.0018  | 0.1879 ± 0.0025  | 0.7925 ± 0.0043  | 0.7432 ± 0.0034 | 0.2347 ± 0.0094 | 0.1453 ± 0.0085  | 0.9831 ± 0.0035  | 0.8948 ± 0.0063 |
| PCT-v2 (single)              | 0.372 ± 0.0029  | 0.5895 ± 0.0025 | 0.7757 ± 0.0009 | 0.1882 ± 0.0024  | 0.7925 ± 0.0045  | 0.7432 ± 0.0034 | 0.2726 ± 0.029  | 0.1785 ± 0.0257  | 0.9765 ± 0.005   | 0.8914 ± 0.0072 |
| PCT (single, base)           | 0.3778 ± 0.0022 | 0.5919 ± 0.0039 | 0.7755 ± 0.0085 | 0.188 ± 0.0011   | 0.7958 ± 0.0045  | 0.7379 ± 0.0068 | 0.2785 ± 0.0162 | 0.1861 ± 0.01    | 0.9723 ± 0.0043  | 0.8868 ± 0.0102 |
| GRU                          | 0.3795 ± 0.0034 | 0.5977 ± 0.0106 | 0.7993 ± 0.0103 | 0.2001 ± 0.001   | 0.8109 ± 0.0117  | 0.7305 ± 0.0119 | 0.3074 ± 0.0564 | 0.2146 ± 0.0514  | 0.9684 ± 0.0054  | nan             |
| LSTM                         | 0.3855 ± 0.005  | 0.6025 ± 0.008  | 0.8042 ± 0.0095 | 0.2035 ± 0.0013  | 0.8059 ± 0.0136  | 0.7315 ± 0.0066 | 0.3618 ± 0.0119 | 0.2716 ± 0.0044  | 0.9576 ± 0.0099  | nan             |
| Transformer                  | 0.3949 ± 0.0074 | 0.6031 ± 0.0116 | 0.7867 ± 0.0084 | 0.1923 ± 0.0032  | 0.8075 ± 0.0186  | 0.7207 ± 0.0142 | 0.3445 ± 0.0213 | 0.2602 ± 0.0194  | 0.9544 ± 0.0069  | nan             |
| AR(ridge)                    | 0.3925 ± 0.0    | 0.6222 ± 0.0    | 0.8321 ± 0.0    | 0.1971 ± 0.0     | 0.8374 ± 0.0     | 0.7293 ± 0.0    | 0.3214 ± 0.0    | 0.2564 ± 0.0     | 0.9377 ± 0.0     | nan             |
| Persistence                  | 0.4301 ± 0.0    | 0.6631 ± 0.0    | 0.8887 ± 0.0    | 0.2128 ± 0.0     | 0.9249 ± 0.0     | 0.5 ± 0.0       | 0.0 ± 0.0       | 0.0 ± 0.0        | 1.0 ± 0.0        | nan             |


**Paired comparison of PCT-E (ours) (paired by seed).** Δ = MAE(other) − MAE(PCT-E); positive = PCT-E better. With only 3 paired units, counts are descriptive; see the bootstrap for uncertainty.

| model                        |   Δ MAE g1s (dB) | PCT-E better in folds g1s   |   Δ MAE g2s (dB) | PCT-E better in folds g2s   |
|:-----------------------------|-----------------:|:----------------------------|-----------------:|:----------------------------|
| GBM                          |          -0.0094 | 0/3                         |           0.0113 | 3/3                         |
| GBM + shrink                 |          -0.0081 | 0/3                         |           0.0102 | 3/3                         |
| PCT-E (ensemble x3) + shrink |          -0.001  | 0/3                         |          -0.0014 | 0/3                         |
| TCN                          |           0.004  | 2/3                         |           0.0418 | 3/3                         |
| PCT-v2 (single) + shrink     |           0.0066 | 3/3                         |           0.0128 | 3/3                         |
| PCT-v2 (single)              |           0.0089 | 3/3                         |           0.0155 | 3/3                         |
| PCT (single, base)           |           0.0113 | 3/3                         |           0.0153 | 3/3                         |
| GRU                          |           0.017  | 3/3                         |           0.0391 | 3/3                         |
| LSTM                         |           0.0219 | 3/3                         |           0.0439 | 3/3                         |
| Transformer                  |           0.0224 | 3/3                         |           0.0265 | 3/3                         |
| AR(ridge)                    |           0.0415 | 3/3                         |           0.0719 | 3/3                         |
| Persistence                  |           0.0825 | 3/3                         |           0.1285 | 3/3                         |


**Pooled-window paired block bootstrap** (seed-0 models, all held-out windows, blocks of 40 consecutive windows, 3000 resamples). Value = MAE(other) − MAE(PCT-E); positive = PCT-E better; 95% CI.

| horizon   | vs                              |   Δ MAE (dB) | 95% CI             | verdict      |
|:----------|:--------------------------------|-------------:|:-------------------|:-------------|
| +1 s      | GBM (best independent baseline) |      -0.013  | [-0.0326, +0.0055] | inconclusive |
| +1 s      | Persistence                     |       0.0799 | [+0.0363, +0.1287] | PCT-E better |
| +2 s      | GBM (best independent baseline) |       0.0091 | [-0.0356, +0.0488] | inconclusive |
| +2 s      | Persistence                     |       0.1277 | [+0.0477, +0.2126] | PCT-E better |


## Ablation (Protocol B, 3 seeds; ± = seed std)

Changes of ≲0.02 dB are inside seed noise (compare std). Ablations were run on the Protocol-B test segment for analysis only; they were NOT used to choose the final configuration (validation-only, see §Selection).

| model                        | MAE_gain_1.0s   | MAE_gain_2.0s   | MAE_delay_1.0s   | cov90           |
|:-----------------------------|:----------------|:----------------|:-----------------|:----------------|
| - cross-receiver ctx         | 0.5674 ± 0.0029 | 0.7484 ± 0.002  | 0.1834 ± 0.0019  | 0.8862 ± 0.0013 |
| - attention pooling          | 0.5675 ± 0.0024 | 0.7496 ± 0.0051 | 0.1857 ± 0.0007  | 0.8891 ± 0.0031 |
| - gating                     | 0.5802 ± 0.0068 | 0.7566 ± 0.0049 | 0.1905 ± 0.0046  | 0.89 ± 0.0088   |
| narrow (h=16)                | 0.5809 ± 0.0071 | 0.7558 ± 0.0082 | 0.1883 ± 0.0017  | 0.8978 ± 0.006  |
| shallow (dil 1,4)            | 0.5849 ± 0.0026 | 0.7662 ± 0.004  | 0.1865 ± 0.0007  | 0.8869 ± 0.0091 |
| PCT full                     | 0.5919 ± 0.0039 | 0.7755 ± 0.0085 | 0.188 ± 0.0011   | 0.8868 ± 0.0102 |
| - AR prior                   | 0.5931 ± 0.0127 | 0.8034 ± 0.0232 | 0.1915 ± 0.0016  | 0.8679 ± 0.01   |
| - absolute level             | 0.5944 ± 0.0034 | 0.7754 ± 0.011  | 0.1884 ± 0.0032  | 0.8978 ± 0.0016 |
| wide (h=64)                  | 0.5961 ± 0.0191 | 0.7864 ± 0.043  | 0.1932 ± 0.0083  | 0.8767 ± 0.0351 |
| - heteroscedastic head       | 0.6019 ± 0.0173 | 0.7928 ± 0.0255 | 0.194 ± 0.0044   | nan             |
| - depthwise-sep (dense conv) | 0.6093 ± 0.0172 | 0.7964 ± 0.0291 | 0.199 ± 0.001    | 0.8598 ± 0.0168 |
| gain-only inputs             | 0.6161 ± 0.0008 | 0.7885 ± 0.0005 | 0.2092 ± 0.0015  | 0.9025 ± 0.0078 |

## Selection (validation sites only)

Candidates scored on each fold's validation site; selected `l1+drop.2` (cfg {'loss': 'l1', 'drop': 0.2}).

|           |   val_g1 |   val_g2 |   val_d1 |   score |
|:----------|---------:|---------:|---------:|--------:|
| l1+drop.2 |   0.5094 |   0.5641 |   0.2214 |  0.5368 |
| base      |   0.5088 |   0.5697 |   0.2187 |  0.5393 |
| l1+wd1e-2 |   0.5091 |   0.5702 |   0.2181 |  0.5396 |
| l1        |   0.5092 |   0.5701 |   0.2182 |  0.5396 |

## Computational complexity (CPU, batch=1 / 256; no GPU)

| model       |   params |   param_MB |   MFLOPs_torch_counter |   MFLOPs_analytic |   latency_ms_b1 |   latency_ms_b256 |
|:------------|---------:|-----------:|-----------------------:|------------------:|----------------:|------------------:|
| LSTM        |    21126 |     0.0806 |                 0.0008 |            1.9423 |          0.4521 |            5.1415 |
| GRU         |    15942 |     0.0608 |                 1.4569 |            1.4569 |          1.1912 |            9.8628 |
| TCN         |    28902 |     0.1103 |                 2.7239 |          nan      |          0.604  |            7.5329 |
| Transformer |    19846 |     0.0757 |                 0.0465 |            2.2092 |          0.5834 |            8.7075 |
| PCT         |    11988 |     0.0457 |                 0.8769 |          nan      |          1.6307 |            8.7861 |

PCT-E uses 3 PCT members → 3× the PCT parameters/FLOPs/latency.

## Optical BER surrogate (real measured OFDM-UWVC tables; n=540 points; leave-one-distance-out / leave-one-medium-out)

|                         |   MAE_log10 |   RMSE_log10 |   feas_acc |
|:------------------------|------------:|-------------:|-----------:|
| ('LODO', 'GBM')         |       0.129 |        0.192 |      0.97  |
| ('LODO', 'MLP')         |       0.265 |        0.343 |      0.902 |
| ('LODO', 'Mean')        |       1.027 |        1.172 |      0.709 |
| ('LODO', 'Ridge(poly)') |       0.272 |        0.341 |      0.924 |
| ('LOMO', 'GBM')         |       0.142 |        0.207 |      0.969 |
| ('LOMO', 'MLP')         |       0.29  |        0.38  |      0.902 |
| ('LOMO', 'Mean')        |       1.027 |        1.173 |      0.709 |
| ('LOMO', 'Ridge(poly)') |       0.29  |        0.36  |      0.926 |

## Uncertainty (Protocol A, pooled held-out windows; non-PCT models use constant σ from validation residuals)

|                            |     NLL |   CRPS |   cov90 |   width90 |
|:---------------------------|--------:|-------:|--------:|----------:|
| ('PCT-E (ours)', '1 s')    | 23.1199 | 0.566  |  0.3346 |    0.6102 |
| ('PCT-E (ours)', '2 s')    | 40.7656 | 0.6932 |  0.3047 |    0.6304 |
| ('PCT-v2 (single)', '1 s') | 33.8964 | 0.5856 |  0.2885 |    0.4864 |
| ('PCT-v2 (single)', '2 s') | 85.6735 | 0.7121 |  0.2713 |    0.5384 |
| ('GBM', '1 s')             |  1.5632 | 0.4878 |  0.7916 |    2.2259 |
| ('GBM', '2 s')             |  1.9811 | 0.6749 |  0.7637 |    2.8349 |
| ('LSTM', '1 s')            |  2.3407 | 0.5133 |  0.7238 |    1.9844 |
| ('LSTM', '2 s')            |  2.923  | 0.6663 |  0.6696 |    2.2758 |
| ('GRU', '1 s')             |  2.3549 | 0.515  |  0.7219 |    1.937  |
| ('GRU', '2 s')             |  2.9381 | 0.6649 |  0.6662 |    2.1972 |
| ('TCN', '1 s')             |  2.3004 | 0.5129 |  0.7214 |    1.9365 |
| ('TCN', '2 s')             |  2.8964 | 0.6569 |  0.6653 |    2.1793 |
| ('Transformer', '1 s')     |  2.3059 | 0.4976 |  0.7251 |    1.9134 |
| ('Transformer', '2 s')     |  2.8709 | 0.6356 |  0.6696 |    2.1525 |
| ('Persistence', '1 s')     |  2.2859 | 0.5119 |  0.7394 |    2.0518 |
| ('Persistence', '2 s')     |  2.7893 | 0.6673 |  0.6857 |    2.3649 |

## Robustness (MAE gain +1 s, dB; mean over folds; seed-0 models)

**dropout**

| model           |   0.0 |   0.1 |   0.3 |   0.5 |   0.7 |
|:----------------|------:|------:|------:|------:|------:|
| GBM             | 0.554 | 0.554 | 0.553 | 0.554 | 0.557 |
| GRU             | 0.555 | 0.555 | 0.554 | 0.555 | 0.556 |
| LSTM            | 0.541 | 0.541 | 0.541 | 0.541 | 0.541 |
| PCT-E (ours)    | 0.521 | 0.521 | 0.522 | 0.523 | 0.526 |
| PCT-v2 (single) | 0.522 | 0.522 | 0.522 | 0.523 | 0.525 |
| Persistence     | 0.539 | 0.539 | 0.539 | 0.539 | 0.539 |
| TCN             | 0.566 | 0.566 | 0.565 | 0.565 | 0.565 |
| Transformer     | 0.54  | 0.54  | 0.54  | 0.539 | 0.54  |

**noise**

| model           |   0.0 |   0.1 |   0.25 |   0.5 |   1.0 |
|:----------------|------:|------:|-------:|------:|------:|
| GBM             | 0.554 | 0.554 |  0.554 | 0.554 | 0.557 |
| GRU             | 0.555 | 0.555 |  0.554 | 0.553 | 0.55  |
| LSTM            | 0.541 | 0.541 |  0.541 | 0.541 | 0.541 |
| PCT-E (ours)    | 0.521 | 0.521 |  0.521 | 0.522 | 0.525 |
| PCT-v2 (single) | 0.522 | 0.522 |  0.521 | 0.521 | 0.522 |
| Persistence     | 0.539 | 0.539 |  0.539 | 0.539 | 0.539 |
| TCN             | 0.566 | 0.566 |  0.566 | 0.566 | 0.565 |
| Transformer     | 0.54  | 0.54  |  0.54  | 0.54  | 0.541 |

**stale**

| model           |   0.0 |   0.5 |   1.0 |   2.0 |   4.0 |
|:----------------|------:|------:|------:|------:|------:|
| GBM             | 0.554 | 0.543 | 0.57  | 0.573 | 0.57  |
| GRU             | 0.555 | 0.549 | 0.555 | 0.558 | 0.555 |
| LSTM            | 0.541 | 0.541 | 0.541 | 0.541 | 0.541 |
| PCT-E (ours)    | 0.521 | 0.533 | 0.555 | 0.556 | 0.555 |
| PCT-v2 (single) | 0.522 | 0.529 | 0.546 | 0.55  | 0.546 |
| Persistence     | 0.539 | 0.539 | 0.539 | 0.539 | 0.539 |
| TCN             | 0.566 | 0.564 | 0.57  | 0.572 | 0.57  |
| Transformer     | 0.54  | 0.543 | 0.55  | 0.552 | 0.55  |

## Scalability (timing only)

| model       |    1 |    4 |   16 |   64 |   256 |   1024 |
|:------------|-----:|-----:|-----:|-----:|------:|-------:|
| GRU         | 1.2  | 1.83 | 2.87 | 4.78 | 10.39 |  32.27 |
| LSTM        | 0.53 | 0.45 | 0.67 | 1.11 |  3.35 |  21.76 |
| PCT         | 1.35 | 2.08 | 2.45 | 3.67 |  7.47 |  28.78 |
| TCN         | 0.81 | 0.96 | 1.56 | 2.06 |  8.24 |  29.67 |
| Transformer | 0.55 | 0.8  | 1.28 | 2.48 |  8.03 |  54.97 |

(inference ms for N nodes on 4 vCPU)


---

# Part II – Simulated hybrid acoustic–optical link: digital twin, controllers, ablations, scalability, robustness

> **Everything in Part II comes from a simulator** (`src/uwfc/linksim.py`). Only the acoustic channel-gain *dynamics* are real (measured recordings); link budgets, optical attenuation (with assumed advected turbidity plumes), power, energy, PDR, latency, reward, battery and 1 s feedback delay are **assumptions** (`docs/SIMULATION_ASSUMPTIONS.md`). Results describe behaviour inside this simulator, **not** real systems. Train sites: blue, red; validation: yellow; test: black, purple; `test_ood` = unseen turbidity (c 0.55–1.1 /m), distances (22–45 m, 2.2–3.5 km) and +6 dB noise.

## Digital-twin forecasting – test_id (mean ± std over 3 seeds; SNR errors in dB; h1 = decision step from 1 s-stale observations, h2 = +0.75 s; optical metrics only on active optical links, empty on OOD)

| model               | MAE_ac_h1     | MAE_ac_h2     | MAE_op_h1     | MAE_op_h2     | MAE_pdr_ac   | MAE_pdr_op   | MAE_log10ber_ac   | MAE_thr_op_kbps   |
|:--------------------|:--------------|:--------------|:--------------|:--------------|:-------------|:-------------|:------------------|:------------------|
| PIGT-DT ensemble x3 | 0.598         | 0.664         | 2.528         | 3.635         | 0.016        | 0.03         | 0.12              | 136.383           |
| PIGT-DT             | 0.6 ± 0.003   | 0.667 ± 0.006 | 2.537 ± 0.007 | 3.646 ± 0.009 | 0.016 ± 0.0  | 0.03 ± 0.0   | 0.12 ± 0.0        | 136.763 ± 0.912   |
| - no physics prior  | 0.602 ± 0.004 | 0.67 ± 0.005  | 2.544 ± 0.007 | 3.673 ± 0.007 | 0.016 ± 0.0  | 0.031 ± 0.0  | 0.12 ± 0.0        | 137.795 ± 0.753   |
| - no uncertainty    | 0.605 ± 0.002 | 0.675 ± 0.006 | 2.519 ± 0.012 | 3.618 ± 0.018 | 0.016 ± 0.0  | 0.03 ± 0.0   | 0.121 ± 0.001     | 135.163 ± 0.727   |
| LSTM                | 0.619 ± 0.002 | 0.688 ± 0.006 | 2.527 ± 0.015 | 3.644 ± 0.027 | 0.017 ± 0.0  | 0.03 ± 0.0   | 0.124 ± 0.0       | 135.875 ± 0.29    |
| - no graph          | 0.622 ± 0.002 | 0.69 ± 0.006  | 2.54 ± 0.016  | 3.661 ± 0.031 | 0.017 ± 0.0  | 0.03 ± 0.0   | 0.124 ± 0.001     | 135.991 ± 1.007   |
| GRU                 | 0.617 ± 0.001 | 0.691 ± 0.003 | 2.533 ± 0.008 | 3.657 ± 0.014 | 0.017 ± 0.0  | 0.03 ± 0.0   | 0.123 ± 0.001     | 136.076 ± 0.785   |
| Data-only           | 0.621 ± 0.002 | 0.694 ± 0.005 | 2.529 ± 0.006 | 3.655 ± 0.013 | 0.017 ± 0.0  | 0.03 ± 0.0   | 0.124 ± 0.0       | 136.055 ± 0.373   |
| Transformer         | 0.626 ± 0.002 | 0.695 ± 0.001 | 2.528 ± 0.007 | 3.648 ± 0.018 | 0.017 ± 0.0  | 0.031 ± 0.0  | 0.126 ± 0.001     | 137.503 ± 1.392   |
| TCN                 | 0.629 ± 0.005 | 0.699 ± 0.001 | 2.521 ± 0.007 | 3.64 ± 0.016  | 0.017 ± 0.0  | 0.031 ± 0.0  | 0.125 ± 0.001     | 137.415 ± 0.504   |
| GNN                 | 0.724 ± 0.002 | 0.78 ± 0.002  | 2.972 ± 0.001 | 4.333 ± 0.001 | 0.022 ± 0.0  | 0.036 ± 0.0  | 0.163 ± 0.0       | 159.634 ± 0.006   |
| - no temporal       | 0.783 ± 0.023 | 0.837 ± 0.027 | 2.935 ± 0.001 | 4.26 ± 0.003  | 0.022 ± 0.0  | 0.035 ± 0.0  | 0.162 ± 0.0       | 155.198 ± 0.103   |
| Physics-only        | 0.813         | 0.871         | 2.959         | 4.343         | 0.022        | 0.035        | 0.162             | 156.395           |

## Digital-twin forecasting – test_ood (mean ± std over 3 seeds; SNR errors in dB; h1 = decision step from 1 s-stale observations, h2 = +0.75 s; optical metrics only on active optical links, empty on OOD)

| model               | MAE_ac_h1     | MAE_ac_h2     |   MAE_op_h1 |   MAE_op_h2 | MAE_pdr_ac   |   MAE_pdr_op | MAE_log10ber_ac   |   MAE_thr_op_kbps |
|:--------------------|:--------------|:--------------|------------:|------------:|:-------------|-------------:|:------------------|------------------:|
| PIGT-DT ensemble x3 | 0.615         | 0.685         |         nan |         nan | 0.0          |          nan | 0.008             |               nan |
| PIGT-DT             | 0.617 ± 0.007 | 0.687 ± 0.005 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.008 ± 0.0       |               nan |
| - no physics prior  | 0.621 ± 0.003 | 0.695 ± 0.008 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.008 ± 0.0       |               nan |
| Transformer         | 0.637 ± 0.005 | 0.703 ± 0.005 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.009 ± 0.0       |               nan |
| LSTM                | 0.631 ± 0.004 | 0.703 ± 0.011 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.009 ± 0.0       |               nan |
| GRU                 | 0.631 ± 0.006 | 0.706 ± 0.011 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.009 ± 0.0       |               nan |
| - no uncertainty    | 0.63 ± 0.015  | 0.71 ± 0.021  |         nan |         nan | 0.0 ± 0.0    |          nan | 0.009 ± 0.0       |               nan |
| Data-only           | 0.638 ± 0.006 | 0.714 ± 0.01  |         nan |         nan | 0.0 ± 0.0    |          nan | 0.009 ± 0.0       |               nan |
| - no graph          | 0.65 ± 0.019  | 0.72 ± 0.026  |         nan |         nan | 0.0 ± 0.0    |          nan | 0.009 ± 0.001     |               nan |
| TCN                 | 0.656 ± 0.019 | 0.733 ± 0.026 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.009 ± 0.001     |               nan |
| Physics-only        | 0.831         | 0.898         |         nan |         nan | 0.0          |          nan | 0.011             |               nan |
| - no temporal       | 0.932 ± 0.065 | 1.053 ± 0.084 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.014 ± 0.001     |               nan |
| GNN                 | 1.003 ± 0.027 | 1.164 ± 0.031 |         nan |         nan | 0.0 ± 0.0    |          nan | 0.012 ± 0.0       |               nan |

Ensemble size (test_id):

| model               |   MAE_ac_h1 |   MAE_ac_h2 |   MAE_op_h1 |   MAE_op_h2 |   NLL |   CRPS |   cov90 |
|:--------------------|------------:|------------:|------------:|------------:|------:|-------:|--------:|
| PIGT-DT ensemble x1 |       0.597 |       0.662 |       2.534 |       3.639 | 1.727 |  1.012 |   0.94  |
| PIGT-DT ensemble x2 |       0.598 |       0.665 |       2.535 |       3.64  | 1.734 |  1.015 |   0.945 |
| PIGT-DT ensemble x5 |       0.599 |       0.664 |       2.525 |       3.63  | 1.729 |  1.011 |   0.942 |

## Digital-twin uncertainty (PIGT-DT ensemble vs constant-σ baselines)

|                                       |   NLL |   CRPS |   cov90 |   width90 |   mean_sigma |
|:--------------------------------------|------:|-------:|--------:|----------:|-------------:|
| ('PIGT-DT ensemble x3', 'test_id')    | 1.731 |  1.013 |   0.944 |     6.575 |        1.999 |
| ('PIGT-DT ensemble x3', 'test_ood')   | 1.359 |  0.494 |   0.972 |     4.03  |        1.225 |
| ('PIGT-DT single member', 'test_id')  | 1.727 |  1.012 |   0.94  |     6.452 |        1.961 |
| ('PIGT-DT single member', 'test_ood') | 1.385 |  0.5   |   0.977 |     4.219 |        1.282 |
| ('LSTM (const. sigma)', 'test_id')    | 1.977 |  1.068 |   0.804 |     4.013 |        1.22  |
| ('LSTM (const. sigma)', 'test_ood')   | 1.316 |  0.48  |   0.832 |     2.314 |        0.703 |
| ('TCN (const. sigma)', 'test_id')     | 1.985 |  1.069 |   0.804 |     4.021 |        1.222 |
| ('TCN (const. sigma)', 'test_ood')    | 1.356 |  0.497 |   0.826 |     2.356 |        0.716 |

## Controllers – test_id (64 episodes × 8 nodes; ± = std over 3 training seeds for MAPPO variants only; goodput kbit/s, energy J/step)

| controller               | reward        | goodput           | pdr          | ber       | energy        | latency       | viol          | switch_rate   |
|:-------------------------|:--------------|:------------------|:-------------|:----------|:--------------|:--------------|:--------------|:--------------|
| Oracle                   | 1.208         | 878.758           | 0.839        | 0.019     | 0.787         | 0.27          | 0.162         | 0.026         |
| MAPPO+DT                 | 1.168 ± 0.121 | 785.739 ± 105.844 | 0.88 ± 0.008 | 0.0 ± 0.0 | 0.742 ± 0.02  | 0.349 ± 0.004 | 0.132 ± 0.012 | 0.011 ± 0.005 |
| PIGT-DT                  | 1.14          | 857.052           | 0.834        | 0.019     | 0.784         | 0.271         | 0.177         | 0.032         |
| PIGT-DT (uncertainty MC) | 1.118         | 859.729           | 0.829        | 0.02      | 0.804         | 0.266         | 0.182         | 0.041         |
| Reactive                 | 1.118         | 851.959           | 0.832        | 0.019     | 0.791         | 0.271         | 0.179         | 0.045         |
| Heuristic                | 1.081         | 749.85            | 0.848        | 0.001     | 0.705         | 0.348         | 0.163         | 0.029         |
| MAPPO                    | 1.07 ± 0.096  | 750.273 ± 108.341 | 0.868 ± 0.01 | 0.0 ± 0.0 | 0.758 ± 0.009 | 0.349 ± 0.005 | 0.148 ± 0.015 | 0.007 ± 0.004 |
| PIGT-DT (risk-averse)    | 1.058         | 829.087           | 0.833        | 0.006     | 0.817         | 0.32          | 0.174         | 0.05          |
| HYB-low                  | 0.238         | 666.812           | 0.754        | 0.003     | 1.125         | 0.352         | 0.276         | 0.0           |
| AC-only                  | -1.392        | 1.771             | 0.443        | 0.0       | 1.111         | 0.374         | 0.559         | 0.0           |
| OP-only                  | -1.436        | 521.907           | 0.116        | 0.198     | 1.111         | 0.0           | 0.89          | 0.0           |

## Controllers – test_ood (64 episodes × 8 nodes; ± = std over 3 training seeds for MAPPO variants only; goodput kbit/s, energy J/step)

| controller               | reward       | goodput   | pdr       | ber         | energy      | latency     | viol      | switch_rate   |
|:-------------------------|:-------------|:----------|:----------|:------------|:------------|:------------|:----------|:--------------|
| PIGT-DT (risk-averse)    | -0.768       | 2.0       | 0.5       | 0.145       | 0.625       | 0.075       | 0.5       | 0.0           |
| PIGT-DT (uncertainty MC) | -0.768       | 2.0       | 0.5       | 0.145       | 0.625       | 0.075       | 0.5       | 0.0           |
| Oracle                   | -0.768       | 2.0       | 0.5       | 0.145       | 0.625       | 0.075       | 0.5       | 0.0           |
| Reactive                 | -0.768       | 2.0       | 0.5       | 0.145       | 0.625       | 0.075       | 0.5       | 0.0           |
| PIGT-DT                  | -0.768       | 2.0       | 0.5       | 0.145       | 0.625       | 0.075       | 0.5       | 0.0           |
| Heuristic                | -1.754       | 2.0       | 0.5       | 0.073       | 0.806       | 1.085       | 0.5       | 0.0           |
| MAPPO                    | -1.754 ± 0.0 | 2.0 ± 0.0 | 0.5 ± 0.0 | 0.073 ± 0.0 | 0.806 ± 0.0 | 1.085 ± 0.0 | 0.5 ± 0.0 | 0.0 ± 0.0     |
| MAPPO+DT                 | -1.754 ± 0.0 | 2.0 ± 0.0 | 0.5 ± 0.0 | 0.073 ± 0.0 | 0.806 ± 0.0 | 1.085 ± 0.0 | 0.5 ± 0.0 | 0.0 ± 0.0     |
| OP-only                  | -2.559       | 0.0       | 0.0       | 0.287       | 1.111       | 0.0         | 1.0       | 0.0           |
| AC-only                  | -2.768       | 0.889     | 0.222     | 0.073       | 1.111       | 1.085       | 0.778     | 0.0           |
| HYB-low                  | -2.991       | 1.8       | 0.45      | 0.141       | 1.125       | 1.085       | 0.55      | 0.0           |

### PIGT-DT controller vs others – paired bootstrap over episodes

| testset   | PIGT-DT minus   |   Δ reward | 95% CI (paired bootstrap over 64 episodes)   | verdict        |
|:----------|:----------------|-----------:|:---------------------------------------------|:---------------|
| test_id   | Reactive        |     0.0215 | [+0.0028, +0.0416]                           | PIGT-DT better |
| test_id   | Heuristic       |     0.0591 | [+0.0179, +0.0989]                           | PIGT-DT better |
| test_id   | MAPPO           |     0.0696 | [+0.0245, +0.1120]                           | PIGT-DT better |
| test_id   | MAPPO+DT        |    -0.0279 | [-0.0768, +0.0125]                           | inconclusive   |
| test_id   | Oracle          |    -0.0687 | [-0.0912, -0.0471]                           | PIGT-DT worse  |
| test_ood  | Reactive        |     0      | [+0.0000, +0.0000]                           | inconclusive   |
| test_ood  | Heuristic       |     0.986  | [+0.9034, +1.0675]                           | PIGT-DT better |
| test_ood  | MAPPO           |     0.986  | [+0.9027, +1.0690]                           | PIGT-DT better |
| test_ood  | MAPPO+DT        |     0.986  | [+0.9028, +1.0668]                           | PIGT-DT better |
| test_ood  | Oracle          |     0      | [+0.0000, +0.0000]                           | inconclusive   |

## Controller-level ablations / stress tests (DT-greedy controller, test_id unless noted)

**component**

| variant                    |   reward |   goodput |   pdr |   energy |   latency |   viol |
|:---------------------------|---------:|----------:|------:|---------:|----------:|-------:|
| PIGT-DT (full)             |    1.14  |   857.052 | 0.834 |    0.784 |     0.271 |  0.177 |
| Reactive (no DT)           |    1.118 |   851.959 | 0.832 |    0.791 |     0.271 |  0.179 |
| Oracle                     |    1.208 |   878.758 | 0.839 |    0.787 |     0.27  |  0.162 |
| - no physics prior         |    1.143 |   858.332 | 0.834 |    0.784 |     0.27  |  0.176 |
| - no graph                 |    1.145 |   858.602 | 0.835 |    0.78  |     0.27  |  0.177 |
| - no temporal              |    1.127 |   855.948 | 0.834 |    0.789 |     0.273 |  0.178 |
| - no uncertainty           |    1.142 |   858.483 | 0.834 |    0.782 |     0.271 |  0.177 |
| + uncertainty MC (full DT) |    1.118 |   859.729 | 0.829 |    0.804 |     0.266 |  0.182 |

**risk**

| variant                                             |   reward |   goodput |   pdr |   energy |   latency |   viol |
|:----------------------------------------------------|---------:|----------:|------:|---------:|----------:|-------:|
| with risk constraint                                |    1.14  |   857.052 | 0.834 |    0.784 |     0.271 |  0.177 |
| without risk constraint                             |    1.188 |   916.342 | 0.769 |    0.679 |     0.162 |  0.253 |
| risk-averse (+50% violation weight, uncertainty MC) |    1.058 |   829.087 | 0.833 |    0.817 |     0.32  |  0.174 |

**ensemble**

| variant   |   reward |   goodput |   pdr |   energy |   latency |   viol |
|:----------|---------:|----------:|------:|---------:|----------:|-------:|
| M=1       |    1.143 |   858.244 | 0.835 |    0.785 |     0.271 |  0.176 |
| M=2       |    1.143 |   858.898 | 0.835 |    0.783 |     0.271 |  0.176 |
| M=3       |    1.14  |   857.052 | 0.834 |    0.784 |     0.271 |  0.177 |
| M=5       |    1.141 |   857.2   | 0.834 |    0.782 |     0.271 |  0.177 |

**real_sim_ratio**

| variant          |   reward |   goodput |   pdr |   energy |   latency |   viol |
|:-----------------|---------:|----------:|------:|---------:|----------:|-------:|
| 100% real traces |    1.143 |   858.898 | 0.835 |    0.783 |     0.271 |  0.176 |
| 50% real traces  |    1.147 |   861.628 | 0.835 |    0.783 |     0.271 |  0.176 |
| 0% real traces   |    1.145 |   858.925 | 0.835 |    0.783 |     0.27  |  0.176 |

**mismatch**

| variant   |   nominal |   SL -3 dB |   SL +3 dB |   absorption x1.5 |   noise +6 dB |   optical c x1.5 |   optical c x0.7 |
|:----------|----------:|-----------:|-----------:|------------------:|--------------:|-----------------:|-----------------:|
| PIGT-DT   |     1.14  |      1.003 |      1.249 |             1.113 |         0.813 |            0.831 |            1.563 |
| Reactive  |     1.118 |      0.975 |      1.218 |             1.088 |         0.783 |            0.807 |            1.534 |
| Oracle    |     1.208 |      1.075 |      1.31  |             1.182 |         0.887 |            0.868 |            1.636 |

**staleness**

| variant   |   0.25 |   1.0 |   2.0 |   3.0 |   4.0 |
|:----------|-------:|------:|------:|------:|------:|
| PIGT-DT   |  1.158 | 1.14  | 1.083 | 1.031 | 0.984 |
| Reactive  |  1.171 | 1.118 | 1.056 | 1.003 | 0.963 |
| Oracle    |  1.209 | 1.208 | 1.201 | 1.194 | 1.188 |

**node_failure**

| variant         |   0.0 |   0.125 |   0.25 |   0.5 |
|:----------------|------:|--------:|-------:|------:|
| PIGT-DT (graph) | 1.14  |   1.054 |  1.107 | 1.129 |
| no graph        | 1.145 |   1.06  |  1.112 | 1.138 |
| Reactive        | 1.118 |   1.032 |  1.079 | 1.104 |

## Robustness of controllers (reward)

**sensor_dropout**

| controller   |   0.0 |   0.1 |   0.3 |   0.5 |
|:-------------|------:|------:|------:|------:|
| Heuristic    |  0.93 |  0.93 |  0.93 |  0.92 |
| Reactive     |  1.32 |  1.32 |  1.31 |  1.31 |
| PIGT-DT      |  1.32 |  1.33 |  1.32 |  1.3  |
| MAPPO        |  1.15 |  1.14 |  1.14 |  1.14 |
| MAPPO+DT     |  1.44 |  1.43 |  1.43 |  1.42 |
| Oracle       |  1.38 |  1.38 |  1.38 |  1.38 |

**noisy_observations**

| controller   |   0.0 |   1.0 |   2.0 |   4.0 |
|:-------------|------:|------:|------:|------:|
| Heuristic    |  0.93 |  0.9  |  0.87 |  0.79 |
| Reactive     |  1.32 |  1.27 |  1.21 |  1.11 |
| PIGT-DT      |  1.32 |  1.29 |  1.27 |  1.19 |
| MAPPO        |  1.15 |  1.16 |  1.15 |  1.11 |
| MAPPO+DT     |  1.44 |  1.42 |  1.39 |  1.32 |
| Oracle       |  1.38 |  1.38 |  1.38 |  1.38 |

**stale_dt**

| controller   |   0.25 |   1.0 |   2.0 |   3.0 |   4.0 |
|:-------------|-------:|------:|------:|------:|------:|
| Heuristic    |   0.95 |  0.93 |  0.89 |  0.85 |  0.82 |
| Reactive     |   1.36 |  1.32 |  1.25 |  1.2  |  1.16 |
| PIGT-DT      |   1.33 |  1.32 |  1.26 |  1.2  |  1.16 |
| MAPPO        |   1.14 |  1.15 |  1.12 |  1.1  |  1.09 |
| MAPPO+DT     |   1.41 |  1.44 |  1.38 |  1.31 |  1.25 |
| Oracle       |   1.39 |  1.38 |  1.38 |  1.38 |  1.37 |

**channel_model_mismatch**

| controller   |   nominal |   SL -3 dB |   SL +3 dB |   absorption x1.5 |   optical c x1.5 |   optical c x0.7 |
|:-------------|----------:|-----------:|-----------:|------------------:|-----------------:|-----------------:|
| Heuristic    |      0.93 |       0.71 |       1.06 |              0.89 |             0.51 |             1.34 |
| Reactive     |      1.32 |       1.17 |       1.42 |              1.29 |             0.88 |             1.72 |
| PIGT-DT      |      1.32 |       1.18 |       1.43 |              1.29 |             0.9  |             1.74 |
| MAPPO        |      1.15 |       0.97 |       1.26 |              1.12 |             0.8  |             1.52 |
| MAPPO+DT     |      1.44 |       1.26 |       1.54 |              1.4  |             0.98 |             1.86 |
| Oracle       |      1.38 |       1.25 |       1.48 |              1.36 |             0.96 |             1.8  |

**environment_mismatch**

| controller   |   nominal |   noise +6 dB |   noise -6 dB |
|:-------------|----------:|--------------:|--------------:|
| Heuristic    |      0.93 |          0.44 |          1.16 |
| Reactive     |      1.32 |          0.97 |          1.51 |
| PIGT-DT      |      1.32 |          0.98 |          1.51 |
| MAPPO        |      1.15 |          0.69 |          1.35 |
| MAPPO+DT     |      1.44 |          1.01 |          1.6  |
| Oracle       |      1.38 |          1.06 |          1.56 |

**unseen_distances**

| controller   |   seen (4-22 m / 150-1800 m) |   unseen near 22-45 m |   unseen far 2200-3500 m |
|:-------------|-----------------------------:|----------------------:|-------------------------:|
| Heuristic    |                         0.93 |                  0.09 |                    -0.95 |
| Reactive     |                         1.32 |                  0.2  |                     0.29 |
| PIGT-DT      |                         1.32 |                  0.2  |                     0.29 |
| MAPPO        |                         1.15 |                  0.19 |                    -0.82 |
| MAPPO+DT     |                         1.44 |                  0.21 |                    -0.56 |
| Oracle       |                         1.38 |                  0.22 |                     0.34 |

**unseen_turbidity**

| controller   |   seen turbidity (c 0.15-0.41) |   unseen clearer (c 0.08-0.15) |   unseen turbid (c 0.55-1.1) |
|:-------------|-------------------------------:|-------------------------------:|-----------------------------:|
| Heuristic    |                           0.93 |                           1.7  |                         0.17 |
| Reactive     |                           1.32 |                           2.21 |                         0.37 |
| PIGT-DT      |                           1.32 |                           2.23 |                         0.36 |
| MAPPO        |                           1.15 |                           1.92 |                         0.34 |
| MAPPO+DT     |                           1.44 |                           2.34 |                         0.41 |
| Oracle       |                           1.38 |                           2.3  |                         0.41 |

**unseen_acoustic_site**

| controller   |   black |   purple |   yellow |   blue |   red |
|:-------------|--------:|---------:|---------:|-------:|------:|
| Heuristic    |    0.85 |     0.74 |     0.85 |   0.75 |  0.74 |
| Reactive     |    1.24 |     1.16 |     1.23 |   1.16 |  1.16 |
| PIGT-DT      |    1.26 |     1.2  |     1.26 |   1.2  |  1.2  |
| MAPPO        |    1.1  |     1.04 |     1.1  |   1.03 |  1.05 |
| MAPPO+DT     |    1.36 |     1.31 |     1.36 |   1.3  |  1.31 |
| Oracle       |    1.31 |     1.26 |     1.31 |   1.25 |  1.26 |

## Scalability (CPU, 4 vCPU, no GPU)

|   N |   dt_infer_ms |   dt_infer_ms_per_node |   dt_ensemble3_ms |   dt_mem_MB |   dt_train_s_per_epoch |
|----:|--------------:|-----------------------:|------------------:|------------:|-----------------------:|
|   4 |         1.59  |                  0.397 |             4.77  |       0.408 |                  0.702 |
|   8 |         2.58  |                  0.322 |             7.739 |       0.815 |                  1.3   |
|  16 |         2.589 |                  0.162 |             7.768 |       1.631 |                  2.71  |
|  32 |         2.467 |                  0.077 |             7.4   |       3.261 |                  4.749 |
|  64 |         3.05  |                  0.048 |             9.151 |       6.522 |                 11.72  |
| 128 |         3.55  |                  0.028 |            10.65  |      13.045 |                 29.272 |

| controller   |    4 |    8 |    16 |    32 |    64 |    128 |
|:-------------|-----:|-----:|------:|------:|------:|-------:|
| Heuristic    | 2638 | 5865 | 13384 | 24043 | 41916 |  84568 |
| MAPPO        | 2496 | 5249 | 13039 | 22740 | 38675 |  80439 |
| MAPPO+DT     | 2996 | 6611 | 16798 | 29347 | 50846 | 103904 |
| Oracle       | 3040 | 6405 | 16381 | 29060 | 50571 | 103687 |
| PIGT-DT      | 3014 | 6339 | 15802 | 28539 | 49655 | 101244 |
| Reactive     | 2970 | 6320 | 15833 | 28050 | 48804 | 100013 |
(network goodput, kbit/s)

## MAPPO training (last 20 iterations of the training batch, mean over seeds)

|          |   reward |   pdr |   goodput |   energy |   viol |
|:---------|---------:|------:|----------:|---------:|-------:|
| MAPPO    |    1.133 | 0.885 |   750.709 |    0.747 |  0.132 |
| MAPPO+DT |    1.217 | 0.895 |   780.664 |    0.733 |  0.119 |

## Verdict (computed from the numbers above)

- **Protocol A**: PCT-E ranks 1/13 at +1 s and 1/13 at +2 s by mean MAE (the list includes PCT variants and shrink controls).
  - +1 s vs Transformer: Δ=+0.0044 dB, 95% CI [-0.0074, +0.0156] → inconclusive (CI includes 0)
  - +1 s vs Persistence: Δ=+0.0199 dB, 95% CI [+0.0059, +0.0341] → **PCT-E significantly better**
  - +2 s vs Transformer: Δ=+0.0362 dB, 95% CI [+0.0209, +0.0513] → **PCT-E significantly better**
  - +2 s vs Persistence: Δ=+0.0747 dB, 95% CI [+0.0534, +0.0955] → **PCT-E significantly better**
- **Protocol B**: PCT-E ranks 4/13 at +1 s and 2/13 at +2 s by mean MAE (the list includes PCT variants and shrink controls).
  - +1 s vs GBM: Δ=-0.0130 dB, 95% CI [-0.0326, +0.0055] → inconclusive (CI includes 0)
  - +1 s vs Persistence: Δ=+0.0799 dB, 95% CI [+0.0363, +0.1287] → **PCT-E significantly better**
  - +2 s vs GBM: Δ=+0.0091 dB, 95% CI [-0.0356, +0.0488] → inconclusive (CI includes 0)
  - +2 s vs Persistence: Δ=+0.1277 dB, 95% CI [+0.0477, +0.2126] → **PCT-E significantly better**

See `docs/LIMITATIONS.md` for caveats (5 sites, persistence-dominated task, validation reuse, scale-normalisation pilot).
