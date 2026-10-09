# Analysis of five RSNA-Knee Kaggle notebooks

Scope: the five notebooks supplied by the owner. Everything below was read from the notebook source and stored outputs; nothing was re-run (no GPU, no competition data here). "Reported" = stated in a notebook's text; "measured in notebook" = printed in a stored output; "inferred" = my reading of the code.

| # | File | What it is |
|---|------|------------|
| 1 | `rsna-knee-abnormalities-efficiency-lb` | Efficiency-Prize leaderboard table (top 100) |
| 2 | `rsna-knee-blend-gold` | Raptor (public) 0.7 + private compact ViT-L model 0.3, rank blend |
| 3 | `rsna-versia-5` | Narrated compilation of six public works (0.937 -> 0.950) + the 0.950 code |
| 4 | `rsna-knee-v2-velciraptor-dinosaur-speed` | The 0.950 code itself: CoAtNet "Raptor" + ConvNeXt MIL, rank fusion |
| 5 | `rsna-knee-dinosaur-v5` | Notebook 4 + an unsupervised reader-weight experiment ("V8") |

**Key structural finding.** Notebooks 3, 4 and 5 share the same code cells (same sizes: 1767 / 96383 / ~160 / 6469 / 8356 / 3670 / 7450 characters). Notebook 3 = notebook 4 + markdown; notebook 5 = notebook 4 + one extra cell. Notebook 2 also embeds the same Raptor script (same checkpoint SHA-256 `7e5315da...72244ef`). So there is effectively **one model family (Raptor CoAtNet) and one second reader (ConvNeXt MIL)**; the notebooks differ only in blending.

---

## 1. Task and metric (common to all)

* 12 study-level findings from knee MRI: ACL, MCL, medial/lateral meniscus, medial/lateral/PF OA, effusion, synovitis, Baker's cyst, contusion, fracture.
* Metric: macro ROC-AUC over 12 labels. AUC depends only on within-column ordering, which is why every notebook ends with per-column **percentile ranks**.
* Training supervision (reported in notebook 4's docstring): the competition gives 4,407 studies but structured labels for only **58** ("gold"). Labels for the other 4,349 studies come from a language model reading radiology reports, turned into 12 **soft probabilities** (suspected tear ≈ 0.8). The 58 gold studies are described as "never trained on" for the original CoAtNet (0.9167 AUC reported on them).

## 2. Notebook 1: Efficiency leaderboard (data)

* 100 rows: `TeamName`, `PublicScore`, `DateSubmitted` (27 Sep – 7 Oct 2026), indexed by `EfficiencyRank`.
* Measured from the table: min 0.929, median 0.948, mean 0.946, max 0.962 (rank 2). 47 of 100 teams ≥ 0.949 and **21 teams sit at exactly 0.949**, the Velociraptor score, i.e. many near-copies of notebook 4.
* Rank is by efficiency, not score: the best score (0.962) is rank 2, the rank-1 team has 0.959. Efficiency ranking trades AUC against compute; the notebook does not expose the compute figures, so the trade-off cannot be reconstructed from this file.
* Note: this table (median 0.948) differs from the 0.916–0.958 / median 0.940 quoted in the repo README §5.8. That section should be refreshed.

## 3. The Raptor CoAtNet pipeline (notebooks 2–5), step by step

1. **Guard rails.** Requires CUDA; finds the checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` in four candidate paths; asserts SHA-256. No substitute weights.
2. **Series selection into 5 fixed slots** totalling 96 slices: Sagittal fluid-sensitive 26, Sagittal non-fluid 22, Coronal fluid 18, Coronal non-fluid 12, Axial 18. A series is used once; if the preferred fluid flag is missing, falls back to any series of that plane; absent slot = zero slices (masked).
3. **Slice sampling.** Within a series, `linspace` over 2 %–98 % of the ordered stack (wide span because collateral ligaments / lateral meniscus sit peripherally). Order by projecting ImagePositionPatient on the slice normal.
4. **Intensity and geometry.** Per-series 2nd–98th percentile clip to [0,1]; centre crop of **140 mm** using DICOM pixel spacing (scale-invariant), `INTER_AREA` resize to 384 px; uint8.
5. **Windows.** 94 (`K_EVAL`) evenly spaced centres across the 96-slice volume; each window = 3 adjacent slices as RGB channels (2.5D); centre-cropped 384 -> 320 (`INPUT_RES`), ImageNet normalised. (Docstring still says "42 windows / 0.924 LB" — stale text.)
6. **Backbone.** timm `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, average pooled, run in chunks of 16 windows.
7. **Head (`RaptorClassifier`).** LayerNorm -> per-class attention scorer (Linear 768->256, tanh, dropout 0.2, Linear 256->12) -> softmax **over windows, separately for each finding** -> pooled feature · class weight + bias = 12 logits. This is attention-MIL with finding-specific attention, the idea the notebook credits as most important.
8. **Anatomical mirror TTA.** Mirror the knee: sagittal windows reverse the 3 slice channels, coronal/axial flip the width; mean of plain and mirrored sigmoid probabilities. Deliberately label-preserving (no naive flip that would swap medial/lateral semantics in a wrong plane).
9. **Inference engineering.** fp16 autocast (T4 has no bf16 conv engines for these shapes), non-finite assert, queue between a process pool (preprocessing) and one thread per GPU, 2×T4. Base64-embedded `_fastread.py` (49 KB decoded): custom explicit/implicit-VR-LE DICOM walker, batched I/O, `study_lut` variant (exact percentiles from raw-code histograms, no float image materialised), falling back to pydicom for anything unusual; claimed byte-identical to the reference reader.
10. **Failure handling.** A study that errors keeps the initial 0.5 in all 12 columns (silent fallback, ties).
11. **Output.** Column-wise percentile rank -> `submission.csv` plus `_sota_0949.csv` backup.

### Reported gains for the CoAtNet arm (notebook 4 comments)

* Single arm replaced a 3-arm blend because on the live LB "CoAtNet alone 0.914, blends 0.914–0.915": the +0.010 blend gain seen on the old **45-study gold set was noise**.
* Corpus expanded 3,200 -> 4,349 studies; on the 58-study gate 0.8923 -> 0.9054 (+0.0131, better in 92.7 % of 2,000 bootstraps).
* Final checkpoint adds SWA, OAI data ("oai" in filename), 384 px, d96 stack, K=94, mirror TTA, and reaches LB ≈ 0.949.

## 4. Second reader: ConvNeXt 2.5D MIL (notebooks 4, 5)

* Preprocessing (`preprocess.py`): canonical orientation from ImageOrientationPatient, 0.4 mm/px, 384×384 (153.6 mm FOV), ≤ 64 slices, 1–99.5 percentile normalisation.
* Slots (`knee.py`): 6 = plane (3) × fat-suppressed or not; k windows per slot (k=16 at inference), slice centres from 4 %–96 %.
* `KneeNet`: timm ConvNeXt (checkpoint-defined, default `convnext_tiny`) on 3-slice windows -> Linear to 384 + LayerNorm + slot embedding + MLP slice-position embedding -> 2-layer transformer over all windows -> per-finding attention pooling. Padding masks for absent slots. Three fold checkpoints `cnxt_v0_fold{0,1,2}.pt`, mean of sigmoids. Reported run time 30 s on the 3-study demo.
* Training is **not** in these notebooks; the weights come from another author's dataset, so its training split and label provenance cannot be audited here.

## 5. Fusion logic

**Notebook 4 / 3 (reported 0.950).** Per-finding rank fusion `(1-w)·rank(CoAtNet) + w·rank(ConvNeXt)`, then re-rank:

| Finding | w(ConvNeXt) | | Finding | w |
|---|---|---|---|---|
| Baker's | 0.35 | | Lateral Meniscus | 0.08 |
| Contusion | 0.30 | | Fracture | 0.08 |
| Medial OA | 0.25 | | Lateral OA | 0.08 |
| ACL | 0.25 | | PF OA | 0.08 |
| Medial Meniscus | 0.20 | | Effusion | 0.04 |
| MCL | 0.20 | | Synovitis | 0.03 | 

Justification in comments is per-finding AUC on the 58 gold studies (e.g. Baker's 0.976 vs 0.940), i.e. **weights selected using the same 58 studies that the CoAtNet checkpoint was also selected on**. Fallback: any exception restores the CoAtNet-only submission.

**Notebook 5 (Dino v5).** Starts from "V6" weights that differ from the above for five findings (Lateral OA .18, Baker's .39, Contusion .33, ACL .27, MCL .22). Then, for those five findings only, a label-free adjustment: `agreement = 1 − clip(mean std of the 3 ConvNeXt fold ranks / 0.22)`, `diversity = clip((0.995 − Spearman(CoAtNet, ConvNeXt))/0.25)`, `boost = clip(0.6(agr−0.7)+0.42(div−0.55), ±0.16)`, `w' = clip(w(1+boost), 0.01, 0.49)`; result blended 80/20 with the V6 route and reverted per finding if more than 2.5 % of pairs flip rank. Skipped when fewer than 25 test studies. The notebook itself says this is an **experiment, not a claim of improvement**. Agreement and decorrelation are not evidence of higher AUC; the effect is bounded small by construction.

**Notebook 2 (Blend-Gold).** Raptor 0.7 + private model 0.3, weights "fixed before any run, not fitted". Failed-study handling (take the other model's rank, 0.5 if both fail), abort if > 3 % Raptor failures, Spearman consistency check between Raptor raw and submitted ranks.

## 6. Private compact model (notebook 2, `compact_train.py` / `compact_infer.py`)

* Encoder: OrthoFoundation-L = timm `vit_large_patch16_dinov3_qkvb` (self-supervised public weights), RoPE removed; patch embed + first 16 of 24 blocks **frozen**, last 8 + norm trained; feature = [CLS, mean patch] (2048); input 256 px crops, 3 physically adjacent slices as channels.
* Views: full field plus 100 mm local views (PF-local, posterior-local, bilateral coronal locals), anchored on a detected knee centre (SKM-TEA anatomy checkpoint), plus an orientation/laterality classifier that re-renders mirrored studies.
* Head: projection to 384 + slot/view/geometry embeddings -> per-finding window attention -> 12 logits. Optional **report-state heads** (categorical state per finding, gold-calibrated prior, gated residual added to the logit).
* Loss (arm `E_lrprior`): likelihood-ratio loss `−log(p·lr + (1−p))`, a marginal likelihood given per-study label likelihood ratios from reports, rather than hard/soft BCE.
* Protocol (code-level): K=5 hash folds over the non-gold pool, gold excluded from all training (asserted), OOF + gold predictions, `--holdout-fold` for leak-free pseudo-label rounds, layer-wise LR decay option, EMA option. Stored run: 3 test studies, 95 s on 2×T4, ~529 windows/study, 35 s decode + 19 s GPU per study on average (measured in notebook), so full-test cost is the dominant risk; deadline 8.3 h.
* Provenance checks: SHA-256 for every source file, bundle and encoder-weight part.
* No AUC for this model is printed anywhere in the notebook.

## 7. Evaluation of the notebooks (critique)

**Strengths**
* Real engineering: 2.5D windows, finding-specific attention, mm-based cropping (scale-invariant), anatomically correct mirror TTA, fixed slot layout with masks.
* Reproducibility hygiene: hashes, deterministic fallback to a known submission, assertions, deadline guards.
* Cheap diversity: fusing two different families (CoAtNet at 384 px vs ConvNeXt at ~256–320 px) with per-finding weights.

**Weaknesses and risks**
1. **Selection on 58 gold studies.** Checkpoint, blend weights and arm choices were selected on them; the notebook admits the earlier +0.010 gain was noise. Treat all gold numbers as optimistic; with n = 58 the AUC CI is about ±0.06 (my estimate, consistent with the repo README).
2. **Leaderboard-driven tuning.** Public-LB scores are used as the success criterion; 21 of the top 100 sit at the identical 0.949, so the leaderboard mostly measures who copied which public notebook. Private-LB shake-up is likely for micro-weight changes (notebook 5 acknowledges this).
3. **Label-probing mechanism.** Notebook 2 contains `PROBE_CLASS`: setting 11 columns to 0.5 so public LB = (11·0.5 + AUC_c)/12 to read out one finding's public AUC. It is switched off (`None`), but it is a way to extract hidden-label information from the leaderboard and should not be copied.
4. **Training is external and unauditable.** Raptor and ConvNeXt weights come from dataset authors; label construction (LLM-labelled reports plus OAI data) is described only in prose. Possible train/test overlap (OAI-derived data) cannot be checked here.
5. **Silent failure value 0.5**, relative-rank outputs (scores depend on the test set composition), and tiny test demo (3 studies) mean the stored outputs prove only that code runs, not accuracy.
6. **Inconsistencies:** stale docstring (0.924 / 42 windows vs K=94 / 0.949); notebooks 3/4 use weights "V3" while notebook 5 calls its base the "0.950 V6" with different weights; `DataParallel` removed after OOM history; two different "Raptor" scripts referenced (hash `bcd864cb...` in notebook 2).
7. **Compute:** CoAtNet-RMLP-2 at 320 px × 94 windows × 2 (mirror) per study plus a ConvNeXt reader needs minutes on 2×T4; poor for the efficiency track unless efficiency is scored on wall-clock only.

## 8. What matters for this repository (MV-MoR)

The gap to the notebooks (README: gold AUC 0.713 vs LB ≈ 0.95) is explained by three things visible above: (a) their labels are LLM/soft and expanded with OAI data while ours reach only 0.735 AUC vs gold; (b) they fine-tune the image backbone end to end at 320–384 px with 94 windows per study, while ours uses frozen features and a 0.35–0.75 M-parameter head; (c) they use finding-specific attention and per-finding fusion. Directly reusable and cheap to test: finding-specific window attention (already in MV-MoR heads, so test per-finding routing), 2.5D windows, mirror TTA, slot-based series selection with masks. Not reusable: gold-set-driven weight selection.

## 9. Open items I did not verify

* The exact test-set size and full-run timing (notebooks ran 3 studies).
* The ConvNeXt backbone variant inside the checkpoints (read from `ck["args"]` at runtime).
* The unread tails of `compact_train.py` main loop, `probe_infer.py`, `probe_dual.py` and `knee_anchor_bank_v2.py` beyond their headers and arguments.
