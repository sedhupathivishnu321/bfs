# Literature review and research gap

Scope: automatic detection of knee abnormalities on multi-sequence, multi-plane MRI; weak supervision from
radiology reports; efficient multi-view aggregation. The summaries below are from the cited papers. Numbers
from other papers were measured on other datasets and are **not** comparable to ours.

## 1. Knee MRI classification

| Work | Data / task | Method | Relevance / limitation |
|---|---|---|---|
| Bien et al., *PLOS Medicine* 2018 (**MRNet**) | 1,370 exams; abnormal / ACL tear / meniscal tear | ImageNet AlexNet per plane (sagittal, coronal, axial). Slice features are max-pooled; per-plane scores are combined by logistic regression | Defined the 2.5-D "slice encoder + pooling + late view fusion" template. Views are fused only at the score level, with 3 targets and a single site |
| Azcona et al., 2020 (MRNet benchmark) | MRNet | ResNet-18 and other 2-D backbones with transfer learning and augmentation | Pretrained 2-D backbones plus pooling are strong baselines. Fusion is still per-view |
| Tsai et al., *MIDL* 2020 (**ELNet**) | MRNet, KneeMRI | Small CNN with multi-slice normalisation and BlurPool, trained from scratch | Shows that lightweight models can compete; single plane per model |
| Liu F. et al., *Radiology* 2018 | Cartilage lesion detection | Segmentation followed by classification | Needs pixel labels |
| Namiri et al., *Radiology: AI* 2020 | ACL severity staging | 2-D/3-D CNNs | Single structure |
| Astuto et al., *Radiology: AI* 2021 | Multi-structure (cartilage, BME, meniscus, ligaments) | 3-D CNN pipelines with localisation | Needs localisation labels; heavy 3-D compute |
| fastMRI / fastMRI+ (Zbontar 2018; Zhao et al., *Sci. Data* 2022) | Knee MRI with bounding-box pathology annotations | Mostly a reconstruction benchmark; + adds labels | A public resource for pathology-level labels |
| RSNA 2024 Lumbar Spine challenge (top solutions) | Multi-series spine MRI, 25 targets | 2.5-D CNN slice encoders, then sequence/attention aggregation across series | Closest competition analogue: models must fuse several sequences and planes |

## 2. Labels from radiology reports

* **CheXpert labeler** (Irvin et al., *AAAI* 2019) is a rule-based mention/negation/uncertainty pipeline that
  outputs {positive, negative, uncertain}. **CheXbert** (Smit et al., *EMNLP* 2020) distils those rules into
  a BERT model.
* NegEx (Chapman et al., 2001) handles negation scope in clinical text.
* Gap: these tools are English-only. The RSNA knee reports are written in at least eight languages (EN, ES,
  TR, HR, DE, BG, NL, FR; our heuristic language ID). No public multilingual knee-MRI labeler exists, and the
  pretrained multilingual language models were not reachable from our compute environment.

## 3. Multiple-instance and multi-view aggregation

* **ABMIL** (Ilse et al., *ICML* 2018) uses gated attention pooling over instances and is the standard
  weakly-supervised bag aggregator.
* **Query2Label** (Liu et al., 2021) and **ML-Decoder** (Ridnik et al., *WACV* 2023) give each label a
  learnable query that cross-attends to spatial tokens. This helps when different labels depend on different
  regions, as in the knee: ACL on sagittal, MCL on coronal, the patellofemoral joint on axial.

## 4. Parameter-efficient depth: weight sharing and adaptive computation

* **Universal Transformer** (Dehghani et al., *ICLR* 2019) and **ALBERT** (Lan et al., *ICLR* 2020) show that
  applying one shared layer repeatedly keeps much of the accuracy of a deeper model with far fewer parameters.
* **Mixture-of-Depths** (Raposo et al., 2024) routes only the top-k tokens through each block, cutting FLOPs.
* **Mixture-of-Recursions (MoR)** (Bae et al., 2025, arXiv:2507.10524) combines the two. A shared recursive
  block is applied up to *R* times, and a router picks which tokens receive each further recursion
  (expert-choice with a shrinking capacity). The update is h ← h + g·f(h) for routed tokens only.
* Gap: MoR was proposed and evaluated for language modelling. We know of no study of token-level recursive
  depth for **multi-view medical volumes**. Such volumes are highly redundant: most of the 72 slices × 4
  cells per knee study are off-target anatomy.

## 5. Sparse Mixture-of-Experts

* **Shazeer et al., *ICLR* 2017** introduced sparsely-gated MoE layers: a learned router sends each token to a
  small top-k subset of expert FFNs, giving conditional capacity at near-constant FLOPs.
* **Switch Transformer** (Fedus et al., *JMLR* 2022) simplifies routing to top-1 and adds a load-balancing
  auxiliary loss; it also introduces fixed per-expert **capacity** with token dropping when a expert overflows,
  which trades some accuracy for hardware-friendly fixed-size batches on TPU pods.
* **ST-MoE** (Zoph et al., 2022) documents MoE training instability (router-logit blow-up, dead experts) and
  fixes it with a **router z-loss** that penalises large router logits, plus other stability recipes.
* **Mixtral** (Jiang et al., 2024) shows sparse top-2 MoE at scale with strong quality/compute trade-offs, using
  the same load-balancing loss form reused here.
* Gap for this task: MoE is normally paired with **more data and larger models** to justify the extra
  parameters. Here the labelled pool is small (4,349 silver + 58 gold studies) and per-study compute is
  dominated by the image backbone (~99.9% of FLOPs, §5.5/§5.8 of the README), so the usual argument for MoE
  (spend more FLOPs on more capacity) does not apply; the argument here is instead **spend near-zero extra
  FLOPs on more head capacity**, since the head is <1M parameters and the backbone is unchanged. We also found
  no prior study combining MoE's *expert*-axis routing with MoR's *depth*-axis routing for small-N, multi-view
  medical volumes, nor one that measures whether Switch-style hard capacity dropping is worth its accuracy cost
  when the token budget is only in the hundreds (as opposed to the millions of tokens per batch MoE is usually
  evaluated on).

## 6. Efficiency toolbox considered

* Pruning, quantisation and distillation (Hinton et al., 2015) usually compress a large trained model
  *after the fact*.
* Weight sharing plus routing builds the saving into the architecture. Here the head is <0.35 M parameters,
  and the frozen backbone dominates the compute (see `results/efficiency.json`). Whole-pipeline compression
  would therefore target the backbone; that is future work, see README §Limitations.

## 7. Research gap addressed

1. **Label scarcity.** Only 58 of 4,407 training studies carry expert labels. We need a leakage-free
   weak-supervision route from multilingual reports to image labels, with its noise measured against the gold
   studies.
2. **Multi-view fusion.** MRNet-style score-level fusion cannot let a finding's evidence come jointly from
   several planes. Label-specific attention over tokens from all planes can.
3. **Efficiency.** A per-study head should be very small, since only ~4k weakly-labelled studies exist, and
   its compute should scale sub-linearly with depth.
4. **Expert specialisation at near-zero extra compute (this extension).** The measured GPU run
   (`notebooks/rsna_knee_kaggle.executed.ipynb`) shows the weakest gold labels are the small/focal ones
   (Medial Meniscus, MCL) next to strong diffuse ones (Medial OA, Effusion) -- exactly the kind of
   heterogeneity a single shared FFN must compromise on. A sparse top-k Mixture-of-Experts FFN inside the
   already-shared MoR block lets different recursion tokens specialise (e.g. focal-tear vs. diffuse-OA
   evidence) without adding an unshared block per specialisation and without materially changing FLOPs,
   since the backbone dominates compute. §5 above is the gap this closes: MoE for capacity under a fixed,
   tiny compute and data budget, not MoE for scale.

**Hypothesis.** A single weight-shared transformer block applied recursively over joint multi-plane slice
tokens, with MoR token routing and label-query decoding (**MV-MoR**), will:

* **H1** outperform view-agnostic pooling heads (mean-pool MLP, logistic regression, ABMIL) on macro AUC;
* **H2** match an unshared transformer of equal depth with about 43% fewer parameters;
* **H3** use fewer FLOPs than full-depth recursion without a significant loss in AUC;
* **H4** beat sagittal-only input, since ACL/OA/effusion evidence is not confined to one plane.

**Extension hypothesis (MV-MoRE).** Replacing the block's single FFN with a sparse top-k Mixture-of-Experts
FFN, trained with a switch-style load-balancing loss and an ST-MoE router z-loss:

* **H5, partially supported (measured once on Kaggle GPU, not yet on the CPU study — see README §3.1).**
  MV-MoRE raised gold macro AUC by +0.009 and OOF macro AUC by +0.010 over the single-FFN head, and MCL (one
  of the three predicted-weakest labels) gained substantially (+0.095 gold AUC). But **Synovitis, another
  predicted-weakest label, lost AUC (−0.043)** — so the clean version of H5 ("every weak label benefits from
  expert specialisation") is falsified, while the weaker version ("net macro AUC improves") holds on this one
  run. The wall-clock cost (+32% for 5 folds) was also larger than the FLOPs-share argument (head is
  three to four orders of magnitude smaller than the backbone's compute) predicted, because that argument is
  about FLOPs, not about the unbatched, Python-loop-based dispatch `MoEFFN` currently uses — see README §3.1
  for the corrected claim. The CPU-study version of this hypothesis (models `mvmore*`, `hybrid_more`, tested
  via `bash scripts/run_experiments.sh` against the frozen-backbone baselines with 3 seeds and bootstrap CIs)
  remains untested.
