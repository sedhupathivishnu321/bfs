"""Builds notebooks/efficiency_lb_insights.ipynb (run: python notebooks/build_lb_insights.py, then nbconvert --execute).

Reference: the organisers' "RSNA Knee Abnormalities - Efficiency LB" notebook (top-100 Efficiency Prize table).
Its leaderboard table is parsed from the uploaded notebook's saved output into data/efficiency_lb_top100.csv.
"""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Efficiency-Prize leaderboard: insights and where this project stands

**Reference:** the organisers' *RSNA Knee Abnormalities – Efficiency LB* notebook, which lists the top 100
teams of the Efficiency Prize track. Its `leaderboard.csv` input is on Kaggle and cannot be reached from this
environment. The table below was parsed from that notebook's saved output (top 100 rows, columns
`EfficiencyRank`, `TeamName`, `PublicScore`, `DateSubmitted`) into `data/efficiency_lb_top100.csv`.

**Not available:** the per-team runtime and the exact efficiency formula. Those live on the competition's
*Efficiency Prize Evaluation* page, which is also unreachable. Statements about runtime below are
therefore **inferences from rank order**, not measurements.

This project's numbers come from `results/summary.csv`. They were measured on held-out folds and on the
58 expert-labelled studies. They are **not** leaderboard scores, because nothing was submitted to Kaggle."""),
    code("""import pandas as pd, numpy as np, matplotlib.pyplot as plt
from scipy.stats import spearmanr

pd.options.display.min_rows = 100
pd.options.display.max_rows = 100

# reference cell, adapted: local copy of the leaderboard shown in the reference notebook
leaderboard = pd.read_csv('../data/efficiency_lb_top100.csv', index_col='EfficiencyRank')
leaderboard['DateSubmitted'] = pd.to_datetime(leaderboard['DateSubmitted'], format='%a %b %d %H:%M:%S %Y')
leaderboard.head(10)"""),
    md("## 1. Score distribution of the top 100"),
    code("""s = leaderboard.PublicScore
summary = pd.Series({'teams': len(s), 'min': s.min(), 'q25': s.quantile(.25), 'median': s.median(),
                     'q75': s.quantile(.75), 'max': s.max(), 'range': s.max() - s.min()}).round(4)
summary"""),
    md("## 2. Does efficiency rank follow accuracy?"),
    code("""rho, p = spearmanr(leaderboard.index, leaderboard.PublicScore)
best = leaderboard.PublicScore.idxmax()
print(f'Spearman(EfficiencyRank, PublicScore) = {rho:.3f} (p = {p:.3g})')
print(f'Highest public score in top-100: {leaderboard.PublicScore.max():.3f} at efficiency rank {best} '
      f'({leaderboard.loc[best, "TeamName"]})')
top10 = leaderboard.head(10).PublicScore
print(f'Top-10 efficiency ranks: public score {top10.min():.3f}-{top10.max():.3f}, mean {top10.mean():.4f}')

# rank inversions: team i ranked above team j despite a lower score => i must be cheaper to run
inv = [(i, j) for i in leaderboard.index for j in leaderboard.index
       if i < j and leaderboard.PublicScore[i] < leaderboard.PublicScore[j]]
print(f'Ordered pairs where the better-ranked team has the LOWER score: {len(inv)} of {100*99//2} '
      f'({len(inv)/(100*99/2):.0%}) -> runtime must be doing much of the ranking')"""),
    code("""fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
ax.scatter(leaderboard.index, leaderboard.PublicScore, s=26, color='#2a78d6', edgecolor='#fcfcfb', linewidth=1)
b = leaderboard.PublicScore.idxmax()
ax.annotate(f'best score {leaderboard.PublicScore[b]:.3f}\\n(efficiency rank {b})', (b, leaderboard.PublicScore[b]),
            xytext=(b + 8, leaderboard.PublicScore[b] - 0.004), fontsize=9, color='#52514e',
            arrowprops=dict(arrowstyle='-', color='#9a9994', lw=0.8))
ax.set_xlabel('Efficiency rank (1 = best)', color='#52514e'); ax.set_ylabel('Public score (macro AUC)', color='#52514e')
ax.set_title('Efficiency rank vs public macro AUC, top 100', loc='left', fontsize=11, color='#0b0b0b')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color('#d9d8d3')
ax.tick_params(colors='#52514e'); ax.grid(axis='y', color='#ecebe7', lw=0.8); ax.set_axisbelow(True)
plt.tight_layout(); plt.show()"""),
    md("## 3. When were the ranked submissions made?"),
    code("""d = leaderboard.DateSubmitted.dt.date.value_counts().sort_index()
print(d.to_string())
print(f'\\n{(leaderboard.DateSubmitted >= "2026-09-20").mean():.0%} of top-100 selected submissions were made in the last week shown')"""),
    md("""## 4. Where this project stands

Two of our numbers are relevant, and neither is a leaderboard score:

* **Gold macro AUC.** Measured on the 58 expert-labelled *training* studies, which were never trained on.
* **OOF macro AUC.** Measured against our report-derived silver labels. If the hidden test labels were
  derived from reports in a similar way, this is the closer analogue.

The public test set, its labelling process and its size are unknown, so the gap below is indicative only."""),
    code("""ours = pd.read_csv('../results/summary.csv').set_index('model')
pick = ['hybrid', 'avg_mlp_mvmor', 'mvmor', 'meanmlp', 'abmil', 'logreg']
tbl = ours.loc[pick, ['params', 'oof_auc_ens', 'gold_auc_ens', 'gold_ci_lo', 'gold_ci_hi']].round(4)
lb_min, lb_med, lb_max = s.min(), s.median(), s.max()
tbl['gap_to_LB_top100_min (gold)'] = (tbl.gold_auc_ens - lb_min).round(3)
tbl['gap_to_LB_best (gold)'] = (tbl.gold_auc_ens - lb_max).round(3)
tbl"""),
    code("""fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=110)
fig.patch.set_facecolor('#fcfcfb'); ax.set_facecolor('#fcfcfb')
bins = np.arange(0.66, 0.965, 0.005)
ax.hist(s, bins=bins, color='#2a78d6', edgecolor='#fcfcfb', linewidth=2, label='Top-100 public scores')
refs = [('Our hybrid, gold (58 studies)', ours.loc['hybrid', 'gold_auc_ens']),
        ('Report labeler vs gold', 0.735),
        ('Our hybrid, OOF vs silver labels', ours.loc['hybrid', 'oof_auc_ens'])]
for k, (name, v) in enumerate(refs):
    ax.axvline(v, color='#52514e', lw=1.2, ls=['-', '--', ':'][k])
    ax.text(v + 0.002, ax.get_ylim()[1] * (0.92 - 0.14 * k), f'{name}\\n{v:.3f}', fontsize=8, color='#52514e', va='top')
ax.set_xlabel('Macro AUC', color='#52514e'); ax.set_ylabel('Teams', color='#52514e')
ax.set_title('Our measured AUCs vs the top-100 public scores (not the same test set)', loc='left', fontsize=11, color='#0b0b0b')
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
for sp in ['left', 'bottom']: ax.spines[sp].set_color('#d9d8d3')
ax.tick_params(colors='#52514e'); ax.legend(frameon=False, fontsize=9, loc='upper left', bbox_to_anchor=(0.6, 1.0))
from matplotlib.ticker import MaxNLocator; ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout(); plt.show()"""),
    md("## 5. Our efficiency profile (measured, CPU, 1 thread)"),
    code("""import json
eff = json.load(open('../results/efficiency.json'))
e = pd.DataFrame({k: v for k, v in eff.items() if isinstance(v, dict)}).T[['params', 'gflops', 'latency_ms_median']]
e.loc['hybrid (backbone excluded)'] = [745383, 0.1278, np.nan]
bb = eff['backbone_resnet18_72slices']['gflops']
e['share_of_pipeline_gflops'] = np.where(e.index.str.startswith('backbone'), 1.0, e.gflops / (e.gflops + bb)).round(4)
e.round(4)"""),
    md("""## 6. Insights

The printed outputs above supply the numbers.

1. **The top 100 is compressed at a very high level.** Public macro AUC runs from 0.916 to 0.958, with an
   IQR of about 0.013. 59% of the top-100 selected submissions came in the final five days shown
   (21–25 Sep), so the board was still moving at the end. Differences of a few thousandths separate teams, so a submission needs to be above
   about 0.92 to enter this table.
2. **The efficiency ranking tracks accuracy only moderately.** The Spearman correlation between rank and
   score is −0.56, so better rank does go with higher score. But in 29% of team pairs the better-ranked team
   has the *lower* score: rank 2 scores 0.948 while rank 79 scores 0.957. The top 10 all score at least 0.944,
   so accuracy still sets a floor, and within that band runtime or cost decides.
3. **Our pipeline is far below that band.** The hybrid's gold AUC of 0.713 is about 0.20 below the lowest
   top-100 public score. Even our OOF-vs-silver AUC of 0.797 is well below it. The sets differ, so the gap
   is indicative, but it is too large to be a test-set artefact.
4. **The gap is mostly supervision, not the head.** Every head we tried lands at 0.69–0.71 gold, and our
   report labeler itself reaches only 0.735 against gold. Teams at 0.94 or more must use much stronger
   labels, most plausibly LLM or multilingual-NLP report labelling validated on the gold studies, plus image
   encoders fine-tuned end-to-end on GPU. This is an inference; their methods are not in the reference
   notebook.
5. **Our efficiency profile is already favourable.** The routed hybrid head is about 0.1% of pipeline
   FLOPs, and the frozen 2-D backbone is essentially the whole cost. For the Efficiency track the lever is
   therefore the backbone:
   * a smaller or distilled encoder;
   * fewer slices, since the router shows middle slices are processed least;
   * INT8 quantisation.

   Head tuning will not help there.
6. **Priorities that follow from the evidence:**
   * (a) Replace the rule labeler with a stronger multilingual labeler and measure it on the 58 gold studies
     with nested CV.
   * (b) Fine-tune a 2.5-D CNN slice encoder on GPU.
   * (c) Keep the hybrid, or the mean-pool MLP, as the cheap aggregation head.
   * (d) Compress the backbone for the efficiency track."""),
    md("""## 7. Follow-up experiments: better labels, honest accuracy, a cheaper backbone

These were run after the leaderboard analysis, to act on its conclusions. All numbers are measured.
Sources: `results/labeler_v2_gold_loo.csv` and `results/efficiency_v2.json`."""),
    code("""lab = pd.read_csv('../results/labeler_v2_gold_loo.csv', index_col=0)
lab[['ACL','MCL','Medial OA','PF OA','Effusion','Synovitis','macro']]"""),
    code("""e2 = json.load(open('../results/efficiency_v2.json'))
acc = e2.pop('accuracy')
t = pd.DataFrame(e2).T
base = 'fp32 · 24 sl · 160px'
t['speedup_vs_fp32'] = (t.loc[base, 'backbone_latency_ms'] / t.backbone_latency_ms.astype(float)).round(2)
t['auc_change'] = (t.gold_auc.astype(float) - t.loc[base, 'gold_auc']).round(4)
print(f"Hybrid gold accuracy, leave-one-out thresholds: {acc['hybrid_loo_threshold_acc']:.3f}  "
      f"vs always-negative {acc['all_negative_acc']:.3f}")
t[['gold_auc','auc_change','backbone_gflops','backbone_latency_ms','speedup_vs_fp32','slices_encoded']]"""),
    md("""**Reading the follow-ups:**

* **Labeler v2 gives no overall gain.** Fitting a label-mapping model on only 58 gold reports lifts OA
  sharply (Medial OA 0.69 → 0.90 leave-one-out) but loses on ligaments and synovitis. The macro AUC stays at
  0.733 or below, against 0.735 for the rules. So v2 was *not* used for retraining.
* **Honest accuracy is 72.0%** against 65.5% for always predicting "negative". Thresholds were fitted by
  leave-one-out on gold.
* **INT8 quantisation of the backbone is about 5× faster on CPU with no measurable AUC loss.** This is the
  efficiency-track recommendation. Halving the slices halves the cost for about −0.01 AUC; 128 px costs
  about −0.03 AUC.
* **99% accuracy is not reachable** with these labels: the labels themselves only reach 0.735 AUC against
  the expert standard."""),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebooks/efficiency_lb_insights.ipynb")
print("wrote notebooks/efficiency_lb_insights.ipynb")
