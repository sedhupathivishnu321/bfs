#!/usr/bin/env bash
# Main experiments: 5-fold CV on silver pool, gold = 58 expert-labelled studies (never trained on).
# Shared budget from the leakage-safe pilot (results/pilot): 10 epochs, OneCycle, AdamW wd 0.05, bs 32.
# lr per model family chosen on the pilot inner split: 5e-4 attention/transformer heads, 1e-3 MIL/MLP heads.
set -u
cd "$(dirname "$0")/.."
LOG=${LOG:-/home/user/data/logs/exp}; mkdir -p "$LOG"
E=10
jobs=(
 "--models mvmor --seeds 0 --lr 5e-4 --save-models"
 "--models mvmor --seeds 1 --lr 5e-4"
 "--models mvmor --seeds 2 --lr 5e-4"
 "--models transformer --seeds 0 --lr 5e-4"
 "--models transformer --seeds 1 --lr 5e-4"
 "--models transformer --seeds 2 --lr 5e-4"
 "--models abmil,meanmlp --seeds 0,1,2 --lr 1e-3"
 "--models mvmor_norouting --seeds 0 --lr 5e-4"
 "--models mvmor_r1 --seeds 0 --lr 5e-4"
 "--models mvmor_meandec --seeds 0 --lr 5e-4"
 "--models mvmor_noplane --seeds 0 --lr 5e-4"
 "--models mvmor_unshared --seeds 0 --lr 5e-4"
 "--models mvmor --seeds 0 --lr 5e-4 --planes Sagittal --tag _sagonly"
 # MV-MoRE (MoR + sparse top-k Mixture-of-Experts FFN) and its ablations, same protocol as MV-MoR above
 "--models mvmore --seeds 0 --lr 5e-4 --save-models"
 "--models mvmore --seeds 1 --lr 5e-4"
 "--models mvmore --seeds 2 --lr 5e-4"
 "--models mvmore_e2 --seeds 0 --lr 5e-4"
 "--models mvmore_e8 --seeds 0 --lr 5e-4"
 "--models mvmore_top1 --seeds 0 --lr 5e-4"
 "--models mvmore_expdrop --seeds 0 --lr 5e-4"
 "--models mvmore_unshared --seeds 0 --lr 5e-4"
 "--models mvmore_nozloss --seeds 0 --lr 5e-4"
 "--models mvmore_nobalance --seeds 0 --lr 5e-4"
 "--models hybrid_more --seeds 0,1,2 --lr 5e-4 --save-models"
)
printf '%s\n' "${jobs[@]}" | xargs -P 4 -I{} sh -c \
  'n=$(echo "{}" | tr -c "a-z0-9_" "_" | cut -c1-80); python3 src/kneemor/train.py {} --epochs '"$E"' --threads 1 > '"$LOG"'/$n.log 2>&1'
