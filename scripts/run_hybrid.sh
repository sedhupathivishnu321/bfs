#!/usr/bin/env bash
# Hybrid global-local head: same protocol/budget as scripts/run_experiments.sh (10 epochs, lr 5e-4, 5-fold CV).
set -u
cd "$(dirname "$0")/.."
LOG=${LOG:-/home/user/data/logs/exp}; mkdir -p "$LOG"
for job in "hybrid 0" "hybrid 1" "hybrid 2" "hybrid_noaux 0" "hybrid_fixedgate 0"; do
  set -- $job
  nohup python3 src/kneemor/train.py --models $1 --seeds $2 --lr 5e-4 --epochs 10 --threads 1 --save-models \
    > "$LOG/$1_s$2.log" 2>&1 &
done
wait
