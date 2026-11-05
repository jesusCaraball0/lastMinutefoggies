#!/bin/bash
set -euo pipefail
cd /Users/jesuscaraball0/code/school_code/67980/fow
export MPLCONFIGDIR=/tmp/fow-matplotlib
export XDG_CACHE_HOME=/tmp/fow-cache
.venv/bin/python -u rl/pretrain_value.py \
  --checkpoint out/rl/cnn-c64-h256-8ebd1cf653/policy.pt \
  --checkpoint out/rl/residual-c32-h128-518d814dce/policy.pt \
  --after-update 400 --batch-size 1024 --threads 4 --device cpu \
  > logs/value-pretrain-after-400.out 2> logs/value-pretrain-after-400.err
.venv/bin/python -u rl/train_pair.py \
  --first out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt \
  --second out/experiments/residual-medium-27890681/best_policy.pt \
  --updates 800 --games-per-update 16 --device cpu --threads 1 \
  > logs/ppo-pair-401-800.out 2> logs/ppo-pair-401-800.err
