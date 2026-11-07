#!/bin/bash
set -eu
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec python3 -u bot.py
