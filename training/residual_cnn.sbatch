#!/bin/bash
#SBATCH --job-name=fow-residual
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/residual-%j.out
#SBATCH --error=logs/residual-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the fow repository root}"
module load miniforge
source .venv-cnn/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/fow-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" out/cluster
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print("PyTorch:", torch.__version__, "GPU:", torch.cuda.get_device_name(0))'

# A fresh directory for every new job. Resume stays in its original run directory.
run_args=(--device cuda --threads "$SLURM_CPUS_PER_TASK")
if [[ "${1:-}" != "--resume" ]]; then
    run_args+=(--run-dir "out/cluster/residual-${SLURM_JOB_ID}")
fi
if [[ "${1:-}" != "--resume" ]]; then
    run_args+=(--model-type feedforward --channels 16 --hidden-size 64
        --residual-blocks 3 --dropout 0.1 --batch-norm --activation relu
        --epochs 50 --batch-size 128 --grad-clip-norm 5
        --learning-rate 1e-4 --peak-learning-rate 1e-3 --min-learning-rate 1e-5
        --warmup-updates 5820 --schedule warmup_cosine
        --beta-1 0.9 --beta-2 0.99 --weight-decay 0.01 --epsilon 1e-8
        --seed 7980 --selection-metric legal_accuracy)
fi
srun python -u training/train.py "${run_args[@]}" "$@"
