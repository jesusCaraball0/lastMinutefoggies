#!/bin/bash
#SBATCH --job-name=fow-ppo
#SBATCH --partition=mit_normal_gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ppo-%j.out
#SBATCH --error=logs/ppo-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the fow repository root}"
module load miniforge
source .venv-cnn/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/fow-ppo-matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" out/rl
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.__version__, torch.cuda.get_device_name(0))'
run_args=(--device cuda --threads "$SLURM_CPUS_PER_TASK")
entrypoint=rl/train.py
if [[ "${1:-}" == "--pair" ]]; then
    entrypoint=rl/train_pair.py
    shift
fi
srun python -u "$entrypoint" "${run_args[@]}" "$@"
