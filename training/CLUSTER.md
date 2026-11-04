# Residual CNN: full-data, 50-epoch run

Submit `sbatch training/residual_cnn.sbatch` from `~/fow` after creating `logs/`.
This uses all games in the fixed 90/10 split, 16 channels, 64 hidden features,
three residual blocks, convolutional BatchNorm, ReLU, default initialization,
and hidden dropout 0.1. AdamW uses betas (0.9, 0.99), eps 1e-8, decay 0.01;
batch size 128 and global clipping 5. Warmup is 5,820 updates from 1e-4 to
1e-3, followed by cosine decay over the rest of all 50 epochs to 1e-5.
One GPU, 32 GiB RAM, four CPU cores, six-hour allocation.

`best_policy.pt` selects the highest legal top-1 accuracy on the same fixed
8,192 held-out moves each epoch, retaining the earlier checkpoint on ties.
Raw validation loss/accuracy still use the entire validation split.
`last_policy.pt` and `final_policy.pt` are also retained, with diagnostics.
Results: `out/cluster/residual-<job-id>/`; logs: `logs/residual-<job-id>.*`.
Resume with `sbatch training/residual_cnn.sbatch --resume out/cluster/residual-JOBID/last_policy.pt`.
Resume preserves architecture, optimizer, schedule and checkpoint selection.
The earlier plain CNN instructions below remain available for old runs.

Code-only sync (does not delete remote data or results):

```bash
rsync -av --progress --prune-empty-dirs \
  --exclude='.git/' --exclude='.venv*/' --exclude='__pycache__/' \
  --exclude='game_data/' --exclude='out/' --exclude='logs/' \
  --exclude='evaluation_results/' --exclude='build/' \
  --include='*/' --include='*.py' --include='*.sbatch' \
  --include='*.txt' --include='*.toml' --include='*.md' --exclude='*' \
  ./ jesusca@orcd-login.mit.edu:~/fow/
```

# Full CNN training on MIT ORCD

The included `training/full_cnn.sbatch` requests one GPU on `mit_normal_gpu`, four
CPU cores, 32 GiB RAM, and six hours. Submit from the repository root. Create
`logs/` before submission because Slurm opens log files before the script runs.
The script uses `.venv-cnn`, checks CUDA, and writes to
`out/cluster/full-cnn-<job-id>/`.

Official references: [ORCD scheduling](https://orcd-docs.mit.edu/running-jobs/overview/),
[GPU requests](https://orcd-docs.mit.edu/running-jobs/requesting-resources/), and
[Python environments](https://orcd-docs.mit.edu/software/python/).

## Configuration

The trainer defaults to the full feedforward CNN (64 channels / 256 features),
ReLU, default initialization, BatchNorm after each convolution, and global L2
clipping at 5. AdamW uses betas `(0.9, 0.99)`, epsilon `1e-8`, and weight decay
`0.01`. Batch size remains 128, matching the experiments.

Use all training games from the fixed 90/10 game split, for eight epochs.
Warmup lasts **5,820 optimizer updates**, from `1e-4` to `1e-3`; cosine decay
covers the remaining updates, ending at `1e-5`. Total updates equal eight times
the number of training batches, and are printed in the job log and stored in
`metrics.json`. Thus this full-data run has a larger budget than the pilots.

Raw validation uses the full validation split. The per-epoch legal-action probe
uses 8,192 fixed held-out moves by default; `--probe-max-positions 0` evaluates
all held-out moves. Activation diagnostics use 512 observations. No dropout or
He initialization is added.

## 1. Sync from the Mac

Run from the local repository. This includes the sbatch file and dataset, skips
existing results/environments/weights, and does not delete remote files:

```bash
cd /Users/jesuscaraball0/code/school_code/67980/fow
rsync -av --progress \
  --exclude='.venv*' --exclude='.git/' --exclude='__pycache__/' \
  --exclude='out/' --exclude='logs/' --exclude='evaluation_results/' \
  --exclude='build/' --exclude='*.pt' --exclude='*.onnx' \
  ./ jesusca@orcd-login.mit.edu:~/fow/
ssh jesusca@orcd-login.mit.edu
```

## 2. Prepare a dedicated cluster environment

On the cluster login node:

```bash
cd ~/fow
module load miniforge
python -m venv .venv-cnn
source .venv-cnn/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r training/requirements.txt
mkdir -p logs out/cluster
```

The CUDA wheel supplies the PyTorch CUDA runtime; the job checks GPU access
inside the allocated compute node. Existing `.venv` environments are preserved.
You only need to install the environment once.

## 3. Submit and monitor

The job file is already created and copied by rsync:

```bash
cnn_job_id=$(sbatch --parsable training/full_cnn.sbatch)
echo "Submitted job: $cnn_job_id"
squeue -j "$cnn_job_id"
```

Once the job starts, follow its log:

```bash
tail -f "logs/full-cnn-${cnn_job_id}.out"
```

The `.err` log contains progress bars and errors:

```bash
tail -f "logs/full-cnn-${cnn_job_id}.err"
```

`Ctrl-C` stops following the log, not the training job. After completion:

```bash
sacct -j "$cnn_job_id" --format=JobID,State,Elapsed,MaxRSS
ls "out/cluster/full-cnn-${cnn_job_id}"
```

To change the epoch budget before starting a new run, append `--epochs 12` to
`sbatch training/full_cnn.sbatch`. The schedule then decays over that full budget.
Do not add `--epochs` when resuming; the saved budget is restored.

## 4. Resume after a timeout

Each completed epoch saves `last_policy.pt`. Resume in a new six-hour allocation
from the previous job's checkpoint (replace `12345678` with its job ID):

```bash
sbatch training/full_cnn.sbatch --resume out/cluster/full-cnn-12345678/last_policy.pt
```

Pass `--resume` first. The new job's logs use its new job ID, while results stay
in the original run directory. Resume restores weights, BatchNorm running stats,
AdamW moments, LR schedule, RNG state, data split, and shuffle-generator state.
Any interrupted partial epoch is replayed. If the job ends before completing
its first epoch, there is no completed-epoch checkpoint to resume yet.

## 5. Download results to the Mac

In a local terminal, from the repository root:

```bash
mkdir -p out/cluster-downloads
rsync -av --progress \
  jesusca@orcd-login.mit.edu:~/fow/out/cluster/ \
  out/cluster-downloads/
```

Each run remains inside its job-specific directory. Existing local experiment
runs and bot checkpoints are untouched. Repeat the download after a resumed job
to update that downloaded run.

`best_policy.pt` is selected by validation loss; `last_policy.pt` is resumable;
`final_policy.pt` is the final epoch. The directory also contains metrics JSON,
data split, loss/LR/gradient/update/legality/dormancy plots, final weight and
activation histograms, and histogram JSON. Final plots are generated when all
epochs finish; checkpoints and metrics are saved after every completed epoch.
