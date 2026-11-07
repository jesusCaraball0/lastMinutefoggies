# Boosted CNN: full-data, 50-epoch pretraining

Submit `sbatch training/boosted_cnn.sbatch` from `~/fow` after creating `logs/`.
This uses every historical game in the same seeded90/10 game-level split;
there are no pilot data limits. The new variant is `boosted_cnn`, with the
medium residual backbone (32 channels,128 features,3 residual blocks,BN,
dropout0.1). Its1819 input planes encode20 own-turn action/observation records,
last-seen contents, observation age/visibility and known own castling rights.
It has1,370,625 parameters. Inputs use only the player's available information.

The50-epoch recipe matches `residual_cnn.sbatch`: AdamW betas(0.9,0.99),
eps1e-8, decay0.01, batch128, clip5, default framework initialization;
warmup5820 updates LR1e-4→1e-3, then cosine to1e-5 over all50 epochs.
Best checkpoint maximizes legal top1 accuracy on8192 fixed validation moves;
raw validation uses the entire held-out split. The existing JSON diagnostics,
loss/LR/gradient plots and final per-layer histograms are retained.

A disk-backed cache at `out/cache/boosted-v1/` stores203 bytes per move plus
six index/target bytes;20-turn features are assembled only per batch. Expect
roughly1.2GiB of disk cache for approximately6million positions, rather than
keeping all float history tensors in RAM. First-run preprocessing appears as
`Encode boosted replays` in the `.err` log. Cache reuse is keyed by schema,
ordered source paths, file sizes and modification times; interrupted encoding
is rebuilt safely. It uses no new pip packages beyond the existing environment.

The job activates `.venv-cnn` itself and checks CUDA. Allocation remains one
GPU, four CPUs,32GiB RAM and six hours. The wider input costs more compute;
50 epochs fitting in six hours is not guaranteed. Resume preserves optimizer,
LR schedule, input definition, split, metric history and best selection.

On your Mac, sync code (the existing remote dataset and results are preserved):

```bash
cd /Users/jesuscaraball0/code/school_code/67980/fow
rsync -av --progress --prune-empty-dirs \
  --exclude='.git/' --exclude='.venv*/' --exclude='__pycache__/' \
  --exclude='game_data/' --exclude='out/' --exclude='logs/' --exclude='work/' \
  --exclude='evaluation_results/' --exclude='build/' \
  --include='*/' --include='*.py' --include='*.sh' --include='*.sbatch' \
  --include='*.txt' --include='*.toml' --include='*.md' --exclude='*' \
  ./ jesusca@orcd-login.mit.edu:~/fow/
ssh jesusca@orcd-login.mit.edu
```

On the cluster:

```bash
cd ~/fow
mkdir -p logs
boosted_job_id=$(sbatch --parsable training/boosted_cnn.sbatch)
echo "Submitted job: $boosted_job_id"
tail -F "logs/boosted-cnn-${boosted_job_id}.out" "logs/boosted-cnn-${boosted_job_id}.err"
```

`Ctrl-C` stops following logs without cancelling the Slurm job. Results are
`out/experiments/boosted-cnn-JOBID/`: `best_policy.pt`, `last_policy.pt`,
`final_policy.pt`, `metrics.json`, split, plots and histograms. To resume after
a wall-time limit, replace OLD_JOB_ID with that original job number:

```bash
boosted_job_id=$(sbatch --parsable training/boosted_cnn.sbatch \
  --resume out/experiments/boosted-cnn-OLD_JOB_ID/last_policy.pt)
tail -F "logs/boosted-cnn-${boosted_job_id}.out" "logs/boosted-cnn-${boosted_job_id}.err"
```

Resume continues in the original result directory, while the new allocation
has its own logs. If the job expires during initial feature caching before
any epoch checkpoint exists, submit a fresh job; its cache is rebuilt/reused.
No local pretraining, PPO continuation or PSRO process is launched by setup.

Experimental findings are maintained in [EXPERIMENT_HISTORY.md](../EXPERIMENT_HISTORY.md). This document covers cluster operations.

# Medium residual CNN: full-data, 50-epoch run

Submit `sbatch training/residual_cnn.sbatch` from `~/fow` after creating `logs/`.
This uses all games in the fixed 90/10 split, 32 channels, 128 hidden features (850,785 parameters),
three residual blocks, convolutional BatchNorm, ReLU, default initialization,
and hidden dropout 0.1. AdamW uses betas (0.9, 0.99), eps 1e-8, decay 0.01;
batch size 128 and global clipping 5. Warmup is 5,820 updates from 1e-4 to
1e-3, followed by cosine decay over the rest of all 50 epochs to 1e-5.
One GPU, 32 GiB RAM, four CPU cores, six-hour allocation.

`best_policy.pt` selects the highest legal top-1 accuracy on the same fixed
8,192 held-out moves each epoch, retaining the earlier checkpoint on ties.
Raw validation loss/accuracy still use the entire validation split.
`last_policy.pt` and `final_policy.pt` are also retained, with diagnostics.
Results: `out/experiments/residual-medium-<job-id>/`; logs: `logs/residual-medium-<job-id>.*`.
Resume with `sbatch training/residual_cnn.sbatch --resume out/experiments/residual-medium-JOBID/last_policy.pt`.
Resume preserves architecture, optimizer, schedule and checkpoint selection.
The earlier plain CNN instructions below remain available for old runs.

Code-only sync (does not delete remote data or results):

```bash
rsync -av --progress --prune-empty-dirs \
  --exclude='.git/' --exclude='.venv*/' --exclude='__pycache__/' \
  --exclude='game_data/' --exclude='out/' --exclude='logs/' \
  --exclude='evaluation_results/' --exclude='build/' \
  --include='*/' --include='*.py' --include='*.sh' --include='*.sbatch' \
  --include='*.txt' --include='*.toml' --include='*.md' --exclude='*' \
  ./ jesusca@orcd-login.mit.edu:~/fow/
```

# Full CNN training on MIT ORCD

The included `training/full_cnn.sbatch` requests one GPU on `mit_normal_gpu`, four
CPU cores, 32 GiB RAM, and six hours. Submit from the repository root. Create
`logs/` before submission because Slurm opens log files before the script runs.
The script uses `.venv-cnn`, checks CUDA, and writes to
`out/experiments/full-cnn-<job-id>/`.

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
mkdir -p logs out/experiments
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
ls "out/experiments/full-cnn-${cnn_job_id}"
```

To change the epoch budget before starting a new run, append `--epochs 12` to
`sbatch training/full_cnn.sbatch`. The schedule then decays over that full budget.
Do not add `--epochs` when resuming; the saved budget is restored.

## 4. Resume after a timeout

Each completed epoch saves `last_policy.pt`. Resume in a new six-hour allocation
from the previous job's checkpoint (replace `12345678` with its job ID):

```bash
sbatch training/full_cnn.sbatch --resume out/experiments/full-cnn-12345678/last_policy.pt
```

Pass `--resume` first. The new job's logs use its new job ID, while results stay
in the original run directory. Resume restores weights, BatchNorm running stats,
AdamW moments, LR schedule, RNG state, data split, and shuffle-generator state.
Any interrupted partial epoch is replayed. If the job ends before completing
its first epoch, there is no completed-epoch checkpoint to resume yet.

## 5. Download results to the Mac

In a local terminal, from the repository root:

```bash
mkdir -p out/experiments
rsync -av --progress \
  jesusca@orcd-login.mit.edu:~/fow/out/experiments/ \
  out/experiments/
```

Each run remains inside its job-specific directory. Existing local experiment
runs and bot checkpoints are untouched. Repeat the download after a resumed job
to update that downloaded run.

`best_policy.pt` is selected by validation loss; `last_policy.pt` is resumable;
`final_policy.pt` is the final epoch. The directory also contains metrics JSON,
data split, loss/LR/gradient/update/legality/dormancy plots, final weight and
activation histograms, and histogram JSON. Final plots are generated when all
epochs finish; checkpoints and metrics are saved after every completed epoch.

## Jobs submitted before artifact consolidation

Jobs already running from the previous script still write under `~/fow/out/cluster/`. Do not rename their server directories while they run. To download the active medium residual after it finishes, use this on the Mac (replace JOBID):

```bash
mkdir -p /Users/jesuscaraball0/code/school_code/67980/fow/out/experiments
scp -r jesusca@orcd-login.mit.edu:~/fow/out/cluster/residual-medium-JOBID \
  /Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/
```

New submissions after syncing the updated scripts write directly under `~/fow/out/experiments/`.

## Boosted outcome CNN: joint move/outcome pretraining

This is a new model,trained from scratch with the same medium residual architecture,
1819 boosted input planes,50epochs and previous hyperparameters. Both policy and
outcome losses train the encoder. Objective: move CE +0.5*outcome MSE; own-side
terminal targets win+1,loss−1,draw+0.1,undiscounted. Keep the exact split from
boosted-cnn-28018186; no new data or hidden-state input. Best remains selected by
legal move accuracy. Original boosted and all RL weights remain separate.

On the Mac,from the fow repository:

```bash
rsync -av --progress --exclude='__pycache__/' --exclude='*.pt' --exclude='*.onnx' \
  training rl starters test_boosted_outcome.py \
  jesusca@orcd-login.mit.edu:~/fow/
ssh jesusca@orcd-login.mit.edu
```

Then on the cluster:

```bash
cd ~/fow
mkdir -p logs out/experiments
test -f out/experiments/boosted-cnn-28018186/data_split.json
outcome_job_id=$(sbatch --parsable training/boosted_outcome_cnn.sbatch)
outcome_job_id=${outcome_job_id%%;*}
echo "$outcome_job_id"
squeue -j "$outcome_job_id"
tail -F "logs/boosted-outcome-cnn-${outcome_job_id}.out" \
        "logs/boosted-outcome-cnn-${outcome_job_id}.err"
```

Uses the existing .venv-cnn/CUDA installation; no reinstall required. First launch
builds outcome-label sidecars in the existing boosted cache; progress goes to stderr.
The job activates its own venv. Closing SSH or stopping tail does not cancel sbatch.
Results: `out/experiments/boosted-outcome-cnn-JOBID/`; all existing diagnostics plus
`outcome_training.png`. A six-hour allocation is requested; completion in that
allocation is not guaranteed. Resume from last completed epoch if needed:

```bash
sbatch training/boosted_outcome_cnn.sbatch \
  --resume out/experiments/boosted-outcome-cnn-OLD_JOB_ID/last_policy.pt
```

Resume stays in the original result directory and preserves optimizer,LR schedule,
data split and shuffle/RNG state. Pass --resume first. Logs use the new job ID.
To pull the directory back,from the Mac:

```bash
scp -r jesusca@orcd-login.mit.edu:~/fow/out/experiments/boosted-outcome-cnn-JOBID \
  /Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/
```

## Wide boosted outcome CNN

The64-channel/256-hidden variant uses the same joint loss and50-epoch recipe as
the32/128 outcome model. Parameters increase1.37M→3.37M. Same data split and
seed,from scratch;it may need multiple6-hour allocations. No completion-time
claim or automatic resubmission is provided. Medium and wide can be submitted
as separate jobs;GPU availability determines scheduling.

On the Mac:

```bash
cd /Users/jesuscaraball0/code/school_code/67980/fow
rsync -av --progress --exclude='__pycache__/' --exclude='*.pt' --exclude='*.onnx' \
  training rl starters test_boosted_outcome.py \
  jesusca@orcd-login.mit.edu:~/fow/
ssh jesusca@orcd-login.mit.edu
```

On the cluster:

```bash
cd ~/fow
mkdir -p logs out/experiments
wide_job_id=$(sbatch --parsable training/boosted_outcome_wide_cnn.sbatch)
wide_job_id=${wide_job_id%%;*}
echo "Wide job: $wide_job_id"
squeue -j "$wide_job_id"
tail -F "logs/boosted-outcome-wide-cnn-${wide_job_id}.out" \
        "logs/boosted-outcome-wide-cnn-${wide_job_id}.err"
```

If the allocation expires,replaceJOBID with the original job ID:

```bash
sbatch training/boosted_outcome_wide_cnn.sbatch \
  --resume out/experiments/boosted-outcome-wide-cnn-JOBID/last_policy.pt
```

Resume keeps the original results directory,total50-epoch LR schedule and optimizer/
shuffle/RNG state. New allocation logs use the new job ID. Pull from the Mac:

```bash
scp -r jesusca@orcd-login.mit.edu:~/fow/out/experiments/boosted-outcome-wide-cnn-JOBID \
  /Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/
```
