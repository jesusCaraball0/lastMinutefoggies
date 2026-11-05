# PPO training

Experiment results, metrics, and takeaways are maintained in [RL_EXPERIMENT_HISTORY.md](../RL_EXPERIMENT_HISTORY.md).

Experimental findings belong in [EXPERIMENT_HISTORY.md](../EXPERIMENT_HISTORY.md).
This document describes the training interface and operating commands.

The actor starts from an existing **feedforward** checkpoint: its CNN/residual
encoder and policy head are reused, then updated by PPO using fresh games.
The architecture stays the same. For imitation checkpoints, the previously
unsupervised value head starts at zero and learns to estimate discounted win/loss/draw return. Starting a new
run from a PPO checkpoint preserves its trained critic; optimizer state is only
restored when continuing the persistent RL checkpoint. There is no imitation replay-dataset pass and no search algorithm in this trainer.

Each feedforward architecture has one persistent RL file, separate from its original
supervised weights: `out/rl/<variant>/policy.pt`. First launch initializes it;
subsequent launches with the same architecture automatically resume it. A variant
name contains the architecture family, widths and a stable configuration digest,
so residual depth, normalization, dropout and activation differences cannot collide.
Recurrent actors are excluded. No RL model file is created until that variant is
first trained. Existing experiment artifacts remain historical evidence.

## Game reuse and agent contract

All games call the existing `referee.run_match`. The only referee extension is
an optional `bot_factory` hook. Rules, visibility, observations, move validation,
clocks, repetition, terminal results, and game-cap draws remain in the provided
referee; `rl/` does not implement them. The default executable-bot API and
existing evaluator retain their prior behavior.

Existing executables need no new RL-specific interface: pass their command with
`--opponent-command 'bash starters/python/run.sh'`. `rl.game.Command` uses the
referee's existing `Bot` implementation for the `fow`, color/time, `ready`,
64-square observations, clock prompt and `move <uci>` protocol. Commands are
parsed as argv, without a shell. Only run commands you trust, as with the referee.
Command opponents should remain unchanged during collection/evaluation; unlike loaded checkpoint agents,
an external executable can otherwise reload changing files between games.

For fast Python agents, implement `rl.game.Agent`:

```python
class MyAgent:
    def reset(self, side, base, increment): ...
    def observe(self, view, ply): ...
    def act(self, remaining): ...  # Return a UCI string, e.g. "e2e4".
    def close(self): ...
```

Pass a factory returning a fresh agent to `rl.game.play_game`. `side` is a bool
(`True` is white); `view` is exactly the provided fog observation, never a full
board. Observations arrive before every ply for both players, as in the existing
wire protocol. This lets neural, rule-based, RL and other agents play together.
Each agent owns/reset its per-game history. Training aborts on bot/protocol
failures rather than silently treating broken opponents as successful learning.

## Start locally

Run commands from the repository root. Existing training dependencies suffice;
if needed, install `.venv/bin/python -m pip install -r rl/requirements.txt`.

A short initial run against the two existing baselines (if this variant already has
RL weights, it continues them; `--updates` must exceed the completed total):

```bash
.venv/bin/python rl/train.py \
  --init-checkpoint out/experiments/residual-medium-27890681/best_policy.pt \
  --updates 2 --games-per-update 4 \
  --eval-every 1 --eval-games 2 --device cpu
```

This verifies the pipeline; two PPO updates do not establish playing-strength
improvement. Both default fixed opponents/evaluation models are loaded once:
the medium residual and the earlier full CNN. Override paths explicitly if those
artifacts are stored elsewhere:

```bash
.venv/bin/python rl/train.py \
  --init-checkpoint out/experiments/residual-medium-27890681/best_policy.pt \
  --opponent out/experiments/residual-medium-27890681/best_policy.pt \
  --opponent out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt \
  --updates 200 --games-per-update 16 --eval-every 10 --eval-games 20
```

`--device auto` chooses CUDA if available, otherwise CPU. The initial collector
runs complete games sequentially in-process; it avoids starting Python/Torch for
every move/game, but does not yet vectorize or distribute rollouts. A GPU can
accelerate optimization without guaranteeing faster single-observation collection.
Executable opponents have normal subprocess startup cost each game. Optimize
rollout throughput only after verifying learning and profiling this pipeline.

## Train two models against each other

For the minimal two-learner loop, run:

```bash
.venv/bin/python rl/train_pair.py \
  --first out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt \
  --second out/experiments/residual-medium-27890681/best_policy.pt \
  --updates 20 --games-per-update 16
```

Both current policies play the same complete games, alternating colors. Each saves
its own observations/actions/masks/old probabilities/values into a separate PPO
batch. Neither model changes during collection. After collection, each receives a
PPO update with its own optimizer. The next round uses both updated models.
Win/loss rewards come from each model's color; a draw gives both +0.1.
This mode has exactly these two training opponents, without historical opponent
sampling, a population payoff table or a PSRO strategy solver.

Both variants use their existing `out/rl/<variant>/policy.pt`. Repeat the command
with a higher --updates total to resume both; existing models must have equal
completed update counts. All common optimizer, collection, evaluation and device
options from rl/train.py are supported; collection settings must match between
models. --first/--second replace the single-learner initialization/resume flags.
A --run-root override is available for isolated experiments/tests. Both models
must be feedforward and have different variant identities.

Each variant retains separate metrics, graphs, optimizer and embedded best weights.
The default frozen baseline evaluation still selects each model's best independently;
only the peer supplies training games. Evaluation games do not update either model.
Games-per-update counts shared games:16 games yield16 episodes per model, not32
different games. Summing game totals across the two logs would double-count games;
learner decision counts are distinct.

After both updates succeed, their checkpoint files commit as a pair. A temporary
commit journal permits recovery if the process stops between the two atomic writes;
it is removed after a successful commit. Training is sequential on one selected
device; "both update" does not require two GPUs. Partial round failures preserve
the preceding completed pair of checkpoints.

On the cluster use the same sbatch script with --pair:

```bash
sbatch rl/ppo.sbatch --pair \
  --first out/cluster/full-cnn-27800942/best_policy.pt \
  --second out/cluster/residual-medium-27890681/best_policy.pt \
  --eval-opponent out/cluster/full-cnn-27800942/best_policy.pt \
  --eval-opponent out/cluster/residual-medium-27890681/best_policy.pt \
  --updates 20 --games-per-update 16
```

Pass the actual checkpoint paths on your server. The evaluation pool is fixed for
both learners and remains separate from their changing training matchup.

## Historical value-head warmup

For the current pair, one epoch can fit the value heads without moving the actors:

```bash
.venv/bin/python rl/pretrain_value.py \
  --checkpoint out/rl/cnn-c64-h256-8ebd1cf653/policy.pt \
  --checkpoint out/rl/residual-c32-h128-518d814dce/policy.pt \
  --after-update 400 --batch-size 1024 --threads 4
```

This makes exactly one shuffled pass over usable training positions from the existing
full-data game split; the original manifest's filenames resolve under local game_data.
Targets use the same own-decision discount and terminal rewards as PPO. Only known
fow-v1 gameplay outcomes are included; protocol failures and unfinished/invalid games
are excluded and counted. Held-out games never enter the optimizer.

Both encoders, actor heads and BN buffers remain frozen. Only linear+tanh value heads
train, with each checkpoint's saved critic learning rate and global clipping0.5.
Reports include before/after full-validation regression metrics, online training error,
head gradient statistics, dataset/exclusion counts, progress and a comparison PNG.
The report is out/rl/value-pretrain-after-400.json/.png. Warmup metadata is embedded in
each policy's training log. Existing evaluated best snapshots remain intact.

The two updated heads commit together into their existing policy.pt files; actor Adam
moments and saved RNG states are preserved. Critic Adam moments are cleared for the
next on-policy phase. --after-update guards against applying a warmup at the wrong
stage; rerunning an already-recorded warmup skips it. Continue paired training with
--updates800 for400 additional updates. Warmup does not count as PPO updates.
PPO plots mark the warmup boundary. This is historical outcome prediction followed by
on-policy adaptation, not an off-policy PPO actor update.

## Self-play and mixed opponents

`--self-play` with no explicit training opponents uses a frozen copy of the
current actor, plus recent saved selves as they become available. It still
evaluates against the default frozen imitation baselines. To mix opponents:

```bash
.venv/bin/python rl/train.py \
  --init-checkpoint out/experiments/residual-medium-27890681/best_policy.pt \
  --self-play \
  --opponent out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt \
  --opponent-command 'bash starters/python/run.sh' \
  --updates 200
```

A new current-self snapshot is frozen for each rollout collection; its weights
never change mid-game. `--snapshot-every 10 --self-play-history 4` adds up to four
recent past selves to the sampling pool. Recent past-self weights are embedded in the same policy.pt, bounded by the history
size; no separate snapshot model files are produced. Opponents are
sampled uniformly, then paired across colors: each selected opponent gets one
white/one black learner game. Actor updates start after the entire rollout.

Frozen recurrent checkpoints are supported as opponents, with memory updated
once per observation and reset each game. **Recurrent PPO actors are rejected**:
they need sequence-based minibatches and hidden-state bookkeeping rather than
the current feedforward minibatch implementation.

Use `--eval-opponent PATH` repeatedly and/or `--eval-command COMMAND` to choose
the evaluation pool. Scores select the embedded `best_policy` weights by the equally weighted
mean score across frozen evaluation opponents. The default pool is a known
baseline benchmark, not an untouched test distribution. To avoid accidentally
overfitting an opponent, evaluate exported policies against additional opponents
and use enough games. `--eval-every 0` disables evaluation and best selection;
export the latest weights for those runs (there is no evaluated best).

## PPO details and diagnostics

- Reward is terminal +1 win, -1 loss, +0.1 draw, from the learner's color.
- Complete episodes are collected. GAE/discounting are over **learner decisions**,
  not all plies; opponent turns are not learner transitions. Defaults are
  gamma .99 and lambda .95. Terminal bootstrap is zero.
- `--max-plies 256` uses the referee's explicit draw cap. It is not a rollout
  truncation bootstrapped as if the game continued. Change the cap for training
  and evaluation together if longer games matter. Draw-cap frequency is logged.
- Masks come from the existing observation tracker/visible safety helper; queen
  promotion matches the imitation action encoding. The rollout saves the exact
  mask, action, old log probability and old value for every learner decision.
  PPO recomputes probabilities with **the same saved mask and temperature**.
- PPO uses clipped policy ratios (.2), normalized advantages, MSE value loss
  (coefficient .5), entropy bonus (.01), four optimization epochs and minibatches
  of 128 by default. Approximate KL above 1.5 times the .02 target stops further
  steps; this is a guardrail, not a strict trust-region guarantee.
- Actor/encoder Adam LR starts at 1e-5; the fresh value head uses 1e-3. No weight
  decay or LR schedule is added to this first PPO recipe. Global gradient clip
  is .5, distinct from the imitation clip 5. These are initial test settings,
  not tuned/proven optimal parameters for this game.
- Dropout stays off and BN uses fixed running statistics throughout both
  collection and PPO updates. Trainable encoder/BN affine/policy/value parameters
  still receive gradients. No stale train/eval mode mismatch is introduced into
  the PPO ratio. This does not claim to fix the earlier learned feature imbalance.
- Temperature defaults to the source checkpoint metadata, or .9 for historical
  imitation checkpoints, and is stored
  in PPO exports. Historical imitation checkpoints still default to .9.

Each persistent variant directory in `out/rl/` contains:

- `metrics.json`: all PPO updates and separate fixed-seed evaluations; rollout
  W/L/D, opponents/colors, lengths, reasons, timing, actor/value losses, masked
  entropy, endpoint KL, ratio clipping, explained variance before/after updates,
  gradient clipping, layer gradient norms and actual parameter update ratios.
- `training.png`: overview of reward, score, critic loss, KL, entropy and ratio clipping.
- `rewards.png`: average terminal reward per game, separate from standard game score.
- `losses.png`: actor, critic and total PPO losses (not directly comparable to supervised cross-entropy).
- `value_quality.png`: MAE, RMSE and explained variance versus Monte Carlo discounted
  terminal returns. Fresh-rollout predictions are measured before learning from those
  games; after-update values show fit on that same batch. Frozen-opponent evaluation
  supplies separate regression measurements. Explained variance is null for constant
  targets, not fabricated as zero. Value is regression, so classification accuracy is
  not the relevant measurement. GAE-target errors are also saved in JSON.
- `outcomes.png`, `evaluation_opponents.png`: outcome rates, length and each frozen
  opponent's score/reward. Reasons in JSON distinguish repetition and cap draws.
- `policy_health.png`: KL, entropy, ratio clipping and gradient norms/clipping.
- `throughput.png`: games/decisions per collection second and cumulative training data.

JSON is saved after every completed update; plots refresh at checkpoint intervals
(default every 10 updates) and at the final update. Rollout metrics include reward
spread/standard error, per-color/per-opponent summaries and cumulative games/decisions.
The reward standard error is descriptive; shared policies/opponents mean games need
not be independent. Training reward can change when the sampled opponent mix changes;
frozen-opponent curves are the more comparable benchmark. Regression errors are
weighted equally per learner decision, so longer games contribute more observations;
evaluation plot aggregates weight opponents equally. The after-update fit is not
out-of-sample value accuracy.

Draw +0.1 applies to both players, making rewards non-zero-sum. Standard score still
uses win=1, draw=0.5, loss=0, including best-checkpoint selection. The bonus rewards
draw outcomes; it does not prove the moves were blunder-free. Track capped/repetition
draw counts to distinguish survival from stronger play.

Additional artifacts:

- `diagnostics/`: a fixed observation probe from the first captured rollout, exact
  layer activation/concentration/dormancy statistics and weight/activation histogram
  bins before optimization and at checkpoint/final updates. These reuse the shared
  diagnostics, isolate their RNG, and are not a held-out validation set.
- `policy.pt`: the only RL model file. Its top-level state_dict, optimizer, RNG,
  counters and logs represent the latest completed update. Its embedded best_policy
  contains a cloned copy of the highest-scoring policy (earlier wins ties).
  Recent self-play snapshots are embedded too. The file is replaced atomically after
  every update and immediately whenever evaluation finds a new best. No separate
  last/final/numbered/best checkpoint files accumulate.
- Optional `--save-replays`: standard full referee replay files for training and
  evaluation. Actors do not consume the replay's hidden-board fields.

Evaluation RNG is isolated from training RNG and the evaluation seed is fixed.
This aids comparison; it does not replace independent evaluation replications.
Layer relative updates are undefined/null when their initial parameter norm is
zero, such as the newly initialized value head.

## Continue instead of restarting

PPO uses constant learning rates and supports extending the completed update budget.
For the medium residual, repeat the initialization command with a higher total:

```bash
.venv/bin/python rl/train.py \
  --init-checkpoint out/experiments/residual-medium-27890681/best_policy.pt \
  --updates 400
```

If its RL file exists, the trainer resumes it automatically, including its saved
optimizer and recipe. It does not restart from the supplied supervised weights.
Alternatively name that RL file explicitly:

```bash
.venv/bin/python rl/train.py \
  --resume out/rl/residual-c32-h128-518d814dce/policy.pt --updates 400
```

`400` means total updates, not 400 more. Model/BN state, Adam moments, Torch and
Python RNG, metrics, best policy and past-self pool are restored. Other recipe
settings are preserved; device/threads may change. Exact continuation is only
expected on the same deterministic hardware/settings. A per-directory file lock
rejects concurrent writers. Atomic saves protect the preceding checkpoint if a
process is interrupted while writing. An interrupted partial update is recollected
from the last completed state.

The current baseline variant names are:

- Medium residual: `residual-c32-h128-518d814dce`.
- Full plain CNN: `cnn-c64-h256-8ebd1cf653`.

Historical draw-zero PPO files cannot silently resume into draw+0.1. They may be
used as an initialization source if a canonical RL policy does not exist yet;
the actor/critic weights are preserved and the optimizer/log starts fresh. If the
variant already has its persistent RL file, that file takes precedence. Existing
historical files are not deleted. An explicit `--run-dir` is retained for isolated
experiments/test fixtures; use the defaults for the one-file-per-variant workflow.
Original frozen opponent paths must remain available when resuming.

## Export and evaluate using existing bots

```bash
.venv/bin/python rl/export.py \
  --checkpoint out/rl/residual-c32-h128-518d814dce/policy.pt --output build/ppo-agent

.venv/bin/python evaluate.py build/ppo-agent \
  out/experiments/arena-medium-residual-vs-full-cnn-100games/full_cnn --games 100
```

Export selects the embedded best policy by default; add `--latest` to export
current training weights. The exported bot contains only inference weights, without
optimizer/history. The output contains the existing standalone neural bot, shared runtime, and PPO
weights. It speaks the same game contract as every other bot. The bundled loader
honors PPO's saved temperature; the original checkpoint topology is unchanged.

## Cluster

Sync the new `rl/`, updated shared loader, and referee with the existing code-only
rsync workflow. The existing `.venv-cnn` dependencies suffice. Pass the actual
server checkpoint paths; old cluster runs may still live in `out/cluster/`:

```bash
cd ~/fow
mkdir -p logs out/rl
sbatch rl/ppo.sbatch \
  --init-checkpoint out/cluster/residual-medium-27890681/best_policy.pt \
  --opponent out/cluster/residual-medium-27890681/best_policy.pt \
  --opponent out/cluster/full-cnn-27800942/best_policy.pt \
  --eval-opponent out/cluster/residual-medium-27890681/best_policy.pt \
  --eval-opponent out/cluster/full-cnn-27800942/best_policy.pt \
  --updates 200 --games-per-update 16
```

The script requests one GPU, four CPU cores, 32 GiB and six hours. Results go to
`out/rl/residual-c32-h128-518d814dce/` for this variant; the Slurm job ID no longer
creates a new model directory. Monitor `logs/ppo-JOBID.out/.err`. Repeat the same
sbatch command with a higher --updates total to continue, or use:
`sbatch rl/ppo.sbatch --resume out/rl/residual-c32-h128-518d814dce/policy.pt --updates 400`.
No cluster training is launched by installing this infrastructure.

## Validation and references

`python -m unittest test_rl` checks canonical-referee parity between Python and
wire agents, reward/GAE math, action masks and old log probabilities, BN/dropout
consistency, evaluation RNG isolation, exact completed-run extension, and export
compatibility. A pipeline smoke run is not evidence that the initial PPO recipe
improves win rate.

- [Schulman et al., PPO](https://arxiv.org/abs/1707.06347).
- [Huang et al., The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

An explicit `--gae-lambda 1` may be supplied when resuming (including paired training). Omit it to retain the saved lambda. The change preserves model/optimizer/RNG state and is recorded with its update boundary in `training_log.recipe_changes`; new update metrics record gamma and lambda. Other recipe changes remain rejected on resume. With complete games, lambda1 makes value targets equal discounted terminal returns.
