# Fog of War Chess starter kit

## Download public replays

Run `python download_games.py` to save full completed-game replays under
`game_data/<game-id>.json` when both teams' **current Elo is at least 1800**.
The script snapshots the leaderboard once per run, matches teams by ID, and
paginates through the entire completed-game listing. Games with missing team
IDs or teams absent from the leaderboard cannot be classified and are counted
as `unknownTeams`. Eligibility uses current ratings, not ratings at game time.

Requests are sequential, with a shared cap of 9 public reads per second,
including retries. Replay downloads use this rate; game-listing requests also
have a separate cap of 1 per second for the stricter archive-search budget.
Use `--interval` to increase the spacing for all requests or
`--listing-interval` to increase the spacing between game-listing requests.
Temporary failures use bounded exponential retries with jitter and respect
`Retry-After`. Unavailable (404) replays are reported, and the script exits
with status 1 if any qualifying replay is unavailable. Other permanent failures
stop the download. Completed files survive interruptions; reruns skip valid
saved replays and retry missing or malformed ones.

Each game retains the complete API replay, including full boards, both players'
observations, moves, clocks, results, and available rules/version metadata.
`downloadMetadata` adds selection-time ratings, timestamps, and the game listing.
Leaderboard, ID-to-Elo map, and last completed run summary are in
`game_data/_metadata/`. Reruns fetch fresh ratings and retain previously saved
games, even if those teams later fall below the threshold. Use a fresh output
directory for a dataset containing only games eligible under a new snapshot.

```sh
python download_games.py
python download_games.py --min-elo 1600 --interval 2 --output game_data_1600
python replay.py game_data/<game-id>.json
```

Install dependencies with `python -m pip install -r requirements.txt`. The downloader
shows a `tqdm` progress bar with games scanned, qualifying games, downloads, and
previously saved files. The percentage and ETA refer to the full game scan.
Downloaded JSON files are
already excluded by this project's `.gitignore`.

Choose a starter from `starters/`, edit it, and upload that folder as a ZIP.
The kit includes Python, C++, Rust and CPU neural examples, plus a referee for
local testing. Submit your bot's folder, not the entire kit.

## Quick start

1. Copy one folder from `starters/python`, `starters/cpp`, `starters/rust`, or
   `starters/neural`. The Python default, `uniform_random.py`, samples uniformly
   among all legal moves, including castling, en passant and all four promotion
   choices.
2. Edit the bot. Keep `run.sh` in the ZIP root. C++, Rust and neural examples
   include `build.sh`; the basic Python bot runs directly. The server invokes
   these with Bash, so you do not need to mark the scripts executable. Upload
   source, scripts and model weights together.
3. ZIP the contents of that folder. A single enclosing folder is also accepted.
   Leave out your virtual environment, `.git` and compiler caches. Symlinks are
   not accepted. The server runs every submission in the course runtime
   described below.
4. Open your instructor-provided team link to connect and sign in, then upload
   the ZIP on the Submit page. You can also enter your course-issued team token
   after connecting manually. Keep the link and token private; never put them
   in source code or the ZIP. Reopen your team link after reloading the page.
5. Automatic activation is checked by default: once the build and tests pass, the
   version joins the arena even if you have left the page. Uncheck it before
   uploading if you prefer to activate manually. Use **Cancel build/test** to
   stop a queued or running build or test. Open **Build & test logs** to follow
   progress.
6. To switch bots later, choose **Activate version** on any version that passed
   validation. Already assigned matches finish with their original versions.

The upload limit is **128 MiB compressed, 512 MiB expanded, and 10,000 entries**.
The compiled bundle has the same limits. There is a 30-second interval between
uploads and a limit of two pending builds per team. Team storage and version
quotas also apply.

## Versions and Elo

Your team starts at 1500 Elo. Every rated game updates that same rating with
K=8, across all versions. Switching versions does not reset your rating or remove
past results. The reference team, `baseline.py`, stays at 1000.

The Leaderboard ranks teams by Elo. Its win/draw/loss counts cover your active
version; the rating includes all versions. Matchmaking chooses the first team
uniformly at random, then favors opponents with similar Elo. Each pair plays
both colors.

Matches run until the competition deadline. Games assigned before the deadline
finish and count.

## Test your own code locally

Open a terminal in this extracted kit, then run:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python referee.py --trusted-local \
  --white '.venv/bin/python uniform_random.py --seed 1' \
  --black '.venv/bin/python uniform_random.py --seed 2' \
  --white-name 'Uniform 1' --black-name 'Uniform 2' \
  --base-time 180 --increment 2 --max-plies 1000 --output example-match.json
```

The referee saves the game data to `example-match.json` for analysis. Replace either
bot command with your own executable. The local runner executes programs on
your machine without a sandbox: use it only for code you trust. The course
server runs submissions in isolated sandboxes.

Compile the C++ or Rust starter by running `bash build.sh` inside its language
folder. The resulting `bot` executable can be supplied to the local referee.
The server compiles your source for Linux amd64; do not upload a Mac executable.

## Stdin / stdout protocol

Each bot is one persistent process. It first receives:

```text
fow
w 180 2
```

The second line gives your color (`w` or `b`), base time in seconds and increment
in seconds. After reading both lines, initialize your bot, write `ready` followed
by a newline, and flush `stdout` within **15 seconds**. For example, in Python:

```python
print("ready", flush=True)
```

Initialization is outside the game clock. Both the local referee and the arena
require this response. The referee sends no observations until both bots are
ready; a missing or late response loses the game.

Before **every ply** (one player's move), both players receive a 64-character
observation in a1, b1, …, h8 order. Uppercase pieces are White, lowercase pieces
are Black, `-` is a visible empty square, and `?` is unseen. For example, White
initially sees:

```text
RNBQKBNRPPPPPPPP----------------????????????????????????????????
```

Only the moving player then receives its remaining time as a numeric line.
Reply with `move`, a space, and a move in UCI notation, such as:

```text
move e2e4
```

End the line with a newline and flush `stdout`. For promotion, append `q`, `r`,
`b` or `n` to the move; for example, `move e7e8q` promotes to a queen.
Never send a move before receiving a clock prompt, including during the
opponent's turn. Any other `stdout` line, including a blank line or repeated
`ready`, forfeits the game and stops the bot;
use `stderr` for diagnostics. `stdout` is limited to 4 KiB per line, 1 MiB per game
and 256 queued lines. Illegal moves, expired clocks, malformed responses,
premature exit and excessive `stdout` lose the game. Excessive `stderr` produces
a warning; see [Debug output and private logs](#debug-output-and-private-logs).

During play, your bot receives only its observations and clock prompts. It
receives no final king-capture prompt or final observation when the game ends.
Afterward, the downloadable replay includes the full board and both players'
observations.

Games use **180 seconds + 2 seconds per move**. Your bot's entire process,
including child processes and threads, is paused after `ready` and after each
move. It resumes on its next turn, with memory preserved and any queued
observations waiting on `stdin`. It cannot compute during the opponent's turn.
Time spent paused, or pausing and resuming, is not charged to the game clock.

## Rules used by the referee

`referee.py` uses the same board rules as the arena. See the
[Rules tab](https://www.mit.edu/~6.7980/fow/?tab=rules) for the full explanation
and examples of unusual situations. The main differences from ordinary chess are:

- There is no check or checkmate restriction. A move that leaves the moving
  player's king capturable loses immediately. Merely threatening the opponent's
  king gives that opponent its normal next turn to escape.
- Castling may start, cross or finish on attacked squares. The king and rook
  must have their castling rights, with no pieces between them. Castling into
  attack is legal but loses after the move.
- Movement, captures, promotion and en passant otherwise follow ordinary chess.
- You see your pieces and the destinations of all moves allowed by these rules,
  even moves that would expose your king. An opposing piece blocking a pawn's
  forward move stays hidden; a pawn capturable en passant
  is visible even when the capture would be pinned in orthodox chess.
- Threefold repetition or 100 plies without a capture or pawn move draw
  automatically. There is no insufficient-material draw. Having no legal moves
  means a loss.
- Games that reach 1,000 plies end as rated draws.

The Python starter uses only its own observations and remembered castling rights.
It infers en passant from consecutive observations. With no `--seed`, it uses
operating-system entropy; supply a seed for repeatable development games.
The separate `baseline.py` prefers captures and omits en passant inference.
The smaller C++ and Rust examples omit both castling and en passant inference.
They are starting points for your own strategy.

## Runtime and resource limits

The course runtime provides the following software. No GPUs are available.

| Software | Version | Notes |
| --- | --- | --- |
| Python | 3.14.8 | Standard CPython |
| NumPy | 2.3.5 | |
| python-chess | 1.999 | Includes chess 1.11.2 |
| PyTorch | 2.10.0+cpu | CPU only |
| JAX | 0.11.2 | CPU only |
| jaxlib | 0.11.2 | CPU only |
| ONNX Runtime | 1.24.2 | Python package and native C/C++ CPU SDK |
| ONNX | 1.20.1 | |
| C++ compiler (`g++`) | 12.2.0 | C++20 supported |
| Rust compiler (`rustc`) | 1.85.1 | Starters use the 2021 edition |

The native ONNX Runtime C/C++ SDK is at `/opt/onnxruntime`; see
the C++ guide at `starters/cpp/README.md` inside the starter ZIP for the include,
linker and runtime-library flags.
Existing versions keep their original runtime; submit a new version to use an
added runtime dependency.

JAX runs on CPU. Import libraries and compile any JIT functions you need before
printing `ready`; this initialization must fit within the 15-second allowance.
Each game starts a fresh process, so an in-memory JIT cache does not carry over
from the build or another game.

Python dependency installation and Cargo/network downloads are unavailable in
student builds and games. Include additional dependencies inside your ZIP
and build Rust packages with `cargo build --release --offline` when using Cargo.

Builds and games run in a gVisor sandbox with these limits:

| Resource | Limit |
| --- | --- |
| CPU | 2 physical cores / 4 hardware threads |
| Memory | 4 GiB per bot |
| Processes | 64 |
| Writable workspace | 768 MiB, stored in RAM |
| `/tmp` | 128 MiB, stored in RAM |
| Build time | 5 minutes |
| Bot initialization | 15 seconds, before the game clock starts |
| Network | Unavailable |

The runtime filesystem is read-only except for the workspace and `/tmp`.
Both writable areas count toward the memory limit, along with model weights
and working memory. No changes to memory or files survive into the next game.
Sandbox setup has a separate 10-second limit before your bot receives its
protocol headers.

Both bots in a game share the same cores. Your opponent is paused on your turn;
other matches use separate cores. The runtime defaults OpenMP, OpenBLAS and MKL
to one thread. To use more, configure the library's thread count explicitly.
Benchmark one, two and four threads: four hardware threads still share two
physical cores.

## Validation

Validation runs a full game against the baseline in both colors under the same
180+2 time control and resource limits. Winning, drawing or losing normally all
pass validation. A timeout or confirmed out-of-memory kill during play produces
a warning, and testing continues in the other color. Other crashes,
initialization failures, illegal moves and malformed protocol output fail
validation. The runtime must confirm an OOM kill; exit code 137 or a printed
OOM message alone is not enough.

Validation games do not count toward Elo. Timeouts and OOM in rated games are
losses.

## Neural policy: train, evaluate, and submit

Training code and dependencies live in `training/`. Run from the repository
root; command-line options override the defaults shown by `training/train.py`:

```sh
.venv/bin/python -m pip install -r training/requirements.txt
.venv/bin/python training/train.py
```

The default is the full feedforward CNN: 64 channels, 256 features, ReLU,
BatchNorm after each convolution, default initialization, and global L2 gradient
clipping at 5. AdamW uses betas (0.9, 0.99), epsilon 1e-8, and weight decay 0.01.
Eight epochs use batches of 128. LR warms from 1e-4 to 1e-3 over 5,820 updates,
then cosine-decays to 1e-5 across the remaining updates. The total update budget
is derived from the dataset and printed and saved before training starts.
All training games in the fixed 90/10 game split are used. Raw validation covers
the whole validation split; legal-action diagnostics use 8,192 held-out moves
by default (`--probe-max-positions 0` evaluates all validation moves).

Use `--model-type recurrent --no-batch-norm --schedule none --grad-clip-norm 1`
for the CNN → GRU model with its original clipping threshold. Both architectures
use shared models, observation encoding, metrics, and checkpoint loading.
See [training/CLUSTER.md](training/CLUSTER.md) for ORCD setup, the included
six-hour Slurm job, monitoring, resume, and result download commands.

The recurrent model keeps a 256-dimensional memory. Its separate dataset in
`training/datasets.py` builds an ordered trajectory for each player in
each game, including observations from opponent turns. Only the player's own
moves have labels; opponent turns and batch padding are excluded from the loss.
Memory starts at zero per player-game and carries between chunks, with gradients
truncated every `SEQUENCE_LENGTH` observations (default 16). Set
`SEQUENCE_BATCH_SIZE` for recurrent batches; `BATCH_SIZE` controls feedforward
batches. Validation follows the same sequence order and memory updates.

The trainer uses both players' moves from all completed wins, losses, and draws.
Each input is the moving player's observation **before** the recorded move;
it never uses the full board. The selected policy predicts 4,096 from/to
classes using cross-entropy loss. Promotion types share a class. White and Black
retain their absolute coordinates and piece colors. The 10% validation split
holds out whole games, using a fixed seed. Every epoch shows training and eval
progress bars, then prints loss and top-1/top-5 move accuracy. These accuracies
are measured before legal-move filtering. Training selects CUDA, MPS, or CPU
automatically; set `DEVICE` to override.

Use `--activation leaky_relu --negative-slope 0.01` to compare activations.
Activation settings and BatchNorm architecture are stored in checkpoints and
honored by inference. Historical checkpoints without those fields retain their
original architecture and ReLU activation.

For the long-update stability ablations (baseline, no decay on normalization/bias,
fixed BatchNorm affine parameters, and BatchNorm after the dense layer):

```sh
.venv/bin/python training/stability_experiment.py
```

This runs four processes with one Torch thread each. All cases use the same
2,560 training games, 64 validation games, seed 7980, shuffled full batches of 128,
and a 16-channel / 64-feature CNN. AdamW settings and global clipping at 5 match
the cluster run. The LR follows the **first 80,000 updates of the cluster's
334,440-update schedule**, with 5,820 warmup updates. Checkpoints are saved at
5k, 20k, 40k, and 80k. The LR is intentionally not forced to its minimum when
the stress test stops. `--parallel`, `--threads`, `--checkpoints`, and network/data
size arguments allow adjusting the workload.

This repeats a smaller dataset much more often than the full-data run; treat
validation loss as an ablation comparison, not a full-scale performance forecast.
A baseline that does not reproduce the instability cannot establish its fix.
Activation diagnostics use a fixed random sample of 512 held-out observations.
Checkpoints record weight-decay groups, BN scales, feature concentration, actual
updates, and batch-stream hashes. Final legal-action metrics cover the entire
held-out split. Results and comparison plots are saved under
`out/stability-experiments/<timestamp>/`; production defaults stay unchanged.

For the matched dropout comparison (baseline, dense dropout 0.1 and 0.2, and
dense BatchNorm plus dropout 0.1), use the same recipe and optionally reuse the
exact cached split from a previous round:

```sh
.venv/bin/python training/stability_experiment.py --suite dropout \
  --reuse-data out/stability-experiments/20261103T221924-729044Z \
  --out-dir out/dropout-experiments
```

Dropout follows the hidden dense ReLU, is disabled for validation/inference, and
its probability is saved in `model_config`. Model defaults retain zero dropout.

For the architecture comparison (dropout-0.1 reference, three residual blocks,
and a 1x1 convolution reducing trunk channels to one quarter before flattening):

```sh
.venv/bin/python training/stability_experiment.py --suite architecture \
  --reuse-data out/dropout-experiments/20261104T002130-541749Z \
  --reference-run out/dropout-experiments/20261104T002130-541749Z \
  --parallel 2 --out-dir out/architecture-experiments
```

The reference case is copied from the completed matching run; only the two new
architectures train. The residual variant uses a stem and three post-activation
blocks (seven 3x3 convolutions total). It retains the dense head and dropout 0.1.
The reduced-head variant retains the two-convolution trunk and adds a 1x1
convolution plus ReLU before flattening; it keeps the hidden width and dropout.
New architecture fields are saved in checkpoints. Existing model defaults and
historical checkpoints retain their original topology. Data hashes and complete
learning-rate traces must match before the comparison is marked complete.

For the residual LR-decay follow-up on the exact same cached dataset:

```sh
.venv/bin/python training/residual_decay_experiment.py
```

This changes only the cosine horizon to 80,000 updates (same 5,820-step warmup),
evaluates the entire validation split every 5,000 updates, and preserves the
training RNG stream across the added evaluations. It saves separate best
checkpoints for raw NLL, legal NLL, legal accuracy, and raw accuracy. Results go to
`out/residual-decay-experiments/<timestamp>/`, and a comparison is appended to the
reference architecture experiment's `report.md`. The reference and output root
can be selected with `--reference-run` and `--out-dir`.

A small local pilot compares both model types, ReLU versus LeakyReLU, and two
training-set sizes. Run from the repository root:

```sh
.venv/bin/python training/experiment.py
```

Defaults: 32 and 128 training games, a fixed 16-game validation set, 8 convolution
channels, a 32-dimensional feature/memory vector, 8 epochs, and four CPU threads.
The full model remains 64 channels and 256 features. The pilot runs eight cases
under `out/experiments/<timestamp>/`, with `experiment.json`, `comparison.png`,
and per-case logs, checkpoints, metrics, and plots. Existing bot checkpoints are
not replaced. Every case records the actual train/validation file list, seed,
parameter count, optimizer-step count, and separate training/diagnostics timing.

For a larger follow-up pilot:

```sh
.venv/bin/python training/experiment.py --train-games 128 512 --epochs 12
```

Use `--models recurrent` or `--activations leaky_relu` to restrict cases;
`--channels`, `--hidden-size`, `--learning-rate`, `--seed`, and `--device` are
also configurable. Within a model type, cases use identical initial weights,
nested seeded training subsets, and the same disjoint validation games. These
short one-seed runs measure early behavior; they do not establish full-scale
convergence, playing strength, or GPU throughput.

For a controlled comparison of CNN size, run:

```sh
.venv/bin/python training/size_experiment.py
```

This uses one seed (7980), 512 training games, 64 fixed validation games, ReLU,
Adam at 0.001, and batches of 128. It compares 8 channels / 32 features against
64 channels / 256 features at exactly 200, 1,000, and 2,000 optimizer updates.
Both models see identical shuffled full batches; SHA-256 hashes verify the
cumulative input streams at each checkpoint. Training metrics average over each
interval between checkpoints; validation metrics describe the checkpoint itself.
Runs use CPU with four threads and save under `out/size-experiments/<timestamp>/`.
Each contains a shared data manifest, `experiment.json`, `comparison.png`, and
per-model metrics, checkpoints, plots, initial activation statistics, and final
histograms. Existing bot checkpoints are not overwritten. Optional arguments
include `--checkpoints 200 1000 2000`, `--train-games 512`, `--val-games 64`, and
`--threads 4`. A single seed gives a diagnostic comparison, not an estimate of
variation across training seeds.

To repeat the size comparison with AdamW and weight decay 0.01:

```sh
.venv/bin/python training/size_experiment.py --optimizer adamw --weight-decay 0.01
```

For AdamW with learning rate 1e-4 and betas (0.9, 0.99):

```sh
.venv/bin/python training/size_experiment.py --optimizer adamw \
  --learning-rate 1e-4 --beta-1 0.9 --beta-2 0.99 \
  --weight-decay 0.01 --eps 1e-8
```

For only the small CNN, with three epochs of linear warmup from 1e-4 to 1e-3,
then five epochs of cosine decay to 1e-5:

```sh
.venv/bin/python training/size_experiment.py --sizes small --optimizer adamw \
  --learning-rate 1e-4 --beta-1 0.9 --beta-2 0.99 \
  --weight-decay 0.01 --eps 1e-8 \
  --warmup-epochs 3 --decay-epochs 5 --peak-lr 1e-3 --end-lr 1e-5
```

Use `--sizes medium --train-games 2560` with the same optimizer and schedule
arguments to train the medium CNN (32 channels / 128 features) on 2,560 games.
The small CNN has 8 channels / 32 features; original has 64 / 256.

Add `--conv-init he_normal` to initialize only Conv2d weights with He normal
(`fan_in`, ReLU gain). Biases and linear layers keep their original initialization.
The default remains unchanged. Initialization choice is recorded in settings,
per-model metrics, and checkpoints.

Add `--batch-norm` for `Conv2d → BatchNorm2d → ReLU` in both convolution blocks.
It defaults off, keeps existing convolution/ReLU names and weight initialization,
and records the architecture choice in checkpoint `model_config`. Validation and
bot inference use saved running statistics in eval mode.

For a larger dataset with a fixed schedule in optimizer updates, use
`--warmup-updates 5820 --decay-updates 9700` instead of `--warmup-epochs`.
Also supply `--checkpoints 1940 3880 5820 7760 9700 11640 13580 15520`.
The last checkpoint must equal the sum of warmup and decay updates. This keeps
the LR trajectory and training budget fixed when changing the number of games.

Scheduled runs evaluate and save checkpoints after every epoch, overriding
`--checkpoints`. An epoch is one shuffled pass through all full training batches;
the final incomplete batch is dropped. LR changes after each optimizer update,
and the actual LR used for every update is saved and plotted. Checkpoints also
include optimizer and scheduler state.

The original command continues to use Adam with zero weight decay. Optimizer and
weight decay are recorded in the experiment settings and checkpoints.

Channel, sparsity, and memory-saturation plots are no longer generated. The JSON
retains activation statistics, and `dormant_features.png` tracks inactive
features throughout training.

Every new training run creates `out/<UTC-timestamp>-<model-type>/`:

- `metrics.json`: configuration and per-epoch loss, accuracy, learning rates,
  layer/global gradient L2 statistics, clipping frequency, and validation
  activation statistics. Written atomically after every epoch.
- `loss.png`, `learning_rate.png`, `gradient_l2.png`: epoch curves.
- `gradient_quantiles.png`: layer gradient medians, p95s, and maxima before clipping.
- `gradient_global_l2.png`: mean/median/p95/max global norms before/after clipping,
  and the fraction of optimizer updates actually clipped.
- `histograms.json`, `weights/*.png`, `activations/*.png`: final-epoch weight
  distributions and validation activation histograms, with raw histogram bins.
- `best_policy.pt`, `final_policy.pt`: best-validation and final-epoch checkpoints,
  including activation settings, epoch, configuration, and diagnostics metadata.

Gradient L2 combines a layer's direct weight and bias gradients. Mean, median,
p95, maximum, step count, and nonfinite count are stored before and after clipping.
The same statistics are logged for the global norm. Clipping frequency measures
actual global rescaling by comparing each step's pre/post norms; layers without
any gradient (e.g. the unused feedforward value head) are `null` rather than zero.
Norms are buffered on the training device and summarized once per epoch.

Additional diagnostics are saved each epoch:

- `update_to_weight.png`: actual Adam parameter changes divided by pre-update
  parameter L2, per layer (including bias), with median/p95. JSON also records
  absolute update and parameter norms. Sampling is every 100 optimizer steps
  by default, every 10 in the pilot; the first step of each epoch is sampled.
- `dormant_features.png`: fraction of channels with mean absolute activation
  at most 0.01 times their layer's average. Per-channel mean-absolute activity,
  normalized activity scores, and variation of channel means across observations
  are included in the activation statistics. Low activity is not the same as
  never-positive output; a useful LeakyReLU channel may stay negative.
- `legal_policy.png`: raw probability mass on game-legal actions, raw versus
  legal-masked NLL/accuracy, and safety-heuristic intervention rates. The masks
  are built once from fixed held-out replay observations and known castling/EP
  history, without reading hidden board fields. The recorded move must belong
  to the game-legal mask, but may be rejected by the safety heuristic; these
  masks are kept separate. Diagnostic probabilities use temperature 1.
- `memory_usefulness.png`: recurrent NLL with normal history versus resetting
  hidden state on every observation; positive reset-minus-normal NLL means
  history improves predictions on this probe sample.
- `gradient_by_lag.png`: gradient L2 of a supervised endpoint loss with respect
  to hidden states at preceding lags, inside the truncated window. A no-grad
  prefix initializes history before that window; diagnostics do not accumulate
  parameter gradients. The probe is bounded to two trajectories per epoch.

`PROBE_MAX_POSITIONS`, `PROBE_GRADIENT_TRAJECTORIES`, and `UPDATE_SAMPLE_EVERY`
control overhead. Training and diagnostics durations are reported separately.
Small models save their channel/hidden dimensions in checkpoint metadata, so
bots can load them alongside the original full-size checkpoints.

Validation activation diagnostics use the first 512 nonpadding observations,
preserving recurrent memory. Exact reductions across **all** values in those
observations record mean/std, zero/near-zero (absolute value <= 1e-6), negative,
and positive fractions, plus all-zero and never-positive channel fractions.
Per-channel values are retained for layers with up to 256 channels/features;
large flattened/logit layers retain aggregate statistics. "All-zero" or
"never-positive" describes this validation sample, not a permanently dead unit.

Histograms use uniform random reservoirs of up to 50,000 values per layer,
independent of training RNG; exact activation statistics do not depend on the
histogram sample. Weight histograms include weights, excluding biases. Sample
limits and bin counts are configurable in `training/train.py`. Final histograms
describe the final epoch, which may differ from the best one.

Every new run saves `best_policy.pt`, `last_policy.pt`, and `final_policy.pt`
in its own output directory; existing bot weights are not replaced. Optionally
export the best policy with `--output /path/to/a/new/policy.pt`. Existing export
files and run directories are protected from accidental replacement.

`--resume /path/to/run/last_policy.pt` restores the last completed epoch,
including model/BatchNorm buffers, AdamW moments, scheduler position, RNG state,
and training shuffle-generator state. An interrupted partial epoch is replayed.
Resume preserves the saved training configuration and data split. Checkpoints
and JSON records are written atomically. Validation and diagnostic timings are
reported separately. The architecture and encoding remain shared with inference.

Evaluate any two bot directories (or starter names):

```sh
.venv/bin/python evaluate.py neural neural_recurrent
# Equivalent: .venv/bin/python evaluate.py starters/neural starters/neural_recurrent
```

`evaluate.py` launches each bot's `run.sh` from an independent snapshot and
freezes both checkpoints before the series. It defaults to 20 games, alternating
colors (10 White and 10 Black for each bot), at 180+2 with a 1,000-ply limit.
Use `--games`, `--base-time`, `--increment`, and `--max-plies` to change those
settings. `--games` must be positive and even. For a short launch check:

```sh
.venv/bin/python evaluate.py neural neural_recurrent --games 2 --max-plies 20
```

The first bot is A; wins, losses, per-color results, win rate, and score rate
(a draw counts as half a point) are reported from A's perspective. Reports also
include game lengths, failures for each bot, wall time, and termination reasons.
Replays and `summary.json`, including both bot paths and checkpoint SHA256 hashes,
are saved under `evaluation_results/<timestamp>/`. Override the parent directory
with `--results-dir`. Future PPO bots can use the same evaluator by exposing
`run.sh` and keeping their checkpoint and inference dependencies in their folder
or the shared runtime. Match results measure playing strength; move prediction
accuracy measures imitation.

The bots have separate folders and checkpoints:

```text
starters/
  neural/             # feedforward bot and policy.pt
  neural_recurrent/   # recurrent bot and policy.pt
  shared/             # models, inference, move tracking, submission bundling
```

Each bot's `bot.py` loads its own `policy.pt`; the checkpoint's `kind` selects
the architecture. Shared inference runs on CPU and samples legal moves from
`softmax(logits / 0.9)`. Each game uses fresh randomness.
Before sampling, the observation move generator filters moves that leave the
king attacked by a visible enemy piece along a known clear path. Hidden threats
remain uncertain; if every candidate is unsafe, it retains the legal moves
as a fallback. Move tracking supports castling and en passant; promotion chooses
a queen. A recurrent bot updates memory once per observation, including opponent
turns, and starts each game with zero memory. Neither policy uses value search.

All bots expose the same `run.sh` interface, so future PPO bots can play any
existing bot through the referee. Activate the virtual environment before using
the launch scripts:

```sh
source .venv/bin/activate
python referee.py --trusted-local \
  --white 'bash starters/neural/run.sh' \
  --black 'bash starters/neural_recurrent/run.sh' \
  --white-name neural --black-name neural_recurrent \
  --base-time 180 --increment 2 --max-plies 1000 \
  --output neural-vs-recurrent.json
python replay.py neural-vs-recurrent.json --delay 0.3
```

For arena submission, bundle a bot with the shared code into a fresh directory:

```sh
.venv/bin/python starters/shared/bundle.py starters/neural --output build/neural
# Or: starters/neural_recurrent --output build/neural_recurrent
```

ZIP the contents of the generated directory. Its `build.sh` validates the
checkpoint and its `run.sh` launches the same bot used locally and by the
evaluator. The bundle includes shared code and weights, so it works outside the
repository. Initialization must fit the arena's 15-second allowance; PyTorch and
python-chess are supplied by the course runtime. No ONNX export is required.

## Debug output and private logs

Write debug logs to `stderr`. The arena saves only the last 16 KiB per bot.
A `stderr` line over 4 KiB or more than 1 MiB total produces a warning in
validation and game reports, without causing a forfeit.
Printing still consumes your bot's CPU and clock budget. `stdout` protocol limits
remain enforced.

For a completed game, open the **Games** tab, choose **Log** in its row, then
**Private stderr**, and enter your team token. You can read or download your
bot's log. The token stays in memory only and is cleared when the dialog closes.
Server details are redacted; logs flagged as possibly containing secrets are
withheld for staff review. Missing or expired logs are shown as unavailable.
