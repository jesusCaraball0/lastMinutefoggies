# Trained neural policy deliverable

This folder is the self-contained submission used by the top-level evaluator.
Include the trained `policy.pt` and ZIP this folder's contents for the arena.

From the repository root, install and train:

```sh
.venv/bin/python -m pip install -r training/requirements.txt
.venv/bin/python training/train.py
.venv/bin/python evaluate.py
```

Training saves the best checkpoint here. Constants in `training/train.py`
control training; constants in `evaluate.py` control the 20-game baseline test.
There is no command-line argument parser in either script.

Set `MODEL_TYPE` to `"feedforward"` or `"recurrent"` in the same trainer. Both models are defined in `model.py`. For the recurrent policy, a CNN encodes each
observation, a GRU maintains a 256-dimensional memory, and a linear head scores
moves. Ordered player-game sequences come from `training/datasets.py`.
The shared training loop carries memory across truncated chunks and ignores
opponent-turn and padding labels. Choose a different `OUTPUT` to preserve an
existing checkpoint, and set the evaluator's `MODEL_PATH` to test that file.

`bot.py` implements the arena's persistent stdin/stdout protocol. It loads
`policy.pt` relative to its own file, uses PyTorch CPU inference, and preserves
the observation tracker between moves. It processes every observation but
outputs a move only after the clock prompt. All stdout is reserved for the
protocol. Legal candidates include castling and en passant; promotion chooses
a queen. The policy sees only the player's fog observation. The unused value
head remains in `model.py` for compatibility with existing training checkpoints.

`build.sh` checks that the supplied checkpoint loads. `run.sh` launches the bot.
The interface selects feedforward or recurrent inference from the checkpoint's
`kind` field. Recurrent memory updates once per observation, including opponent
turns, and resets for each new game. There is no ONNX export,
search, or random fallback weights. Initialization must fit the 15-second arena
limit. PyTorch and python-chess are provided by the course runtime.

The evaluator runs a copy of this deliverable and freezes its weights for the
whole series. It saves replays and prints W/L/D, results by color, score rate,
game lengths, failures, and timing. See the root README for details.
