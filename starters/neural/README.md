# Feedforward neural bot

Experimental findings live in [EXPERIMENT_HISTORY.md](../../EXPERIMENT_HISTORY.md); this file documents the bot interface and usage.

This bot owns its `policy.pt` and uses the model definitions, observation tracker,
and arena protocol implementation in `starters/shared/`.

From the repository root:

```sh
.venv/bin/python starters/neural/bot.py
```

For matches, use `bash starters/neural/run.sh` as a referee bot command.
Activate `.venv` first so `run.sh` and `build.sh` use the installed dependencies.
`build.sh` validates the checkpoint. The bot uses CPU inference, samples legal moves
from `softmax(logits / 0.9)`, and chooses queen promotions. Recurrent memory
updates once per observation and resets each game.

Train with `python training/train.py --model-type feedforward`.
The run saves its own best and final checkpoints under `out/`; it does not
replace this bot's weights. Use `--output` with a fresh checkpoint path for an
additional export. See `training/CLUSTER.md` for GPU cluster instructions.
To evaluate against the other CNN bot, run:

```sh
.venv/bin/python evaluate.py neural neural_recurrent
```

The evaluator plays 20 games, alternating colors, and reports results from the
first bot's perspective. Both checkpoints are frozen for the series.

Create a standalone arena submission with shared code included:

```sh
.venv/bin/python starters/shared/bundle.py starters/neural --output build/neural
```

Use a fresh output directory. ZIP the contents of `build/neural/` for submission.
The bundled `build.sh` and `run.sh` work without the repository. PyTorch and
python-chess must be available in the runtime; initialization must fit 15 seconds.
