# Shared neural bot code

- `model.py`: feedforward and recurrent architectures, observation encoding,
  and the training interface. Training datasets remain in `training/`.
- `moves.py`: observation history, legal move generation, and visible king safety.
- `policy_bot.py`: persistent arena protocol and checkpoint inference, sampling
  from the legal-move softmax at temperature 0.9 (fresh randomness each game).
- `bundle.py`: packages one bot with a frozen checkpoint and shared runtime.

Each bot directory owns its entrypoint, checkpoint, and launch/build scripts.
Local entrypoints import this directory. Bundled submissions import their own
included `shared/` directory. A future PPO bot can reuse these modules and expose
the same `run.sh` interface, with its own inference code when needed.

Move tracking uses the root `referee.py` as the canonical source of game rules.
The bundler includes that module in generated standalone submissions.

Before sampling, move generation simulates each candidate and filters moves
that leave the king attacked by visible enemy pieces along known clear paths.
Hidden attackers and unknown blockers remain uncertain. If no candidate passes,
the bot retains the legal candidates so it can still submit a move. The referee
rules are unchanged; FOW castling checks only final-position safety here.

New training runs can use LeakyReLU in the encoder, with activation and slope
stored in checkpoint metadata. The bot reads those settings; checkpoints that
omit them use the historical ReLU architecture. Weight shapes and encoder layer
indices are unchanged, so existing checkpoints retain their original behavior.
