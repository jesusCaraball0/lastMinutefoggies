"""Play a trained feedforward or recurrent policy using the arena protocol."""
import sys
import math

import chess
import torch

from model import create_policy, encode
from moves import OBSERVATION_CHARACTERS, ObservationTracker

TEMPERATURE = 0.9


def load_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = create_policy(checkpoint["kind"], checkpoint.get("activation", "relu"),
                          checkpoint.get("negative_slope", 0.01),
                          **checkpoint.get("model_config", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.recurrent = checkpoint["kind"] == "recurrent"
    # PPO exports retain the exact sampling temperature used for rollouts.
    # Historical imitation policies keep the requested 0.9 behavior.
    model.temperature = float(checkpoint.get('inference_temperature', TEMPERATURE))
    if not math.isfinite(model.temperature) or model.temperature <= 0:
        raise ValueError('Checkpoint sampling temperature must be positive and finite')
    return model


def main(checkpoint):
    if sys.stdin.readline().strip() != "fow":
        return
    color, _, _ = sys.stdin.readline().split()
    # Games run on CPU; avoid thread-pool overhead for a single small position.
    torch.set_num_threads(1)
    model = load_policy(checkpoint)
    tracker = ObservationTracker(color == "w")
    hidden = logits = None
    with torch.inference_mode():
        model(torch.zeros(1, 14, 8, 8))
        print("ready", flush=True)
        for raw in sys.stdin:
            line = raw.strip()
            if len(line) == 64 and set(line) <= OBSERVATION_CHARACTERS:
                tracker.observe(line)
                if model.recurrent:
                    logits, hidden = model(encode(line).unsqueeze(0), hidden)
                continue
            float(line)  # Only respond after receiving the clock prompt.
            moves = tracker.legal_moves()
            if not moves:
                return
            # The training target merges promotion types; prefer queen promotion.
            moves = [move for move in moves if move.promotion in (None, chess.QUEEN)]
            if not model.recurrent:
                logits, _, _ = model(encode(tracker.current).unsqueeze(0))
            indices = [move.from_square * 64 + move.to_square for move in moves]
            # Normalize only legal candidates; each new process gets fresh RNG state.
            probabilities = torch.softmax(logits[0, indices] / model.temperature, dim=0)
            move = moves[torch.multinomial(probabilities, num_samples=1).item()]
            tracker.record_move(move)
            print("move " + move.uci(), flush=True)
