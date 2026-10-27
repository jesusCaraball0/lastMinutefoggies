"""Play a trained feedforward or recurrent policy using the arena protocol."""
import sys
from pathlib import Path

import chess
import torch

from model import create_policy, encode
from moves import OBSERVATION_CHARACTERS, ObservationTracker


CHECKPOINT = Path(__file__).resolve().with_name("policy.pt")


def load_policy():
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model = create_policy(checkpoint["kind"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.recurrent = checkpoint["kind"] == "recurrent"
    return model


def main():
    if sys.stdin.readline().strip() != "fow":
        return
    color, _, _ = sys.stdin.readline().split()
    # Games run on CPU; avoid thread-pool overhead for a single small position.
    torch.set_num_threads(1)
    model = load_policy()
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
            move = moves[int(logits[0, indices].argmax())]
            tracker.record_move(move)
            print("move " + move.uci(), flush=True)


if __name__ == "__main__":
    main()
