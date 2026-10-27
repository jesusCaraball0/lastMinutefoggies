"""Replay datasets, game splitting, and observation encoding."""
import json
import random
import re

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


SYMBOLS = "PNBRQKpnbrqk?-"
TOKEN = {symbol: index for index, symbol in enumerate(SYMBOLS)}
MOVE = re.compile(r"[a-h][1-8][a-h][1-8][qrbn]?")


def replay_examples(replay):
    """Use the BEFORE-move view; frame t+1 stores the move from frame t.

    Never read the full-board field. Both colors retain the starter's absolute
    coordinates and piece colors. Promotions share a from/to class.
    """
    result = replay.get("result")
    if result not in ("1-0", "0-1", "1/2-1/2"):
        return
    frames = replay["frames"]
    for before, after in zip(frames, frames[1:]):
        ply = before["ply"]
        if after["ply"] != ply + 1:
            raise ValueError("Replay frames must have consecutive ply numbers")
        white = ply % 2 == 0
        observation = before["white" if white else "black"]
        if len(observation) != 64 or any(s not in TOKEN for s in observation):
            raise ValueError("Invalid observation")
        uci = after["move"]
        if not isinstance(uci, str) or MOVE.fullmatch(uci) is None:
            raise ValueError(f"Invalid recorded move: {uci!r}")
        origin = ord(uci[0]) - ord("a") + 8 * (int(uci[1]) - 1)
        destination = ord(uci[2]) - ord("a") + 8 * (int(uci[3]) - 1)
        own_pieces = "PNBRQK" if white else "pnbrqk"
        if observation[origin] not in own_pieces:
            raise ValueError("Recorded move does not start on a visible own piece")
        yield [TOKEN[s] for s in observation], origin * 64 + destination


def observation_planes(tokens, device):
    tokens = tokens.to(device=device, dtype=torch.long)
    # Preserve any leading batch/time dimensions; the last dimension is squares.
    return nn.functional.one_hot(tokens, len(SYMBOLS)).transpose(-1, -2).reshape(
        *tokens.shape[:-1], 14, 8, 8).float()


class ReplayDataset(Dataset):
    """Store 64 uint8 tokens per position; expand to planes only per batch."""

    def __init__(self, paths):
        observations, targets = [], []
        self.games = 0
        for path in tqdm(paths, desc="Read replays", unit="game"):
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
                examples = list(replay_examples(replay))
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Bad replay {path}: {error}") from error
            if examples:
                views, moves = zip(*examples)
                observations.append(np.asarray(views, dtype=np.uint8))
                targets.append(np.asarray(moves, dtype=np.int64))
                self.games += 1
        if not observations:
            raise ValueError("No training examples in this split")
        self.observations = np.concatenate(observations)
        self.targets = np.concatenate(targets)
        self.positions = len(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Both datasets expose [time, squares] inputs and [time] labels.
        return self.observations[index][None], self.targets[index:index + 1]


def split_games(paths, fraction, seed):
    paths = sorted(paths)
    random.Random(seed).shuffle(paths)
    if len(paths) < 2:
        raise ValueError("Need at least two replay files for a game-level split")
    count = max(1, min(len(paths) - 1, round(len(paths) * fraction)))
    return paths[count:], paths[:count]


class SequenceDataset(Dataset):
    """One ordered trajectory per player per game, including opponent turns."""

    def __init__(self, paths):
        self.sequences = []
        self.games = self.positions = 0
        for path in tqdm(paths, desc="Read sequences", unit="game"):
            try:
                replay = json.loads(path.read_text(encoding="utf-8"))
                examples = list(replay_examples(replay))
                if not examples:
                    continue
                frames = replay["frames"][:-1]
                for white, side in ((True, "white"), (False, "black")):
                    views = [[TOKEN[s] for s in frame[side]] for frame in frames]
                    if any(len(view) != 64 for view in views):
                        raise ValueError("Invalid observation length")
                    labels = [target if (frame["ply"] % 2 == 0) == white else -100
                              for frame, (_, target) in zip(frames, examples)]
                    self.sequences.append((np.asarray(views, dtype=np.uint8),
                                           np.asarray(labels, dtype=np.int64)))
                self.games += 1
                self.positions += len(examples)
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Bad replay {path}: {error}") from error
        if not self.sequences:
            raise ValueError("No training examples in this split")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def collate_sequences(sequences):
    views, labels = zip(*sequences)
    return (
        pad_sequence([torch.from_numpy(view) for view in views], batch_first=True,
                     padding_value=TOKEN["?"]),
        pad_sequence([torch.from_numpy(label) for label in labels], batch_first=True,
                     padding_value=-100),
    )
