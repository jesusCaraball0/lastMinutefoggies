"""Policies with a shared training interface and standalone game inference.

make_loader selects the data representation; forward_chunk returns [B,T,4096]
logits and optional memory; clip_gradients applies the policy's training rule.
"""
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(14, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(4096, 256), nn.ReLU())
        self.policy = nn.Linear(256, 4096)
        self.value = nn.Linear(256, 1)

    def forward(self, observation):
        features = self.encoder(observation)
        return self.policy(features), self.value(features).tanh(), features

    def make_loader(self, paths, *, batch_size, sequence_batch_size, shuffle=False, seed=0):
        from training.datasets import ReplayDataset
        return _make_loader(ReplayDataset(paths), batch_size, None, shuffle, seed)

    def forward_chunk(self, observations, hidden=None):
        batch, steps = observations.shape[:2]
        logits, _, _ = self(observations.reshape(-1, 14, 8, 8))
        return logits.reshape(batch, steps, 4096), None

    def clip_gradients(self):
        pass


class RecurrentPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(14, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(4096, 256), nn.ReLU(),
        )
        self.memory = nn.GRUCell(256, 256)
        self.policy = nn.Linear(256, 4096)

    def forward(self, observations, hidden=None):
        """Accept one frame [B,14,8,8] or a sequence [B,T,14,8,8].

        Hidden is [B,256], initialized to zero when omitted. Return move logits
        for each frame and the final memory. Callers detach memory between
        training chunks and reset it at the start of each game.
        """
        single = observations.ndim == 4
        if single:
            observations = observations.unsqueeze(1)
        batch, steps = observations.shape[:2]
        features = self.encoder(observations.reshape(-1, 14, 8, 8))
        features = features.reshape(batch, steps, 256)
        if hidden is None:
            hidden = features.new_zeros(batch, 256)
        states = []
        for step in range(steps):
            hidden = self.memory(features[:, step], hidden)
            states.append(hidden)
        logits = self.policy(torch.stack(states, dim=1))
        return (logits[:, 0] if single else logits), hidden

    def make_loader(self, paths, *, batch_size, sequence_batch_size, shuffle=False, seed=0):
        from training.datasets import SequenceDataset, collate_sequences
        return _make_loader(SequenceDataset(paths), sequence_batch_size,
                            collate_sequences, shuffle, seed)

    def forward_chunk(self, observations, hidden=None):
        return self(observations, hidden)

    def clip_gradients(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)


def _make_loader(dataset, batch_size, collate, shuffle, seed):
    # Training dependencies are imported only when constructing training data;
    # the standalone arena deliverable does not need the training directory.
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


def encode(observation):
    planes = torch.zeros(14, 8, 8)
    symbols = 'PNBRQKpnbrqk?-'
    for square, symbol in enumerate(observation):
        planes[symbols.index(symbol), square // 8, square % 8] = 1
    return planes


def create_policy(kind):
    if kind == "feedforward":
        return Policy()
    if kind == "recurrent":
        return RecurrentPolicy()
    raise ValueError(f"Unknown policy type: {kind}")
