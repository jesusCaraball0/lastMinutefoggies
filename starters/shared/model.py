"""Policies with a shared training interface and standalone game inference.

make_loader selects the data representation; forward_chunk returns [B,T,4096]
logits and optional memory; clip_gradients applies the policy's training rule.
"""
import torch
from torch import nn
try:
    from .observation_history import ObservationHistory
    from .boosted_features import BoostedHistory, boosted_planes, CHANNELS as BOOSTED_CHANNELS
except ImportError:  # Standalone arena bot imports model directly.
    from observation_history import ObservationHistory
    from boosted_features import BoostedHistory, boosted_planes, CHANNELS as BOOSTED_CHANNELS


class ResidualBlock(nn.Module):
    """Two spatial convolutions with an identity shortcut and post-add activation."""
    def __init__(self, channels, nonlinearity, batch_norm, affine):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(channels, affine=affine) if batch_norm else nn.Identity()
        self.relu1 = nonlinearity()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(channels, affine=affine) if batch_norm else nn.Identity()
        self.relu2 = nonlinearity()

    def forward(self, inputs):
        residual = self.relu1(self.norm1(self.conv1(inputs)))
        return self.relu2(inputs + self.norm2(self.conv2(residual)))


def make_encoder(activation="relu", negative_slope=0.01, channels=64, hidden_size=256, batch_norm=False,
                 batch_norm_affine=True, dense_batch_norm=False, dropout=0., residual_blocks=0, head_channels=None,
                 input_channels=14, match_current_initialization=True):
    """Keep layer indices/weight shapes stable for historical checkpoints."""
    def nonlinearity():
        if activation == "relu":
            return nn.ReLU()
        if activation == "leaky_relu":
            if negative_slope <= 0:
                raise ValueError("LeakyReLU slope must be positive")
            return nn.LeakyReLU(negative_slope)
        raise ValueError(f"Unknown activation: {activation}")
    layers = [nn.Conv2d(14, channels, 3, padding=1), nonlinearity(),
              nn.Conv2d(channels, channels, 3, padding=1), nonlinearity(),
              nn.Flatten(), nn.Linear(channels * 64, hidden_size), nonlinearity()]
    if input_channels != 14:
        # The old frame-stack pilot preserves its matched-control initialization.
        # Boosted inputs use ordinary Conv2d initialization over all input planes.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            expanded = nn.Conv2d(input_channels, channels, 3, padding=1)
            if match_current_initialization:
                expanded.weight.zero_()
                expanded.weight[:, :14].copy_(layers[0].weight)
                expanded.bias.copy_(layers[0].bias)
        layers[0] = expanded
    if not 0 <= dropout < 1:
        raise ValueError("Dropout probability must be in [0, 1)")
    if not isinstance(residual_blocks, int) or residual_blocks < 0:
        raise ValueError("Residual block count must be a nonnegative integer")
    if head_channels is not None and (not isinstance(head_channels, int) or not 1 <= head_channels <= channels):
        raise ValueError("Head channels must be between 1 and the trunk width")
    # Extra layers must not change the initial stem/dense/output weights or the
    # training RNG stream for matched architecture ablations.
    with torch.random.fork_rng(devices=[]):
        if residual_blocks:
            tower = nn.Sequential(*(ResidualBlock(channels, nonlinearity, batch_norm, batch_norm_affine)
                                    for _ in range(residual_blocks)))
            tower[0].conv1 = layers[2]
            layers[2], layers[3] = tower, nn.Identity()
        if head_channels is not None:
            reduction = nn.Conv2d(channels, head_channels, 1)
            layers[5] = nn.Linear(head_channels * 64, hidden_size)
    encoder = nn.Sequential()
    for index, layer in enumerate(layers):
        encoder.add_module(str(index), layer)
        if batch_norm and isinstance(layer, nn.Conv2d):
            # Preserve existing conv/ReLU names for checkpoint comparisons.
            encoder.add_module(f'norm{index}', nn.BatchNorm2d(channels, affine=batch_norm_affine))
        if head_channels is not None and index == 3:
            encoder.add_module('head_reduce', reduction)
            encoder.add_module('head_relu', nonlinearity())
        if dense_batch_norm and index == 5:
            encoder.add_module('norm5', nn.BatchNorm1d(hidden_size, affine=batch_norm_affine))
    if dropout:
        encoder.add_module("dropout", nn.Dropout(dropout))
    return encoder


class Policy(nn.Module):
    def __init__(self, activation="relu", negative_slope=0.01, channels=64, hidden_size=256, batch_norm=False,
                 batch_norm_affine=True, dense_batch_norm=False, dropout=0., residual_blocks=0, head_channels=None,
                 history_frames=1, boosted=False, outcome_supervision=False):
        super().__init__()
        if not isinstance(history_frames, int) or history_frames < 1:
            raise ValueError('History frames must be a positive integer')
        self.history_frames = history_frames
        if boosted and history_frames != 1:
            raise ValueError('Boosted turn histories and raw frame stacking are separate input formats')
        if outcome_supervision and not boosted:
            raise ValueError("Outcome-supervised variant requires boosted inputs")
        self.outcome_supervision = outcome_supervision
        self.boosted = boosted
        self.input_channels = BOOSTED_CHANNELS if boosted else 14*history_frames
        self.encoder = make_encoder(activation, negative_slope, channels, hidden_size, batch_norm,
                                    batch_norm_affine, dense_batch_norm, dropout, residual_blocks, head_channels,
                                    input_channels=self.input_channels, match_current_initialization=not boosted)
        self.policy = nn.Linear(hidden_size, 4096)
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, observation):
        features = self.encoder(observation)
        return self.policy(features), self.value(features).tanh(), features

    def make_loader(self, paths, *, batch_size, sequence_batch_size, shuffle=False, seed=0):
        from training.datasets import ReplayDataset
        if self.boosted:
            from training.boosted_dataset import BoostedReplayDataset
            if self.outcome_supervision:
                from training.outcomes import BoostedOutcomeDataset
                return _make_loader(BoostedOutcomeDataset(paths), batch_size, None, shuffle, seed)
            return _make_loader(BoostedReplayDataset(paths), batch_size, None, shuffle, seed)
        return _make_loader(ReplayDataset(paths, self.history_frames), batch_size, None, shuffle, seed)

    def forward_chunk(self, observations, hidden=None):
        batch, steps = observations.shape[:2]
        logits, _, _ = self(observations.reshape(-1, self.input_channels, 8, 8))
        return logits.reshape(batch, steps, 4096), None

    def forward_chunk_with_value(self, observations, hidden=None):
        batch, steps = observations.shape[:2]
        logits, value, _ = self(observations.reshape(-1, self.input_channels, 8, 8))
        return logits.reshape(batch, steps, 4096), value.reshape(batch, steps), None

    def clip_gradients(self):
        pass


class RecurrentPolicy(nn.Module):
    def __init__(self, activation="relu", negative_slope=0.01, channels=64, hidden_size=256, batch_norm=False,
                 batch_norm_affine=True, dense_batch_norm=False, dropout=0., residual_blocks=0, head_channels=None):
        super().__init__()
        self.encoder = make_encoder(activation, negative_slope, channels, hidden_size, batch_norm,
                                    batch_norm_affine, dense_batch_norm, dropout, residual_blocks, head_channels)
        self.memory = nn.GRUCell(hidden_size, hidden_size)
        self.policy = nn.Linear(hidden_size, 4096)

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
        features = features.reshape(batch, steps, self.memory.hidden_size)
        if hidden is None:
            hidden = features.new_zeros(batch, self.memory.hidden_size)
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
    if not observation or len(observation) % 64:
        raise ValueError('Expected complete observation frames')
    planes = torch.zeros(14*(len(observation)//64), 8, 8)
    symbols = 'PNBRQKpnbrqk?-'
    for square, symbol in enumerate(observation):
        planes[14*(square//64)+symbols.index(symbol), (square % 64) // 8, square % 8] = 1
    return planes


def create_policy(kind, activation="relu", negative_slope=0.01, channels=64, hidden_size=256, batch_norm=False,
                 batch_norm_affine=True, dense_batch_norm=False, dropout=0., residual_blocks=0, head_channels=None,
                 history_frames=1, boosted=False, outcome_supervision=False):
    if channels < 1 or hidden_size < 1:
        raise ValueError("Channel count and hidden size must be positive")
    if kind == "feedforward":
        return Policy(activation, negative_slope, channels, hidden_size, batch_norm,
                      batch_norm_affine, dense_batch_norm, dropout, residual_blocks, head_channels, history_frames, boosted, outcome_supervision)
    if kind == "recurrent":
        if outcome_supervision:
            raise ValueError("Outcome supervision is supported for boosted feedforward policies only")
        if boosted:
            raise ValueError('Boosted inputs are supported for feedforward CNNs only')
        if history_frames != 1:
            raise ValueError('Stacked history is supported for feedforward CNNs only')
        return RecurrentPolicy(activation, negative_slope, channels, hidden_size, batch_norm,
                               batch_norm_affine, dense_batch_norm, dropout, residual_blocks, head_channels)
    raise ValueError(f"Unknown policy type: {kind}")
