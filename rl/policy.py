"""Checkpoint-compatible policies, observation masks and game-contract agents."""
from dataclasses import dataclass
from pathlib import Path

import chess
import torch
from torch.distributions import Categorical

from referee import BotFailure
from starters.shared.model import create_policy
from starters.shared.moves import ObservationTracker
from training.datasets import TOKEN, observation_planes


def load_policy(path, device='cpu', best=False):
    saved = torch.load(Path(path), map_location='cpu', weights_only=True)
    if best and saved.get('best_policy') is not None:
        saved=saved['best_policy']
    # Loading an opponent must not consume the learner's sampling/shuffle RNG.
    with torch.random.fork_rng(devices=[]):
        model = create_policy(saved['kind'], saved.get('activation', 'relu'),
                              saved.get('negative_slope', .01), **saved.get('model_config', {}))
    model.load_state_dict(saved['state_dict'])
    return model.to(device).eval(), saved


def masked_distribution(logits, masks, temperature):
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ValueError('Temperature must be positive and finite')
    if not masks.any(-1).all():
        raise ValueError('Action mask has an empty row')
    return Categorical(logits=(logits / temperature).masked_fill(~masks, -torch.inf))


def initialize_value(model, saved):
    if saved.get('algorithm') == 'ppo':
        return 'preserved trained PPO critic'
    torch.nn.init.zeros_(model.value.weight)
    torch.nn.init.zeros_(model.value.bias)
    return 'zero initialized; imitation did not supervise this head'


@dataclass
class Transition:
    tokens: torch.Tensor
    mask: torch.Tensor
    action: int
    log_probability: float
    value: float


class PolicyAgent:
    """A fast Python agent with the same information as the stdin/stdout bot.

    Learners are feedforward actor-critics. Frozen opponents may also be GRUs;
    their history updates once per observation, exactly as the existing bot.
    """
    def __init__(self, model, temperature=.9, record=False):
        self.model, self.temperature, self.record = model, temperature, record
        if record and hasattr(model, 'memory'):
            raise ValueError('Recurrent PPO requires sequence rollouts; recurrent opponents are supported')
        self.device = next(model.parameters()).device
        self.transitions = []

    def reset(self, side, base, increment):
        self.tracker = ObservationTracker(side)
        self.hidden = self.logits = None
        self.transitions = []
        # eval() keeps dropout off and BN running statistics fixed, even when
        # gradients are later enabled during PPO. It does not freeze weights.
        self.model.eval()

    def observe(self, view, ply):
        self.tracker.observe(view)
        if hasattr(self.model, 'memory'):
            with torch.no_grad():
                tokens = torch.tensor([[TOKEN[s] for s in view]], dtype=torch.uint8)
                self.logits, self.hidden = self.model(observation_planes(tokens, self.device), self.hidden)

    def act(self, remaining):
        moves = [move for move in self.tracker.legal_moves()
                 if move.promotion in (None, chess.QUEEN)]
        if not moves:
            raise BotFailure('No observation-based move candidates')
        tokens = torch.tensor([TOKEN[s] for s in self.tracker.current], dtype=torch.uint8)
        mask = torch.zeros(4096, dtype=torch.bool)
        by_action = {m.from_square*64+m.to_square: m for m in moves}
        mask[list(by_action)] = True
        with torch.no_grad():
            if hasattr(self.model, 'memory'):
                logits, value = self.logits, 0.
            else:
                logits, values, _ = self.model(observation_planes(tokens[None], self.device))
                value = float(values[0, 0])
            distribution = masked_distribution(logits, mask[None].to(self.device), self.temperature)
            action = distribution.sample()
            if self.record:
                self.transitions.append(Transition(tokens, mask, int(action),
                                                    float(distribution.log_prob(action)), value))
        move = by_action[int(action)]
        self.tracker.record_move(move)
        return move.uci()

    def close(self):
        pass
