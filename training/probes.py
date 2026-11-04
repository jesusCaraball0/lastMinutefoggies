"""Bounded policy diagnostics on replay observations; never use hidden boards."""
from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from starters.shared.moves import ObservationTracker, candidates, king_safe_moves
from training.datasets import TOKEN, observation_planes, replay_examples
from training.diagnostics import norm_statistics


class UpdateRatios:
    """Measure actual optimizer updates on sampled steps, not LR * gradients."""

    def __init__(self, model, every=100):
        if every < 1:
            raise ValueError('Update sampling interval must be positive')
        self.layers = {name: list(module.parameters(recurse=False))
                       for name, module in model.named_modules()
                       if any(p.requires_grad for p in module.parameters(recurse=False))}
        self.every, self.steps = every, 0
        self.before = {}
        self.samples = {name: [] for name in self.layers}

    def prepare(self):
        if self.steps % self.every == 0:
            self.before = {name: [p.detach().clone() for p in parameters]
                           for name, parameters in self.layers.items()
                           if any(p.grad is not None for p in parameters)}

    def record(self):
        for name, old in self.before.items():
            parameters = self.layers[name]
            delta = torch.stack([(p.detach() - before).float().square().sum()
                                 for p, before in zip(parameters, old)]).sum().sqrt()
            weight = torch.stack([before.float().square().sum() for before in old]).sum().sqrt()
            self.samples[name].append(torch.stack((delta / (weight + 1e-12), delta, weight)))
        self.before = {}
        self.steps += 1

    def summary(self):
        result = {}
        for name, samples in self.samples.items():
            if not samples:
                result[name] = None
                continue
            values = torch.stack(samples).cpu().numpy()
            result[name] = {key: norm_statistics(values[:, i], len(values)) for i, key in enumerate(
                ('relative_l2', 'update_l2', 'parameter_l2'))}
        return {'sample_every_steps': self.every, 'optimizer_steps': self.steps, 'layers': result}


class PolicyProbes:
    """Precompute legal/safety masks once, keeping full history for selected moves."""

    def __init__(self, paths, max_positions=256):
        if max_positions < 1:
            raise ValueError('Policy probe position limit must be positive')
        self.trajectories = []
        self.positions = 0
        self.games = []
        for path in paths:
            replay = json.loads(Path(path).read_text())
            examples = list(replay_examples(replay))
            if not examples:
                continue
            self.games.append(str(path))
            frames = replay['frames'][:-1]
            for side, color in ((True, 'white'), (False, 'black')):
                tracker = ObservationTracker(side)
                tokens, targets, masks = [], [], {}
                for i, frame in enumerate(frames):
                    view = frame[color]
                    tracker.observe(view)
                    tokens.append([TOKEN[c] for c in view])
                    own_turn = (frame['ply'] % 2 == 0) == side
                    targets.append(examples[i][1] if own_turn else -100)
                    if own_turn:
                        board, legal = candidates(view, side, tracker.rights, tracker.previous,
                                                  apply_king_safety=False)
                        safe = king_safe_moves(board, legal, view)
                        # Promotions share one training class; count it once.
                        index = lambda moves: sorted({m.from_square * 64 + m.to_square for m in moves})
                        legal_ids, safe_ids = index(legal), index(safe)
                        if examples[i][1] not in legal_ids:
                            raise ValueError(f'Probe target outside legal mask: {path}, ply {frame["ply"]}')
                        masks[i] = (legal_ids, safe_ids)
                        tracker.record_move(chess.Move.from_uci(replay['frames'][i + 1]['move']))
                        self.positions += 1
                    if self.positions >= max_positions:
                        break
                self.trajectories.append({'tokens': torch.tensor(tokens, dtype=torch.uint8),
                                          'targets': torch.tensor(targets), 'masks': masks})
                if self.positions >= max_positions:
                    break
            if self.positions >= max_positions:
                break
        if not self.positions:
            raise ValueError('No labeled policy probe observations')

    def evaluate(self, model, device, sequence_length=16, gradient_trajectories=2):
        was_training = model.training
        model.eval()
        totals = dict(legal_probability_mass=0., legal_nll=0., legal_top1=0.,
                      legal_entropy=0., preferred_legal_move_removed_by_safety=0.,
                      probability_mass_removed_by_safety=0., normal_nll=0., normal_top1=0.,
                      reset_nll=0., reset_top1=0.)
        recurrent = hasattr(model, 'memory')
        try:
            with torch.inference_mode():
                for trajectory in self.trajectories:
                    planes = observation_planes(trajectory['tokens'].unsqueeze(0), device)
                    logits, _ = model.forward_chunk(planes)
                    logits = logits[0]
                    if recurrent:
                        # One independent batch item per frame, so each sees h=0.
                        reset, _ = model.forward_chunk(planes[0].unsqueeze(1))
                        reset = reset[:, 0]
                    for i, (legal, safe) in trajectory['masks'].items():
                        target = int(trajectory['targets'][i])
                        scores = logits[i]
                        log_probs = scores.log_softmax(0)
                        legal_scores = scores[legal]
                        legal_log_probs = legal_scores.log_softmax(0)
                        target_index = legal.index(target)
                        totals['legal_probability_mass'] += log_probs[legal].exp().sum().item()
                        totals['legal_nll'] -= legal_log_probs[target_index].item()
                        totals['legal_top1'] += int(legal[int(legal_scores.argmax())] == target)
                        totals['legal_entropy'] -= (legal_log_probs.exp() * legal_log_probs).sum().item()
                        totals['normal_nll'] -= log_probs[target].item()
                        totals['normal_top1'] += int(int(scores.argmax()) == target)
                        preferred = legal[int(legal_scores.argmax())]
                        totals['preferred_legal_move_removed_by_safety'] += int(preferred not in safe)
                        unsafe = [j for j, action in enumerate(legal) if action not in safe]
                        totals['probability_mass_removed_by_safety'] += legal_log_probs[unsafe].exp().sum().item()
                        if recurrent:
                            totals['reset_nll'] -= reset[i].log_softmax(0)[target].item()
                            totals['reset_top1'] += int(int(reset[i].argmax()) == target)
            averaged = {key: value / self.positions for key, value in totals.items()}
            result = {'positions': self.positions, 'temperature': 1.,
                      'raw_nll': averaged['normal_nll'], 'raw_top1': averaged['normal_top1'],
                      'legal_moves': {key: averaged[key] for key in (
                          'legal_probability_mass', 'legal_nll', 'legal_top1', 'legal_entropy',
                          'preferred_legal_move_removed_by_safety', 'probability_mass_removed_by_safety')}}
            result['memory'] = ({key: averaged[key] for key in (
                'normal_nll', 'normal_top1', 'reset_nll', 'reset_top1')} if recurrent else None)
            if recurrent:
                result['memory']['delta_nll_reset_minus_normal'] = averaged['reset_nll'] - averaged['normal_nll']
                result['gradient_by_lag'] = self.gradient_by_lag(
                    model, device, sequence_length, gradient_trajectories)
            else:
                result['gradient_by_lag'] = None
            return result
        finally:
            model.train(was_training)

    def gradient_by_lag(self, model, device, window, max_trajectories):
        """d(loss at t)/d(hidden at t-k), without accumulating parameter grads."""
        measurements = {}
        for trajectory in self.trajectories[:max_trajectories]:
            # A supervised endpoint after some history, if the game is long enough.
            endpoints = [i for i in trajectory['masks'] if i < 2 * window]
            if not endpoints:
                continue
            end = endpoints[-1]
            start = max(0, end + 1 - window)
            planes = observation_planes(trajectory['tokens'][:end + 1].unsqueeze(0), device)
            with torch.no_grad():
                if start:
                    _, initial = model.forward_chunk(planes[:, :start])
                else:
                    initial = torch.zeros(1, model.memory.hidden_size, device=device)
            with torch.enable_grad():
                hidden = initial.detach().requires_grad_(True)
                states = [hidden]
                for step in range(start, end + 1):
                    logits, hidden = model.forward_chunk(planes[:, step:step + 1], hidden)
                    states.append(hidden)
                label = trajectory['targets'][end:end + 1].to(device)
                loss = F.cross_entropy(logits[:, 0], label)
                gradients = torch.autograd.grad(loss, states)
            for lag, gradient in enumerate(reversed(gradients)):
                measurements.setdefault(lag, []).append(gradient.detach().float().norm().item())
        return {'window': window, 'definition': 'L2 of d(loss_t)/d(hidden_{t-lag}); no gradient crosses the window boundary',
                'lags': {str(lag): norm_statistics(np.asarray(values), len(values))
                         for lag, values in sorted(measurements.items())}}
