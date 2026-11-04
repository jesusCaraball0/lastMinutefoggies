"""Numerical checks for optimization, legality, and recurrent diagnostics."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from referee import FowBoard, make_frame
from starters.shared.model import create_policy
from training.probes import PolicyProbes, UpdateRatios


class ConstantPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(4096))

    def forward_chunk(self, planes, hidden=None):
        return self.logits.expand(*planes.shape[:2], -1), None


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'game.json'

    def replay(self, board, moves):
        frames = [make_frame(board, [180, 180])]
        for move in moves:
            board.push_uci(move)
            frames.append(make_frame(board, [180, 180], move))
        self.path.write_text(json.dumps({'result': '0-1', 'frames': frames}))

    def test_actual_adam_update_ratio(self):
        model = nn.Sequential(nn.Linear(1, 1, bias=False))
        with torch.no_grad():
            model[0].weight.fill_(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=.1)
        tracker = UpdateRatios(model, every=1)
        model[0].weight.grad = torch.tensor([[4.]])
        tracker.prepare()
        optimizer.step()
        tracker.record()
        measured = tracker.summary()['layers']['0']
        self.assertAlmostEqual(measured['update_l2']['mean'], .1, places=6)
        self.assertAlmostEqual(measured['relative_l2']['mean'], .05, places=6)
        self.assertAlmostEqual(measured['parameter_l2']['mean'], 2)

    def test_uniform_legal_probability_and_masked_nll(self):
        self.replay(FowBoard(), ['e2e4'])
        probes = PolicyProbes([self.path], max_positions=1)
        report = probes.evaluate(ConstantPolicy(), 'cpu')
        legal = report['legal_moves']
        self.assertAlmostEqual(legal['legal_probability_mass'], 20 / 4096)
        self.assertAlmostEqual(legal['legal_nll'], math.log(20), places=6)
        self.assertAlmostEqual(report['raw_nll'], math.log(4096), places=6)
        self.assertEqual(legal['preferred_legal_move_removed_by_safety'], 0)
        self.assertIsNone(report['memory'])

    def test_safety_filter_does_not_define_legal_training_mask(self):
        self.replay(FowBoard('4r2k/8/8/8/8/8/4R3/4K3 w - - 0 1'), ['e2d2'])
        probes = PolicyProbes([self.path], max_positions=1)
        policy = ConstantPolicy()
        label = int(probes.trajectories[0]['targets'][0])
        with torch.no_grad():
            policy.logits[label] = 15
        report = probes.evaluate(policy, 'cpu')
        legal = report['legal_moves']
        self.assertEqual(legal['legal_top1'], 1)
        self.assertTrue(math.isfinite(legal['legal_nll']))
        self.assertEqual(legal['preferred_legal_move_removed_by_safety'], 1)
        self.assertGreater(legal['probability_mass_removed_by_safety'], .99)

    def test_memory_probe_and_temporal_gradients_preserve_model_state(self):
        self.replay(FowBoard(), ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'd2d3', 'd7d6'])
        policy = create_policy('recurrent', channels=2, hidden_size=8)
        policy.train()
        report = PolicyProbes([self.path], 8).evaluate(policy, 'cpu', sequence_length=3)
        memory = report['memory']
        self.assertAlmostEqual(memory['delta_nll_reset_minus_normal'], memory['reset_nll'] - memory['normal_nll'])
        self.assertEqual(set(report['gradient_by_lag']['lags']), {'0', '1', '2', '3'})
        self.assertGreater(report['gradient_by_lag']['lags']['3']['mean'], 0)
        self.assertTrue(all(p.grad is None for p in policy.parameters()))
        self.assertTrue(policy.training)


if __name__ == '__main__':
    unittest.main()
