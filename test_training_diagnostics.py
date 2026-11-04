"""Regression checks for epoch logging, gradient aggregation, and diagnostics."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from referee import FowBoard, make_frame
from training import train
from training.diagnostics import GradientNorms, final_histograms
from training.datasets import TOKEN
from starters.shared import model as shared_model, moves as shared_moves


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = nn.Linear(14 * 8 * 8, 6)
        self.value = nn.Linear(14 * 8 * 8, 1)

    def forward_chunk(self, planes, hidden=None):
        return self.policy(planes.flatten(2)), None


class TrainingDiagnosticsTests(unittest.TestCase):
    def test_gradients_combine_layer_parameters_and_capture_clipping(self):
        model = nn.ModuleDict({'used': nn.Linear(1, 1), 'unused': nn.Linear(1, 1)})
        tracker = GradientNorms(model)
        model['used'].weight.grad = torch.tensor([[3.]])
        model['used'].bias.grad = torch.tensor([4.])
        tracker.record('pre_clip')
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        tracker.record('post_clip')
        model['used'].weight.grad = torch.tensor([[0.]])
        model['used'].bias.grad = torch.tensor([2.])
        tracker.record('pre_clip')
        result = tracker.summary()
        self.assertAlmostEqual(result['used']['pre_clip']['mean'], 3.5)
        self.assertEqual(result['used']['pre_clip']['max'], 5.)
        self.assertEqual(result['used']['pre_clip']['steps'], 2)
        self.assertEqual(result['used']['pre_clip']['median'], 3.5)
        self.assertAlmostEqual(result['used']['pre_clip']['p95'], 4.85)
        clipping = tracker.global_summary()
        self.assertEqual(clipping['clipped_steps'], 1)
        self.assertEqual(clipping['clipped_fraction'], 1.)
        self.assertAlmostEqual(result['used']['post_clip']['mean'], 1., places=5)
        self.assertIsNone(result['unused']['pre_clip'])

    def test_activation_padding_excluded_and_hooks_removed(self):
        model = TinyPolicy()
        # One actual position and three all-unknown padding positions.
        tokens = torch.full((2, 2, 64), TOKEN['?'], dtype=torch.uint8)
        tokens[0, 0] = TOKEN['-']
        loader = [(tokens, torch.zeros((2, 2), dtype=torch.long))]
        data = final_histograms(model, loader, 'cpu', bins=5,
                                max_observations=10, max_values_per_layer=100)
        self.assertEqual(data['activation_observations'], 1)
        self.assertEqual(data['activations']['policy']['count'], 6)
        self.assertEqual(sum(data['activations']['policy']['counts']), 6)
        self.assertNotIn('value', data['activations'])  # Not called by this policy.
        self.assertIn('value', data['weights'])
        self.assertTrue(model.training)
        self.assertTrue(all(not m._forward_hooks for m in model.modules()))

    def test_activation_sample_cap(self):
        model = TinyPolicy()
        tokens = torch.full((2, 4, 64), TOKEN['-'], dtype=torch.uint8)
        data = final_histograms(model, [(tokens, None)], 'cpu', bins=5,
                                max_observations=3, max_values_per_layer=7)
        self.assertEqual(data['activation_observations'], 3)
        self.assertEqual(data['activations']['policy']['count'], 7)

    def test_activation_statistics_use_all_values_not_histogram_sample(self):
        model = TinyPolicy()
        with torch.no_grad():
            model.policy.weight.zero_()
            model.policy.bias.copy_(torch.tensor([-2., -1., 0., 0., 1., 2.]))
        tokens = torch.full((2, 1, 64), TOKEN['-'], dtype=torch.uint8)
        data = final_histograms(model, [(tokens, None)], 'cpu', max_values_per_layer=1)
        stats = data['activation_stats']['policy']
        self.assertEqual(data['activations']['policy']['count'], 1)
        self.assertEqual(stats['count'], 12)
        self.assertAlmostEqual(stats['zero_fraction'], 1 / 3)
        self.assertAlmostEqual(stats['negative_fraction'], 1 / 3)
        self.assertAlmostEqual(stats['all_zero_channel_fraction'], 1 / 3)
        self.assertAlmostEqual(stats['never_positive_channel_fraction'], 2 / 3)
        self.assertEqual(stats['per_channel'][0]['negative_fraction'], 1.)
        self.assertEqual(stats['per_channel'][2]['zero_fraction'], 1.)
        self.assertAlmostEqual(stats['dormant_fraction'], 1 / 3)
        self.assertEqual(stats['per_channel'][0]['observation_mean_std'], 0.)
        self.assertEqual(stats['per_channel'][0]['relative_activity'], 2.)

    def test_leaky_encoder_passes_gradient_through_negative_activations(self):
        for kind in ('feedforward', 'recurrent'):
            model = shared_model.create_policy(kind, 'leaky_relu', .01)
            with torch.no_grad():
                for layer in model.encoder:
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        layer.weight.fill_(.001)
                        layer.bias.fill_(-2)
            output = model.encoder(torch.zeros(1, 14, 8, 8))
            self.assertTrue((output < 0).all())
            output.sum().backward()
            self.assertGreater(model.encoder[0].bias.grad.abs().sum().item(), 0)

    def test_batchnorm_preserves_base_weights_and_loads_running_statistics(self):
        with patch.dict('sys.modules', {'model': shared_model, 'moves': shared_moves}):
            from starters.shared.policy_bot import load_policy
        config = {'channels': 2, 'hidden_size': 8}
        torch.manual_seed(7980)
        baseline = shared_model.create_policy('feedforward', **config)
        torch.manual_seed(7980)
        model = shared_model.create_policy('feedforward', batch_norm=True, **config)
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
        self.assertEqual(list(model.encoder._modules),
                         ['0', 'norm0', '1', '2', 'norm2', '3', '4', '5', '6'])
        model.train()
        inputs = torch.randn(4, 1, 14, 8, 8)
        model.forward_chunk(inputs)
        self.assertEqual(model.encoder.norm0.num_batches_tracked.item(), 1)
        running_mean = model.encoder.norm0.running_mean.clone()
        model.eval()
        with torch.inference_mode():
            expected = model.forward_chunk(inputs)[0]
        self.assertTrue(torch.equal(running_mean, model.encoder.norm0.running_mean))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'policy.pt'
            torch.save({'kind': 'feedforward', 'activation': 'relu',
                        'model_config': dict(config, batch_norm=True),
                        'state_dict': model.state_dict()}, path)
            restored = load_policy(path)
            with torch.inference_mode():
                self.assertTrue(torch.equal(expected, restored.forward_chunk(inputs)[0]))
            self.assertTrue(torch.equal(running_mean, restored.encoder.norm0.running_mean))

    def test_legacy_and_new_checkpoints_keep_their_activation_on_load(self):
        with patch.dict('sys.modules', {'model': shared_model, 'moves': shared_moves}):
            from starters.shared.policy_bot import load_policy
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.pt'
            for kind in ('feedforward', 'recurrent'):
                for activation in ('relu', 'leaky_relu'):
                    original = shared_model.create_policy(kind, activation)
                    checkpoint = {'kind': kind, 'state_dict': original.state_dict()}
                    if activation == 'leaky_relu':
                        checkpoint.update(activation=activation, negative_slope=.01)
                    torch.save(checkpoint, path)
                    restored = load_policy(path)
                    expected = nn.ReLU if activation == 'relu' else nn.LeakyReLU
                    self.assertIsInstance(restored.encoder[1], expected)
                    with torch.inference_mode():
                        inputs = torch.zeros(1, 1, 14, 8, 8)
                        self.assertTrue(torch.equal(original.forward_chunk(inputs)[0],
                                                    restored.forward_chunk(inputs)[0]))

    def test_checkpoint_score_uses_requested_accuracy(self):
        first = dict(loss=2., top1=.3, policy_probes=dict(legal_moves=dict(legal_top1=.4)))
        second = dict(loss=3., top1=.35, policy_probes=dict(legal_moves=dict(legal_top1=.5)))
        self.assertGreater(train.checkpoint_score(second, 'accuracy'),
                           train.checkpoint_score(first, 'accuracy'))
        self.assertGreater(train.checkpoint_score(second, 'legal_accuracy'),
                           train.checkpoint_score(first, 'legal_accuracy'))
        self.assertLess(train.checkpoint_score(second, 'loss'),
                        train.checkpoint_score(first, 'loss'))

    def test_epoch_resume_matches_uninterrupted_batchnorm_training(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            board = FowBoard()
            frames = [make_frame(board, [180, 180])]
            for uci in ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'd2d3', 'd7d6'):
                board.push_uci(uci)
                frames.append(make_frame(board, [180, 180], uci))
            for i in range(4):
                (data / f'{i}.json').write_text(json.dumps({'result': '1/2-1/2', 'frames': frames}))
            base = ['--data-dir', str(data), '--device', 'cpu', '--epochs', '2',
                    '--channels', '2', '--hidden-size', '8', '--batch-size', '4',
                    '--warmup-updates', '2', '--grad-clip-norm', '0.1',
                    '--residual-blocks', '3', '--dropout', '0.1',
                    '--selection-metric', 'legal_accuracy']
            continuous, interrupted = root / 'continuous', root / 'interrupted'
            with patch.multiple(train, OUTPUT=None, HISTOGRAM_BINS=8,
                                HISTOGRAM_MAX_OBSERVATIONS=8, HISTOGRAM_MAX_VALUES_PER_LAYER=64), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                train.main(base + ['--run-dir', str(continuous)])
                original_save = train.save_checkpoint

                def stop_after_epoch_one(checkpoint, path):
                    original_save(checkpoint, path)
                    if path.name == 'last_policy.pt' and checkpoint['epoch'] == 1:
                        raise InterruptedError('Simulated job interruption after a completed epoch')

                with patch.object(train, 'save_checkpoint', side_effect=stop_after_epoch_one):
                    with self.assertRaises(InterruptedError):
                        train.main(base + ['--run-dir', str(interrupted)])
                # CLI restores saved architecture, schedule, optimizer, and shuffle state.
                train.main(['--resume', str(interrupted / 'last_policy.pt')])
            uninterrupted = torch.load(continuous / 'final_policy.pt', weights_only=True, map_location='cpu')
            resumed = torch.load(interrupted / 'final_policy.pt', weights_only=True, map_location='cpu')
            for name, value in uninterrupted['state_dict'].items():
                self.assertTrue(torch.equal(value, resumed['state_dict'][name]), name)
            self.assertEqual(uninterrupted['scheduler_state_dict'], resumed['scheduler_state_dict'])
            self.assertEqual(uninterrupted['metrics']['train']['loss'], resumed['metrics']['train']['loss'])
            self.assertEqual(uninterrupted['metrics']['validation']['loss'], resumed['metrics']['validation']['loss'])
            self.assertEqual(uninterrupted['metrics']['train']['learning_rates_by_step'],
                             resumed['metrics']['train']['learning_rates_by_step'])
            self.assertEqual(json.loads((interrupted / 'metrics.json').read_text())['status'], 'completed')

    def test_both_trainers_produce_epoch_logs_checkpoints_and_plots(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / 'data'
            data.mkdir()
            board = FowBoard()
            frames = [make_frame(board, [180, 180])]
            for uci in ('e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4', 'f8c5', 'd2d3', 'd7d6'):
                board.push_uci(uci)
                frames.append(make_frame(board, [180, 180], uci))
            for i in range(4):
                (data / f'{i}.json').write_text(json.dumps({'result': '1/2-1/2', 'frames': frames}))
            for kind in ('feedforward', 'recurrent'):
                with self.subTest(kind=kind):
                    destination = root / kind / 'policy.pt'
                    with patch.multiple(train, MODEL_TYPE=kind, OUTPUT=destination,
                                        DATA_DIR=data, OUT_DIR=root / 'out', DEVICE='cpu',
                                        CHANNELS=2, HIDDEN_SIZE=8,
                                        BATCH_NORM=kind == "feedforward",
                                        SCHEDULE="warmup_cosine" if kind == "feedforward" else "none",
                                        WARMUP_UPDATES=2, GRAD_CLIP_NORM=0.1,
                                        EPOCHS=2, BATCH_SIZE=4, SEQUENCE_BATCH_SIZE=2,
                                        SEQUENCE_LENGTH=4, HISTOGRAM_BINS=8,
                                        HISTOGRAM_MAX_OBSERVATIONS=16,
                                        HISTOGRAM_MAX_VALUES_PER_LAYER=128), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        train.main()
                    run = next((root / 'out').glob(f'*-{kind}'))
                    log = json.loads((run / 'metrics.json').read_text())
                    self.assertEqual(log['status'], 'completed')
                    self.assertEqual(len(log['epochs']), 2)
                    for epoch in log['epochs']:
                        if kind == 'recurrent':
                            self.assertEqual(epoch['learning_rates'], [train.LEARNING_RATE])
                        self.assertLessEqual(epoch['train']['gradient_global_l2']['post_clip']['max'], .100001)
                        self.assertGreater(epoch['train']['loss'], 0)
                        self.assertGreater(epoch['validation']['loss'], 0)
                        grads = epoch['train']['gradient_l2']
                        self.assertGreater(grads['policy']['pre_clip']['mean'], 0)
                        self.assertIn('median', grads['policy']['pre_clip'])
                        self.assertIn('p95', grads['policy']['pre_clip'])
                        self.assertIn('clipped_fraction', epoch['train']['gradient_global_l2'])
                        self.assertIn('per_channel', epoch['validation']['activation_stats']['encoder.1'])
                        self.assertEqual(epoch['validation']['activation_stats']['encoder.1']['module_type'], 'ReLU')
                        if kind == 'feedforward':
                            self.assertGreater(epoch['train']['gradient_global_l2']['clipped_fraction'], 0.)
                        if kind == 'recurrent':
                            self.assertGreater(grads['memory']['pre_clip']['mean'], 0)
                        else:
                            self.assertIsNone(grads['value']['pre_clip'])
                    self.assertEqual((run / 'best_policy.pt').read_bytes(), destination.read_bytes())
                    best = torch.load(destination, weights_only=True, map_location='cpu')
                    self.assertEqual(best['epoch'], log['best_epoch'])
                    self.assertEqual(best['activation'], 'relu')
                    self.assertEqual(best['model_config'], {'channels': 2, 'hidden_size': 8, 'batch_norm': kind == 'feedforward'})
                    self.assertIn('optimizer_state_dict', best)
                    if kind == 'feedforward':
                        self.assertIsNotNone(best['scheduler_state_dict'])
                        trace = [r[0] for e in log['epochs'] for r in e['train']['learning_rates_by_step']]
                        self.assertAlmostEqual(trace[0], train.LEARNING_RATE)
                        self.assertAlmostEqual(max(trace), train.PEAK_LEARNING_RATE)
                        self.assertAlmostEqual(trace[-1], train.MIN_LEARNING_RATE)
                    with patch.dict('sys.modules', {'model': shared_model, 'moves': shared_moves}):
                        from starters.shared.policy_bot import load_policy
                    restored = load_policy(destination)
                    self.assertEqual(restored.encoder[0].out_channels, 2)
                    self.assertEqual(restored.policy.in_features, 8)
                    final = torch.load(run / 'final_policy.pt', weights_only=True, map_location='cpu')
                    self.assertEqual(final['epoch'], 2)
                    hist = json.loads((run / 'histograms.json').read_text())
                    self.assertEqual(hist['epoch'], 2)
                    self.assertEqual(hist['model_state'], 'final_epoch')
                    self.assertGreater(hist['activation_observations'], 0)
                    for key in ('weights', 'activations'):
                        self.assertIn('policy', hist[key])
                        for layer, stats in hist[key].items():
                            self.assertEqual(sum(stats['counts']), stats['count'])
                            image = run / key / f'{layer}.png'
                            self.assertGreater(image.stat().st_size, 1000)
                    for filename in ('loss.png', 'learning_rate.png', 'gradient_l2.png',
                                     'gradient_quantiles.png', 'gradient_global_l2.png',
                                     'dormant_features.png', 'update_to_weight.png', 'legal_policy.png'):
                        self.assertGreater((run / filename).stat().st_size, 1000)
                    for removed in ('activation_channels', 'activation_sparsity.png', 'memory_saturation.png'):
                        self.assertFalse((run / removed).exists())


if __name__ == '__main__':
    unittest.main()
