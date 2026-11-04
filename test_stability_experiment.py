"""Check ablation isolation, optimizer grouping, and inference compatibility."""
import tempfile
from pathlib import Path
import unittest

import torch
from starters.shared.model import create_policy
from training.optim import weight_decay_groups


class StabilityExperimentTests(unittest.TestCase):
    def test_checkpoint_selection_separates_masked_nll_and_accuracy(self):
        from training.stability_experiment import checkpoint_improvements
        report = {}
        first = dict(top1=.2, policy_probes={'legal_moves': dict(legal_nll=3., legal_top1=.3)})
        self.assertEqual(set(checkpoint_improvements(report, first, 5000)),
                         {'best_legal_nll_policy.pt', 'best_accuracy_policy.pt', 'best_raw_accuracy_policy.pt'})
        second = dict(top1=.25, policy_probes={'legal_moves': dict(legal_nll=3.1, legal_top1=.28)})
        self.assertEqual(checkpoint_improvements(report, second, 10000), ['best_raw_accuracy_policy.pt'])
        third = dict(top1=.24, policy_probes={'legal_moves': dict(legal_nll=2.8, legal_top1=.32)})
        self.assertEqual(set(checkpoint_improvements(report, third, 15000)),
                         {'best_legal_nll_policy.pt', 'best_accuracy_policy.pt'})
        self.assertEqual(report['best_raw_accuracy_updates'], 10000)
        self.assertEqual(report['best_legal_nll_updates'], 15000)

    def test_faster_decay_preserves_warmup_and_reaches_floor(self):
        from training.size_experiment import warmup_cosine_lr
        def lr(step, horizon):
            return warmup_cosine_lr(step, 5820, horizon-5820, 1e-4, 1e-3, 1e-5)
        self.assertTrue(all(lr(step,80000)==lr(step,334440) for step in range(5820)))
        self.assertEqual(lr(0,80000), 1e-4)
        self.assertEqual(lr(5819,80000), 1e-3)
        self.assertAlmostEqual(lr(79999,80000), 1e-5)
        self.assertLess(lr(39999,80000), lr(39999,334440))

    def test_no_decay_groups_exempt_norms_and_biases_without_dropping_parameters(self):
        torch.manual_seed(7980)
        model = create_policy('feedforward', channels=2, hidden_size=8,
                              batch_norm=True, dense_batch_norm=True)
        groups, names = weight_decay_groups(model, .1, exclude_norm_bias=True)
        grouped = [p for group in groups for p in group['params']]
        self.assertEqual(len(grouped), len({id(p) for p in grouped}))
        self.assertEqual({id(p) for p in grouped}, {id(p) for p in model.parameters()})
        self.assertIn('encoder.norm2.weight', names['no_decay'])
        self.assertIn('encoder.norm5.bias', names['no_decay'])
        self.assertIn('encoder.5.weight', names['decay'])
        original = {k:v.detach().clone() for k,v in model.named_parameters()}
        optimizer = torch.optim.AdamW(groups, lr=.1)
        for parameter in model.parameters(): parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            expected = original[name] if name in names['no_decay'] else original[name] * .99
            self.assertTrue(torch.allclose(parameter, expected), name)

    def test_dropout_preserves_initialization_and_is_disabled_for_inference(self):
        torch.manual_seed(7980)
        baseline = create_policy('feedforward', channels=2, hidden_size=8, batch_norm=True)
        for dense, probability in ((False, .1), (False, .2), (True, .1)):
            torch.manual_seed(7980)
            config = dict(channels=2, hidden_size=8, batch_norm=True,
                          dense_batch_norm=dense, dropout=probability)
            model = create_policy('feedforward', **config)
            for name, value in baseline.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            self.assertEqual(list(model.encoder._modules)[-1], 'dropout')
            inputs = torch.ones(1024, 8)
            model.train()
            dropped = model.encoder.dropout(inputs)
            self.assertGreater((dropped == 0).float().mean().item(), probability - .03)
            self.assertLess((dropped == 0).float().mean().item(), probability + .03)
            model.eval()
            self.assertTrue(torch.equal(model.encoder.dropout(inputs), inputs))
            restored = create_policy('feedforward', **config)
            restored.load_state_dict(model.state_dict())
            restored.eval()
            board = torch.randn(1, 1, 14, 8, 8)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model.forward_chunk(board)[0], restored.forward_chunk(board)[0]))

    def test_architecture_shapes_common_initialization_and_checkpoint_loading(self):
        torch.manual_seed(7980)
        baseline = create_policy('feedforward', channels=4, hidden_size=8, batch_norm=True, dropout=.1)
        for extra in ({'residual_blocks': 3}, {'head_channels': 1}):
            torch.manual_seed(7980)
            config = dict(channels=4, hidden_size=8, batch_norm=True, dropout=.1, **extra)
            model = create_policy('feedforward', **config)
            for name in ('encoder.0.weight', 'encoder.0.bias', 'policy.weight', 'policy.bias'):
                self.assertTrue(torch.equal(model.state_dict()[name], baseline.state_dict()[name]), name)
            if 'residual_blocks' in extra:
                self.assertTrue(torch.equal(model.get_submodule('encoder.5').weight, baseline.get_submodule('encoder.5').weight))
                self.assertTrue(torch.equal(model.get_submodule('encoder.2')[0].conv1.weight, baseline.get_submodule('encoder.2').weight))
                self.assertEqual(sum(isinstance(m, torch.nn.Conv2d) for m in model.modules()), 7)
            else:
                self.assertEqual(model.get_submodule('encoder.5').in_features, 64)
            model.train()
            logits, _ = model.forward_chunk(torch.randn(4, 1, 14, 8, 8))
            self.assertEqual(logits.shape, (4, 1, 4096))
            logits.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for name, p in model.named_parameters() if not name.startswith('value.')))
            model.eval()
            restored = create_policy('feedforward', **config)
            restored.load_state_dict(model.state_dict())
            restored.eval()
            board = torch.randn(1, 1, 14, 8, 8)
            with torch.inference_mode():
                self.assertTrue(torch.equal(model.forward_chunk(board)[0], restored.forward_chunk(board)[0]))

    def test_residual_shortcut_transmits_input_and_gradient(self):
        from starters.shared.model import ResidualBlock
        block = ResidualBlock(2, torch.nn.ReLU, True, True).eval()
        with torch.no_grad():
            block.conv2.weight.zero_()
            block.conv2.bias.zero_()
        inputs = torch.ones(2, 2, 8, 8, requires_grad=True)
        outputs = block(inputs)
        self.assertTrue(torch.equal(inputs, outputs))
        outputs.sum().backward()
        self.assertTrue(torch.equal(inputs.grad, torch.ones_like(inputs)))

    def test_ablation_architectures_preserve_common_weights_and_load_for_single_board(self):
        torch.manual_seed(7980)
        baseline = create_policy('feedforward', channels=2, hidden_size=8, batch_norm=True)
        for affine, dense in ((False,False), (True,True)):
            config = dict(channels=2, hidden_size=8, batch_norm=True,
                          batch_norm_affine=affine, dense_batch_norm=dense)
            torch.manual_seed(7980)
            model = create_policy('feedforward', **config)
            for name, value in baseline.state_dict().items():
                if name in model.state_dict():
                    self.assertTrue(torch.equal(value, model.state_dict()[name]), name)
            self.assertEqual(model.encoder.norm2.affine, affine)
            self.assertEqual('norm5' in model.encoder._modules, dense)
            model.train()
            model.forward_chunk(torch.randn(4,1,14,8,8))
            model.eval()
            board = torch.randn(1,1,14,8,8)
            with torch.inference_mode(): expected = model.forward_chunk(board)[0]
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'policy.pt'
                torch.save(dict(kind='feedforward', activation='relu', model_config=config,
                                state_dict=model.state_dict()), path)
                checkpoint = torch.load(path, map_location='cpu', weights_only=True)
                restored = create_policy(checkpoint['kind'], checkpoint['activation'], **checkpoint['model_config'])
                restored.load_state_dict(checkpoint['state_dict'])
                restored.eval()
                with torch.inference_mode(): actual = restored.forward_chunk(board)[0]
                self.assertEqual(actual.shape, (1,1,4096))
                self.assertTrue(torch.equal(expected, actual))


if __name__ == '__main__': unittest.main()
