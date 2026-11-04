#!/usr/bin/env python3
"""Compare ReLU CNN sizes on identical data and optimizer updates."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import math
from itertools import islice
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from starters.shared.model import create_policy
from training import train
from training.datasets import ReplayDataset, split_games
from training.diagnostics import final_histograms, plot_diagnostics, write_json
from training.probes import PolicyProbes


class BatchStream:
    """Keep shuffle position across checkpoints; hash the actual training inputs."""
    def __init__(self, dataset, batch_size, seed):
        if len(dataset) < batch_size:
            raise ValueError('Training split must contain at least one full batch')
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 drop_last=True, generator=torch.Generator().manual_seed(seed))
        self.iterator = iter(self.loader)
        self.digest = hashlib.sha256()
        self.positions = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        for tensor in batch:
            self.digest.update(tensor.numpy().tobytes())
        self.positions += batch[1].numel()
        return batch


def initialize_convolutions(model, initialization, seed):
    """Change only Conv2d weights; preserve biases, other layers, and RNG state."""
    if initialization == 'default':
        return
    if initialization != 'he_normal':
        raise ValueError(f'Unknown convolution initialization: {initialization}')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')


def warmup_cosine_lr(update, warmup_steps, decay_steps, start_lr, peak_lr, end_lr):
    """LR for a zero-based optimizer update, with exact phase endpoints."""
    if update < warmup_steps:
        return start_lr + (peak_lr - start_lr) * update / max(1, warmup_steps - 1)
    progress = min(1.0, (update - warmup_steps + 1) / decay_steps)
    return end_lr + (peak_lr - end_lr) * (1 + math.cos(math.pi * progress)) / 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-games', type=int, default=512)
    parser.add_argument('--val-games', type=int, default=64)
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[200, 1000, 2000])
    parser.add_argument('--sizes', nargs='+', choices=['small', 'medium', 'original'], default=['small', 'original'])
    parser.add_argument('--batch-norm', action='store_true',
                        help='Add BatchNorm2d between each convolution and ReLU')
    parser.add_argument('--conv-init', choices=['default', 'he_normal'], default='default')
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--warmup-updates', type=int, default=0,
                        help='Use an update-based schedule instead of epoch-based scheduling')
    parser.add_argument('--decay-updates', type=int, default=0)
    parser.add_argument('--decay-epochs', type=int, default=5)
    parser.add_argument('--peak-lr', type=float, default=1e-3)
    parser.add_argument('--end-lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=7980)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--optimizer', choices=['adam', 'adamw'], default='adam')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Defaults to 0 for Adam and 0.01 for AdamW')
    parser.add_argument('--beta-1', type=float, default=.9)
    parser.add_argument('--beta-2', type=float, default=.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--diagnostic-positions', type=int, default=128)
    parser.add_argument('--activation-observations', type=int, default=512)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'game_data')
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'out/size-experiments')
    args = parser.parse_args(argv)
    if min(args.train_games, args.val_games, args.batch_size, args.threads,
           args.diagnostic_positions, args.activation_observations, *args.checkpoints) < 1:
        parser.error('Counts must be positive')
    if sorted(set(args.checkpoints)) != args.checkpoints:
        parser.error('Checkpoints must be unique and increasing')
    if not 0 < args.learning_rate < float('inf'):
        parser.error('Learning rate must be positive and finite')
    if args.weight_decay is None:
        args.weight_decay = .01 if args.optimizer == 'adamw' else 0.0
    if not 0 <= args.weight_decay < float('inf'):
        parser.error('Weight decay must be nonnegative and finite')
    if not (0 <= args.beta_1 < 1 and 0 <= args.beta_2 < 1):
        parser.error('Betas must be finite and in [0, 1)')
    if not 0 < args.eps < float('inf'):
        parser.error('Epsilon must be positive and finite')
    if args.warmup_epochs < 0 or args.decay_epochs < 1:
        parser.error('Warmup epochs must be nonnegative; decay epochs must be positive')
    if not (0 < args.end_lr <= args.peak_lr < float('inf')) or (args.warmup_epochs and args.learning_rate > args.peak_lr):
        parser.error('Schedule requires 0 < end LR <= peak LR and start LR <= peak LR')
    if len(set(args.sizes)) != len(args.sizes):
        parser.error('Model sizes must be unique')
    if args.warmup_updates < 0 or args.decay_updates < 0:
        parser.error('Schedule update counts must be nonnegative')
    if bool(args.warmup_updates) != bool(args.decay_updates):
        parser.error('Specify both warmup and decay update counts')
    if args.warmup_updates and args.warmup_epochs:
        parser.error('Choose either epoch-based or update-based scheduling')
    if args.warmup_updates:
        if args.learning_rate > args.peak_lr:
            parser.error('Warmup start LR must not exceed peak LR')
        if args.checkpoints[-1] != args.warmup_updates + args.decay_updates:
            parser.error('Last checkpoint must equal warmup + decay updates')
    return args


def comparison_plots(cases, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name, case in cases.items():
        rows = case['checkpoints']
        steps = [row['updates'] for row in rows]
        axes[0].plot(steps, [row['validation']['loss'] for row in rows], marker='o', label=name)
        axes[1].plot(steps, [row['validation']['top1'] for row in rows], marker='o', label=name)
        for layer in ('encoder.1', 'encoder.3', 'encoder.6'):
            axes[2].plot(steps, [row['validation']['activation_stats'][layer]['dormant_fraction']
                                for row in rows], marker='o', label=f'{name}, {layer}')
    for ax, ylabel in zip(axes, ('Validation NLL', 'Validation top-1 accuracy', 'Dormant fraction')):
        ax.set(xlabel='Optimizer updates', ylabel=ylabel)
        ax.set_xticks(steps)
        ax.legend(fontsize=7)
    axes[2].set_ylim(-.02, 1.02)
    fig.tight_layout()
    fig.savefig(destination / 'comparison.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    train_paths, val_paths = split_games(args.data_dir.glob('*.json'), .1, args.seed)
    if len(train_paths) < args.train_games or len(val_paths) < args.val_games:
        raise ValueError('Not enough games in the fixed train/validation split')
    train_paths, val_paths = train_paths[:args.train_games], val_paths[:args.val_games]
    destination = args.out_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S-%fZ')
    destination.mkdir(parents=True)
    write_json(destination / 'data_split.json', {'train': list(map(str, train_paths)),
                                               'validation': list(map(str, val_paths))})
    report = {'status': 'running', 'settings': {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()}, 'cases': {},
              'design': 'One initialization seed; identical games and shuffled full batches; ReLU, selected optimizer, CPU; training metrics average over the interval since the previous checkpoint.'}
    write_json(destination / 'experiment.json', report)
    print(f'Experiment: {destination}', flush=True)
    started = time.perf_counter()
    try:
        dataset = ReplayDataset(train_paths)
        validation = ReplayDataset(val_paths)
        val_loader = DataLoader(validation, batch_size=args.batch_size)
        probes = PolicyProbes(val_paths, args.diagnostic_positions)
        device = torch.device('cpu')
        steps_per_epoch = len(dataset) // args.batch_size
        if steps_per_epoch < 1:
            raise ValueError('Training split must contain at least one full batch')
        checkpoints = args.checkpoints
        if args.warmup_epochs:
            checkpoints = [steps_per_epoch * epoch for epoch in range(1, args.warmup_epochs + args.decay_epochs + 1)]
        warmup_steps = args.warmup_updates or args.warmup_epochs * steps_per_epoch
        decay_steps = args.decay_updates or args.decay_epochs * steps_per_epoch
        report['schedule_updates'] = {'warmup': warmup_steps, 'decay': decay_steps} if warmup_steps else None
        report['resolved_checkpoints'] = checkpoints
        report['steps_per_epoch'] = steps_per_epoch
        for name, channels, hidden_size in (('small', 8, 32), ('medium', 32, 128), ('original', 64, 256)):
            if name not in args.sizes:
                continue
            case_dir = destination / name
            case_dir.mkdir()
            torch.manual_seed(args.seed)
            config = {'channels': channels, 'hidden_size': hidden_size, 'batch_norm': args.batch_norm}
            model = create_policy('feedforward', 'relu', **config).to(device)
            initialize_convolutions(model, args.conv_init, args.seed)
            optimizer_class = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[args.optimizer]
            optimizer = optimizer_class(model.parameters(), lr=args.learning_rate,
                                        weight_decay=args.weight_decay,
                                        betas=(args.beta_1, args.beta_2), eps=args.eps)
            optimizer_settings = {key: value for key, value in optimizer.param_groups[0].items()
                                  if key in ("lr", "betas", "eps", "weight_decay")}
            scheduler = None
            if warmup_steps:
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda update: warmup_cosine_lr(
                        update, warmup_steps, decay_steps, args.learning_rate,
                        args.peak_lr, args.end_lr) / args.learning_rate)
            stream = BatchStream(dataset, args.batch_size, args.seed)
            case = {'conv_initialization': args.conv_init, 'optimizer_settings': optimizer_settings, 'parameters': sum(p.numel() for p in model.parameters()),
                    'model_config': config, 'train_positions': len(dataset),
                    'validation_positions': len(validation), 'checkpoints': []}
            report['cases'][name] = case
            print(f'{name}: {case["parameters"]:,} parameters, {len(dataset):,} training positions', flush=True)
            current = 0
            best_loss = float('inf')
            with (case_dir / 'train.log').open('w', buffering=1) as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    initial = final_histograms(model, val_loader, device, bins=60,
                                              max_observations=args.activation_observations,
                                              max_values_per_layer=4096, collect_histograms=False,
                                              seed=args.seed)
                    write_json(case_dir / 'initial_activations.json', initial)
            for target in checkpoints:
                interval_learning_rates = [group["lr"] for group in optimizer.param_groups]
                interval_started = time.perf_counter()
                with (case_dir / 'train.log').open('a', buffering=1) as log:
                    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                        metrics = train.run_epoch(model, islice(stream, target-current), device,
                                                  f'{name} updates {current}-{target}', optimizer, scheduler=scheduler)
                        train_seconds = time.perf_counter() - interval_started
                        eval_started = time.perf_counter()
                        val = train.run_epoch(model, val_loader, device, f'{name} validation {target}')
                        validation_seconds = time.perf_counter() - eval_started
                        diagnostics_started = time.perf_counter()
                        histograms = final_histograms(
                            model, val_loader, device, bins=60,
                            max_observations=args.activation_observations,
                            max_values_per_layer=10000, collect_histograms=target == checkpoints[-1],
                            seed=args.seed)
                        val.update(activation_stats=histograms['activation_stats'],
                                   activation_observations=histograms['activation_observations'],
                                   policy_probes=probes.evaluate(model, device))
                assert metrics['update_to_weight']['optimizer_steps'] == target-current
                row = {'updates': target, 'positions_seen': stream.positions,
                       'batch_stream_sha256': stream.digest.hexdigest(),
                       'learning_rates': interval_learning_rates,
                       'next_learning_rates': [group['lr'] for group in optimizer.param_groups], 'train': metrics, 'validation': val,
                       'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                       'diagnostics_seconds': time.perf_counter()-diagnostics_started}
                row['data_passes'] = target / steps_per_epoch
                if args.warmup_epochs:
                    row['epoch'] = target // steps_per_epoch
                case['checkpoints'].append(row)
                checkpoint = {'kind': 'feedforward', 'activation': 'relu', 'negative_slope': .01,
                              'model_config': config, 'state_dict': model.state_dict(),
                              'updates': target, 'metrics': row, 'conv_initialization': args.conv_init,
                              'optimizer': args.optimizer, 'weight_decay': args.weight_decay,
                              'learning_rate': args.learning_rate, 'seed': args.seed,
                              'optimizer_settings': optimizer_settings,
                              'optimizer_state_dict': optimizer.state_dict(),
                              'scheduler_state_dict': scheduler.state_dict() if scheduler else None}
                train.save_checkpoint(checkpoint, case_dir / f'policy-{target:06d}.pt')
                if val['loss'] < best_loss:
                    best_loss = val['loss']
                    case.update(best_updates=target, best_validation_loss=best_loss)
                    train.save_checkpoint(checkpoint, case_dir / 'best_policy.pt')
                write_json(case_dir / 'metrics.json', case)
                write_json(destination / 'experiment.json', report)
                print(f'{name} @ {target}: train {metrics["loss"]:.4f}, val {val["loss"]:.4f}, '
                      f'top1 {val["top1"]:.1%}, training {train_seconds:.1f}s', flush=True)
                current = target
            train.save_checkpoint(checkpoint, case_dir / 'final_policy.pt')
            histograms.update(model_state='final_checkpoint', updates=current)
            write_json(case_dir / 'histograms.json', histograms)
            plot_diagnostics(case['checkpoints'], histograms, case_dir,
                             x_key='updates', x_label='Optimizer updates', state_label='Final checkpoint')
        cases = list(report['cases'].values())
        for case in cases[1:]:
            for reference, candidate in zip(cases[0]['checkpoints'], case['checkpoints']):
                if reference['batch_stream_sha256'] != candidate['batch_stream_sha256']:
                    raise AssertionError('Training inputs differ between model sizes')
        report['batch_streams_match'] = True if len(report['cases']) > 1 else None
        comparison_plots(report['cases'], destination)
        report.update(status='completed', wall_seconds=time.perf_counter()-started)
        write_json(destination / 'experiment.json', report)
    except BaseException as error:
        report.update(status='failed', error=str(error))
        write_json(destination / 'experiment.json', report)
        raise
    print(f'Saved: {destination}', flush=True)
    return destination


if __name__ == '__main__':
    main()
