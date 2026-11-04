#!/usr/bin/env python3
"""Run a reproducible local pilot across model types, activations, and data sizes."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from training import train
from training.datasets import split_games
from training.artifacts import EXPERIMENTS, unused_run_path
from training.diagnostics import write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-games', type=int, nargs='+', default=[32, 128])
    parser.add_argument('--val-games', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--sequence-batch-size', type=int, default=8)
    parser.add_argument('--sequence-length', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=7980)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--models', nargs='+', choices=['feedforward', 'recurrent'],
                        default=['feedforward', 'recurrent'])
    parser.add_argument('--activations', nargs='+', choices=['relu', 'leaky_relu'],
                        default=['relu', 'leaky_relu'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'auto'], default='cpu')
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'game_data')
    parser.add_argument('--out-dir', type=Path, default=EXPERIMENTS)
    parser.add_argument('--diagnostic-positions', type=int, default=128)
    parser.add_argument('--activation-observations', type=int, default=64)
    args = parser.parse_args(argv)
    integers = [*args.train_games, args.val_games, args.epochs, args.channels, args.hidden_size,
                args.batch_size, args.sequence_batch_size, args.sequence_length, args.threads,
                args.diagnostic_positions, args.activation_observations]
    if min(integers) < 1 or not 0 < args.learning_rate < float('inf'):
        parser.error('Counts, sizes, and learning rate must be positive and finite')
    if len(set(args.train_games)) != len(args.train_games):
        parser.error('--train-games must not contain duplicates')
    return args


def comparison_plots(cases, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    groups = sorted({(case['model'], case['activation']) for case in cases})
    for model, activation in groups:
        rows = sorted([case for case in cases if (case['model'], case['activation']) == (model, activation)],
                      key=lambda row: row['train_games'])
        x = [row['train_games'] for row in rows]
        label = f'{model}, {activation}'
        for ax, key in zip(axes, ('best_validation_loss', 'train_seconds_per_epoch', 'diagnostics_seconds_per_epoch')):
            ax.plot(x, [row[key] for row in rows], marker='o', label=label)
    for ax, title, ylabel in zip(axes, ('Validation quality', 'Training cost', 'Diagnostics cost'),
                                ('Best raw validation NLL', 'Seconds / epoch', 'Seconds / epoch')):
        ax.set(xlabel='Training games', ylabel=ylabel, title=title)
        ax.set_xticks(sorted({case['train_games'] for case in cases}))
    axes[-1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(destination / 'comparison.png', dpi=150)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    available_train, available_val = split_games(args.data_dir.glob('*.json'), .1, args.seed)
    if max(args.train_games) > len(available_train) or args.val_games > len(available_val):
        raise ValueError(f'Request exceeds available split: {len(available_train)} train, {len(available_val)} validation games')
    torch.set_num_threads(args.threads)
    destination = unused_run_path(args.out_dir, f'activation-recurrence-pilot-c{args.channels}-h{args.hidden_size}-{args.epochs}epochs')
    destination.mkdir(parents=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report = {'status': 'running', 'settings': settings, 'cases': [],
              'interpretation': 'One seed; fixed held-out games; same initial weights within each model across activation/data-size cases. CPU pilot timings do not predict GPU throughput.'}
    manifest = destination / 'experiment.json'
    write_json(manifest, report)
    print(f'Experiment: {destination}', flush=True)
    started = time.perf_counter()
    try:
        for games in args.train_games:
            for model in args.models:
                for activation in args.activations:
                    name = f'{model}-{activation}-{games}games'
                    case_dir = destination / name
                    case_dir.mkdir()
                    print(f'Starting {name}', flush=True)
                    train.MODEL_TYPE, train.ACTIVATION = model, activation
                    train.CHANNELS, train.HIDDEN_SIZE = args.channels, args.hidden_size
                    train.TRAIN_GAME_LIMIT, train.VALIDATION_GAME_LIMIT = games, args.val_games
                    train.DATA_DIR, train.DEVICE, train.SEED = args.data_dir, args.device, args.seed
                    train.EPOCHS, train.LEARNING_RATE = args.epochs, args.learning_rate
                    train.BATCH_SIZE, train.SEQUENCE_BATCH_SIZE = args.batch_size, args.sequence_batch_size
                    train.SEQUENCE_LENGTH = args.sequence_length
                    train.HISTOGRAM_MAX_OBSERVATIONS = args.activation_observations
                    train.HISTOGRAM_MAX_VALUES_PER_LAYER = 4096
                    train.PROBE_MAX_POSITIONS = args.diagnostic_positions
                    train.UPDATE_SAMPLE_EVERY = 10
                    train.OUT_DIR, train.OUTPUT = case_dir, case_dir / 'policy.pt'
                    with (case_dir / 'train.log').open('w', buffering=1) as log:
                        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                            run = train.main()
                    metrics = json.loads((run / 'metrics.json').read_text())
                    last = metrics['epochs'][-1]
                    probe = last['validation']['policy_probes']
                    case = {'name': name, 'model': model, 'activation': activation,
                            'train_games': games, 'validation_games': args.val_games,
                            'parameters': metrics['config']['parameter_count'],
                            'run_dir': str(run.resolve()), 'best_epoch': metrics['best_epoch'],
                            'best_validation_loss': metrics['best_validation_loss'],
                            'final_train_loss': last['train']['loss'],
                            'final_validation_loss': last['validation']['loss'],
                            'final_validation_top1': last['validation']['top1'],
                            'legal_nll': probe['legal_moves']['legal_nll'],
                            'legal_top1': probe['legal_moves']['legal_top1'],
                            'legal_probability_mass': probe['legal_moves']['legal_probability_mass'],
                            'memory_delta_nll': probe['memory']['delta_nll_reset_minus_normal'] if probe['memory'] else None,
                            'clipped_fraction': last['train']['gradient_global_l2']['clipped_fraction'],
                            'dormant_encoder_1': last['validation']['activation_stats']['encoder.1']['dormant_fraction'],
                            'dormant_encoder_3': last['validation']['activation_stats']['encoder.3']['dormant_fraction'],
                            'train_seconds_per_epoch': sum(row['train_seconds'] for row in metrics['epochs']) / args.epochs,
                            'diagnostics_seconds_per_epoch': sum(row['diagnostics_seconds'] for row in metrics['epochs']) / args.epochs,
                            'optimizer_steps_per_epoch': last['train']['update_to_weight']['optimizer_steps'],
                            'train_positions': metrics['config']['train_positions'],
                            'wall_seconds': metrics['wall_seconds']}
                    report['cases'].append(case)
                    write_json(manifest, report)
                    print(f"Finished {name}: val loss {case['best_validation_loss']:.4f}, "
                          f"{case['parameters']:,} parameters, "
                          f"{case['train_seconds_per_epoch']:.2f}s training/epoch", flush=True)
        comparison_plots(report['cases'], destination)
        report.update(status='completed', wall_seconds=time.perf_counter() - started)
        write_json(manifest, report)
    except BaseException as error:
        report.update(status='failed', error=str(error))
        write_json(manifest, report)
        raise
    print(f'Saved comparison and per-case diagnostics: {destination}', flush=True)
    return destination


if __name__ == '__main__':
    main()
