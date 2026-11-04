#!/usr/bin/env python3
"""Long-update, small-network ablations for the cluster run's scale drift."""
from __future__ import annotations
import argparse
import contextlib
import copy
from itertools import islice
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from starters.shared.model import create_policy
from training import train
from training.datasets import ReplayDataset, split_games, observation_planes
from training.artifacts import EXPERIMENTS, unused_run_path
from training.diagnostics import final_histograms, plot_diagnostics, write_json
from training.optim import weight_decay_groups
from training.probes import PolicyProbes
from training.size_experiment import BatchStream, warmup_cosine_lr

STABILITY_CASES = ('baseline', 'no_decay_norm_bias', 'fixed_bn_affine', 'dense_batchnorm')
DROPOUT_CASES = ('baseline', 'dropout_01', 'dropout_02', 'dense_bn_dropout_01')
ARCHITECTURE_CASES = ('dropout_01', 'residual_3', 'reduced_head')
CASES = tuple(dict.fromkeys(STABILITY_CASES + DROPOUT_CASES + ARCHITECTURE_CASES))

def suite_cases(suite):
    return {'stability': STABILITY_CASES, 'dropout': DROPOUT_CASES, 'architecture': ARCHITECTURE_CASES}[suite]


class CachedDataset(Dataset):
    def __init__(self, directory, split):
        self.observations = np.load(directory / f'{split}-observations.npy', mmap_mode='r')
        self.targets = np.load(directory / f'{split}-targets.npy', mmap_mode='r')
        self.positions = len(self.targets)

    def __len__(self):
        return self.positions

    def __getitem__(self, index):
        return self.observations[index:index+1].copy(), self.targets[index:index+1].copy()


def args_from_cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-games', type=int, default=2560)
    parser.add_argument('--val-games', type=int, default=64)
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--checkpoints', type=int, nargs='+', default=[5000, 20000, 40000, 80000])
    parser.add_argument('--schedule-updates', type=int, default=334440)
    parser.add_argument('--warmup-updates', type=int, default=5820)
    parser.add_argument('--seed', type=int, default=7980)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--suite', choices=('stability', 'dropout', 'architecture'), default='stability')
    parser.add_argument('--full-validation-probes', action='store_true')
    parser.add_argument('--preserve-legacy-eval-rng', action='store_true',
                        help='Isolate RNG for evaluations added between the original four checkpoints')
    parser.add_argument('--reference-run', type=Path, help='Reuse the completed dropout-0.1 baseline from this run')
    parser.add_argument('--reuse-data', type=Path, help='Reuse the exact cached dataset and split from a prior experiment')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'game_data')
    parser.add_argument('--out-dir', type=Path, default=EXPERIMENTS)
    args = parser.parse_args(argv)
    if min(args.train_games, args.val_games, args.channels, args.hidden_size, args.batch_size,
           args.schedule_updates, args.warmup_updates, args.threads, args.parallel, *args.checkpoints) < 1:
        parser.error('Counts must be positive')
    if args.checkpoints != sorted(set(args.checkpoints)) or args.checkpoints[-1] > args.schedule_updates:
        parser.error('Checkpoints must increase and not exceed the LR horizon')
    if args.schedule_updates <= args.warmup_updates:
        parser.error('Schedule must include a decay phase')
    return args


def feature_health(model, activation_stats, probe_tokens, probe_targets):
    result = {}
    for key in ('encoder.3', 'encoder.6'):
        channels = activation_stats[key]['per_channel']
        activity = sorted([v['mean_absolute'] for v in channels], reverse=True)
        result[key] = {'top4_activity_share': sum(activity[:4]) / sum(activity) if sum(activity) else 1.,
                       'dormant_fraction': activation_stats[key]['dormant_fraction'],
                       'zero_fraction': activation_stats[key]['zero_fraction']}
    norm_path = ('encoder.norm2' if hasattr(model.encoder, 'norm2') else
                 f"encoder.2.{len(model.get_submodule('encoder.2'))-1}.norm2")
    norm = model.get_submodule(norm_path)
    result['norm2'] = {'module_path': norm_path, 'affine': norm.affine,
                       'gamma_median': norm.weight.detach().median().item() if norm.affine else 1.,
                       'gamma_min': norm.weight.detach().min().item() if norm.affine else 1.,
                       'beta_median': norm.bias.detach().median().item() if norm.affine else 0.,
                       'running_variance_min': norm.running_var.min().item()}
    # Independent copy: train-mode gradient probe cannot change the trained model's BN buffers.
    probe = copy.deepcopy(model).train()
    probe.zero_grad(set_to_none=True)
    scores, _ = probe.forward_chunk(observation_planes(probe_tokens, torch.device('cpu')))
    torch.nn.functional.cross_entropy(scores[:, 0], probe_targets).backward()
    squared = {key: float(parameter.grad.double().square().sum())
               for key, parameter in probe.named_parameters() if parameter.grad is not None}
    total = sum(squared.values())
    result['gradient_probe'] = {'global_l2': total ** .5,
                               'norm2_squared_share': sum(v for k, v in squared.items()
                                                          if k.startswith(norm_path + '.')) / total}
    return result


def checkpoint_improvements(report, validation, updates):
    """Keep distinct optima for masked policy NLL, masked accuracy, and raw accuracy."""
    legal = validation['policy_probes']['legal_moves']
    choices = [('best_legal_nll', legal['legal_nll'], True, 'best_legal_nll_policy.pt'),
               ('best_legal_accuracy', legal['legal_top1'], False, 'best_accuracy_policy.pt'),
               ('best_raw_accuracy', validation['top1'], False, 'best_raw_accuracy_policy.pt')]
    improved = []
    for key, value, minimize, filename in choices:
        if key not in report or (value < report[key] if minimize else value > report[key]):
            report[key], report[key + '_updates'] = value, updates
            improved.append(filename)
    return improved


def run_case(args):
    destination = args.run_dir
    output = destination / args.case
    output.mkdir()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dataset, validation = CachedDataset(destination, 'train'), CachedDataset(destination, 'validation')
    val_loader = DataLoader(validation, batch_size=args.batch_size)
    activation_indices = np.random.default_rng(args.seed).choice(len(validation), min(512, len(validation)), replace=False)
    activation_loader = DataLoader(Subset(validation, activation_indices.tolist()), batch_size=args.batch_size)
    probe_indices = activation_indices[:min(128, len(activation_indices))]
    probe_tokens = torch.from_numpy(validation.observations[probe_indices].copy())[:, None]
    probe_targets = torch.from_numpy(validation.targets[probe_indices].copy())
    manifest = json.loads((destination / 'data_split.json').read_text())
    probes = PolicyProbes([Path(p) for p in manifest['validation']],
                         len(validation) if args.full_validation_probes else 1024)
    model_config = dict(channels=args.channels, hidden_size=args.hidden_size, batch_norm=True,
                        batch_norm_affine=args.case != 'fixed_bn_affine',
                        dense_batch_norm=args.case in ('dense_batchnorm', 'dense_bn_dropout_01'),
                        dropout={'dropout_01': .1, 'dropout_02': .2, 'dense_bn_dropout_01': .1,
                                 'residual_3': .1, 'reduced_head': .1}.get(args.case, 0.))
    if args.case == 'residual_3': model_config['residual_blocks'] = 3
    if args.case == 'reduced_head': model_config['head_channels'] = max(1, args.channels // 4)
    model = create_policy('feedforward', 'relu', **model_config)
    groups, names = weight_decay_groups(model, .01, args.case == 'no_decay_norm_bias')
    optimizer = torch.optim.AdamW(groups, lr=1e-4, betas=(.9, .99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda update: warmup_cosine_lr(
        update, args.warmup_updates, args.schedule_updates-args.warmup_updates, 1e-4, 1e-3, 1e-5) / 1e-4)
    stream = BatchStream(dataset, args.batch_size, args.seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(model_config=model_config, parameter_groups=names, optimizer='AdamW',
                  betas=[.9, .99], epsilon=1e-8, weight_decay=.01, global_clip=5.,
                  activation_sample='Fixed random sample of up to 512 held-out moves',
                  interpretation='Long-update stress test: smaller network and repeated dataset; not a full-scale performance prediction')
    report = dict(status='running', config=config, parameter_count=sum(p.numel() for p in model.parameters()),
                  train_positions=len(dataset), validation_positions=len(validation), checkpoints=[])
    write_json(output / 'metrics.json', report)
    best, current = float('inf'), 0
    started = time.perf_counter()
    print(f'Starting {args.case}: {report["parameter_count"]:,} parameters', flush=True)
    for target in args.checkpoints:
        interval_started = time.perf_counter()
        lrs = [group['lr'] for group in optimizer.param_groups]
        with (output / 'train.log').open('a', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                metrics = train.run_epoch(model, islice(stream, target-current), torch.device('cpu'),
                                          f'{args.case} {current}-{target}', optimizer, scheduler=scheduler,
                                          max_grad_norm=5.)
                train_seconds = time.perf_counter()-interval_started
                isolate = args.preserve_legacy_eval_rng and target not in (5000, 20000, 40000, 80000)
                with torch.random.fork_rng(devices=[]) if isolate else contextlib.nullcontext():
                    val = train.run_epoch(model, val_loader, torch.device('cpu'), f'Validation {target}')
                    histograms = final_histograms(model, activation_loader, torch.device('cpu'), bins=60,
                                                  max_observations=512, max_values_per_layer=10000,
                                                  collect_histograms=target == args.checkpoints[-1], seed=args.seed)
                    val.update(activation_stats=histograms['activation_stats'],
                               activation_observations=histograms['activation_observations'],
                               policy_probes=probes.evaluate(model, torch.device('cpu')))
                    health = feature_health(model, val['activation_stats'], probe_tokens, probe_targets)
        row = dict(updates=target, positions_seen=stream.positions,
                   batch_stream_sha256=stream.digest.hexdigest(), learning_rates=lrs,
                   train=metrics, validation=val, health=health, train_seconds=train_seconds,
                   diagnostics_and_validation_seconds=time.perf_counter()-interval_started-train_seconds)
        report['checkpoints'].append(row)
        state = dict(kind='feedforward', activation='relu', model_config=model_config,
                     state_dict=model.state_dict(), torch_rng_state=torch.get_rng_state(),
                     optimizer_state_dict=optimizer.state_dict(),
                     scheduler_state_dict=scheduler.state_dict(), config=config, updates=target, metrics=row)
        train.save_checkpoint(state, output / f'policy-{target:06d}.pt')
        if val['loss'] < best:
            best = val['loss']
            report.update(best_updates=target, best_validation_loss=best)
            train.save_checkpoint(state, output / 'best_policy.pt')
        if args.full_validation_probes:
            for filename in checkpoint_improvements(report, val, target):
                train.save_checkpoint(state, output / filename)
        write_json(output / 'metrics.json', report)
        print(f'{args.case} @ {target}: val={val["loss"]:.4f}, '
              f'clip={metrics["gradient_global_l2"]["clipped_fraction"]:.1%}, '
              f'grad median={metrics["gradient_global_l2"]["pre_clip"]["median"]:.2f}, '
              f'dense top4={health["encoder.6"]["top4_activity_share"]:.1%}, '
              f'gamma={health["norm2"]["gamma_median"]:.3f}', flush=True)
        current = target
    train.save_checkpoint(state, output / 'final_policy.pt')
    histograms.update(updates=current, model_state='final_checkpoint')
    write_json(output / 'histograms.json', histograms)
    full_probe = PolicyProbes([Path(p) for p in manifest['validation']], len(validation))
    report['full_validation_policy_probe'] = full_probe.evaluate(model, torch.device('cpu'))
    plot_diagnostics(report['checkpoints'], histograms, output, x_key='updates',
                     x_label='Optimizer updates', state_label='Final checkpoint')
    report.update(status='completed', wall_seconds=time.perf_counter()-started)
    write_json(output / 'metrics.json', report)
    print(f'Finished {args.case}: best val={best:.4f}', flush=True)


def compare(destination, cases=STABILITY_CASES):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports = {case: json.loads((destination / case / 'metrics.json').read_text()) for case in cases}
    assert all(r['status'] == 'completed' for r in reports.values())
    ref = reports[cases[0]]['checkpoints']
    for r in reports.values():
        assert len(ref) == len(r['checkpoints'])
        assert all(x['updates'] == y['updates'] and
                   x['batch_stream_sha256'] == y['batch_stream_sha256'] and
                   x['train']['learning_rates_by_step'] == y['train']['learning_rates_by_step']
                   for x, y in zip(ref, r['checkpoints']))
    summary = dict(status='completed', batch_streams_match=True, learning_rate_traces_match=True, cases={})
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for name, r in reports.items():
        rows = r['checkpoints']; x = [row['updates'] for row in rows]; last = rows[-1]
        summary['cases'][name] = dict(parameter_count=r['parameter_count'], best_loss=r['best_validation_loss'], best_updates=r['best_updates'],
                                     final_train_loss=last['train']['loss'], final_validation_loss=last['validation']['loss'],
                                     final_top1=last['validation']['top1'], health=last['health'],
                                     clip_fraction=last['train']['gradient_global_l2']['clipped_fraction'],
                                     gradient_l2=last['train']['gradient_global_l2']['pre_clip'],
                                     full_validation_policy_probe=r['full_validation_policy_probe'])
        values = [[row['validation']['loss'] for row in rows],
                  [row['train']['gradient_global_l2']['clipped_fraction'] for row in rows],
                  [row['train']['gradient_global_l2']['pre_clip']['median'] for row in rows],
                  [row['health']['encoder.6']['top4_activity_share'] for row in rows],
                  [row['health']['norm2']['gamma_median'] for row in rows],
                  [row['health']['encoder.6']['dormant_fraction'] for row in rows]]
        if tuple(cases) == ARCHITECTURE_CASES:
            values = [[row['validation']['loss'] for row in rows],
                      [100*row['validation']['top1'] for row in rows],
                      [100*row['train']['gradient_global_l2']['clipped_fraction'] for row in rows],
                      [row['train']['gradient_global_l2']['pre_clip']['median'] for row in rows],
                      [100*row['health']['encoder.6']['top4_activity_share'] for row in rows],
                      [100*row['validation']['top5'] for row in rows]]
        for ax, y in zip(axes.flat, values): ax.plot(x, y, marker='o', label=name)
    titles = ['Validation loss', 'Fraction of updates clipped', 'Median pre-clip gradient L2',
              'Top 4 dense features: activity share', 'Final trunk BatchNorm median gamma', 'Dormant dense fraction']
    if tuple(cases) == ARCHITECTURE_CASES:
        titles = ['Validation loss', 'Validation top-1 accuracy (%)', 'Updates clipped (%)',
                  'Median pre-clip gradient L2', 'Top 4 dense features: activity share (%)',
                  'Validation top-5 accuracy (%)']
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel='Optimizer updates'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(f'Matched-data ablations: {next(iter(reports.values()))["config"]["channels"]} channels, one seed')
    fig.tight_layout(); fig.savefig(destination / 'comparison.png', dpi=150); plt.close(fig)
    write_json(destination / 'summary.json', summary)
    return summary


def main(argv=None):
    args = args_from_cli(argv)
    if args.case:
        if not args.run_dir: raise ValueError('--case requires --run-dir')
        run_case(args)
        return args.run_dir
    torch.set_num_threads(args.threads)
    destination = args.run_dir or unused_run_path(args.out_dir, f'{args.suite}-ablations-c{args.channels}-h{args.hidden_size}-{args.train_games}games-{args.checkpoints[-1]}updates')
    destination.mkdir(parents=True)
    if args.reuse_data:
        import os
        manifest = json.loads((args.reuse_data / 'data_split.json').read_text())
        if len(manifest['train']) != args.train_games or len(manifest['validation']) != args.val_games:
            raise ValueError('Cached split game counts do not match requested counts')
        write_json(destination / 'data_split.json', manifest)
        for split in ('train', 'validation'):
            for kind in ('observations', 'targets'):
                filename = f'{split}-{kind}.npy'
                os.link(args.reuse_data / filename, destination / filename)
    else:
        train_paths, val_paths = split_games(args.data_dir.glob('*.json'), .1, args.seed)
        if len(train_paths) < args.train_games or len(val_paths) < args.val_games:
            raise ValueError('Not enough games in the fixed split')
        paths = dict(train=train_paths[:args.train_games], validation=val_paths[:args.val_games])
        write_json(destination / 'data_split.json', {k:list(map(str,v)) for k,v in paths.items()})
        for split, files in paths.items():
            data = ReplayDataset(files)
            np.save(destination / f'{split}-observations.npy', data.observations)
            np.save(destination / f'{split}-targets.npy', data.targets)
    print(f'Experiment: {destination}', flush=True)
    cases = suite_cases(args.suite)
    references = {}
    if args.reference_run:
        import os
        import shutil
        case = 'dropout_01'
        if case not in cases: raise ValueError('Reference reuse requires a dropout-0.1 baseline')
        source = args.reference_run / case
        report = json.loads((source / 'metrics.json').read_text())
        if report['status'] != 'completed': raise ValueError('Reference must be complete')
        for key in ('train_games', 'val_games', 'channels', 'hidden_size', 'batch_size',
                    'checkpoints', 'schedule_updates', 'warmup_updates', 'seed', 'threads'):
            if report['config'][key] != getattr(args, key):
                raise ValueError(f'Reference differs at {key}')
        if json.loads((args.reference_run / 'data_split.json').read_text()) != json.loads((destination / 'data_split.json').read_text()):
            raise ValueError('Reference data split differs')
        shutil.copytree(source, destination / case, copy_function=shutil.copy2)
        references[case] = str(source.resolve())
    started = time.perf_counter(); running = []; pending = [case for case in cases if case not in references]
    experiment = dict(status='running', run_dir=str(destination), cases=list(cases), reused_references=references)
    write_json(destination / 'experiment.json', experiment)
    try:
        while running or pending:
            while pending and len(running) < args.parallel:
                case = pending.pop(0)
                command = [sys.executable, str(Path(__file__).resolve()), '--case', case,
                           '--run-dir', str(destination), '--suite', args.suite, '--train-games', str(args.train_games),
                           '--val-games', str(args.val_games), '--channels', str(args.channels),
                           '--hidden-size', str(args.hidden_size), '--batch-size', str(args.batch_size),
                           '--threads', str(args.threads), '--seed', str(args.seed),
                           '--warmup-updates', str(args.warmup_updates), '--schedule-updates', str(args.schedule_updates),
                           '--checkpoints', *map(str,args.checkpoints)]
                if args.full_validation_probes: command.append('--full-validation-probes')
                if args.preserve_legacy_eval_rng: command.append('--preserve-legacy-eval-rng')
                running.append((case, subprocess.Popen(command)))
            time.sleep(1)
            for case, process in list(running):
                result = process.poll()
                if result is not None:
                    running.remove((case,process))
                    if result: raise RuntimeError(f'{case} exited with status {result}')
        compare(destination, cases)
        experiment.update(status='completed', wall_seconds=time.perf_counter()-started)
        write_json(destination / 'experiment.json', experiment)
    except BaseException as error:
        for _, process in running: process.terminate()
        experiment.update(status='failed', error=str(error)); write_json(destination / 'experiment.json', experiment)
        raise
    print(f'All comparisons saved: {destination}', flush=True)
    return destination


if __name__ == '__main__': main()
