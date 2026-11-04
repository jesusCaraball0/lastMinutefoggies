#!/usr/bin/env python3
"""Train a feedforward or recurrent FOW policy on replay observations."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import time

import torch
from torch import nn
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from starters.shared.model import create_policy
from training.artifacts import EXPERIMENTS, unused_run_path
from training.datasets import observation_planes, split_games
from training.diagnostics import GradientNorms, final_histograms, plot_diagnostics, write_json
from training.probes import PolicyProbes, UpdateRatios


DATA_DIR = ROOT / "game_data"
MODEL_TYPE = "feedforward"  # "feedforward" or "recurrent"
ACTIVATION = "relu"  # Historical checkpoints without this field use ReLU.
NEGATIVE_SLOPE = 0.01
CHANNELS = 64
HIDDEN_SIZE = 256
BATCH_NORM = True
GRAD_CLIP_NORM = 5.0
TRAIN_GAME_LIMIT = None
VALIDATION_GAME_LIMIT = None
UPDATE_SAMPLE_EVERY = 100
PROBE_MAX_POSITIONS = 8192
PROBE_GRADIENT_TRAJECTORIES = 2
OUTPUT = None  # Optional export; run checkpoints always stay under out/.
EPOCHS = 8
BATCH_SIZE = 128
SEQUENCE_BATCH_SIZE = 16  # Player-game sequences per recurrent batch.
SEQUENCE_LENGTH = 16  # Truncated backpropagation length, in observations.
LEARNING_RATE = 1e-4
PEAK_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-5
WARMUP_UPDATES = 5820
SCHEDULE = "warmup_cosine"
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.99)
EPSILON = 1e-8
VALIDATION_FRACTION = 0.1
SEED = 7980
DEVICE = "auto"  # "auto", "cpu", "mps", or "cuda"
OUT_DIR = EXPERIMENTS
HISTOGRAM_BINS = 60
HISTOGRAM_MAX_OBSERVATIONS = 512
HISTOGRAM_MAX_VALUES_PER_LAYER = 50000


def run_epoch(model, loader, device, description, optimizer=None, scheduler=None,
              max_grad_norm=None, sequence_length=None):
    training = optimizer is not None
    chunk_length = sequence_length or SEQUENCE_LENGTH
    model.train(training)
    gradients = GradientNorms(model) if training else None
    updates = UpdateRatios(model, UPDATE_SAMPLE_EVERY) if training else None
    learning_rates_by_step = []
    loss_sum = correct = top5 = count = 0
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc=description, unit="batch")
        for tokens, targets in progress:
            hidden = None  # New batch = new games; never share player memories.
            for start in range(0, tokens.shape[1], chunk_length):
                planes = observation_planes(tokens[:, start:start + chunk_length], device)
                labels = targets[:, start:start + chunk_length].to(device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits, hidden = model.forward_chunk(planes, hidden)
                valid = labels != -100  # Opponent turns and padding have no loss.
                scores, labels = logits[valid], labels[valid]
                if labels.numel():
                    loss = nn.functional.cross_entropy(scores, labels)
                    if training:
                        loss.backward()
                        gradients.record("pre_clip")
                        if max_grad_norm is None:
                            model.clip_gradients()
                        else:
                            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm,
                                                     error_if_nonfinite=True)
                        gradients.record("post_clip")
                        updates.prepare()
                        if scheduler is not None:
                            learning_rates_by_step.append([group["lr"] for group in optimizer.param_groups])
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        updates.record()
                    count += labels.numel()
                    loss_sum += loss.item() * labels.numel()
                    correct += (scores.argmax(1) == labels).sum().item()
                    top5 += (scores.topk(5, dim=1).indices == labels[:, None]).any(1).sum().item()
                    progress.set_postfix(loss=f"{loss_sum / count:.4f}", refresh=False)
                if hidden is not None:
                    hidden = hidden.detach()  # Keep memory, truncate gradients.
    if not count:
        raise ValueError("Epoch contained no labeled moves")
    metrics = {"loss": loss_sum / count, "top1": correct / count, "top5": top5 / count,
               "positions": count}
    if gradients is not None:
        metrics["gradient_l2"] = gradients.summary()
        metrics["gradient_global_l2"] = gradients.global_summary()
        metrics["update_to_weight"] = updates.summary()
    if scheduler is not None:
        metrics["learning_rates_by_step"] = learning_rates_by_step
    return metrics


def choose_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def warmup_cosine_factor(update, warmup_updates, total_updates, start_lr, peak_lr, min_lr):
    """LR for the next zero-based optimizer update, including phase endpoints."""
    if update < warmup_updates:
        rate = start_lr + (peak_lr - start_lr) * update / max(1, warmup_updates - 1)
    else:
        progress = min(1., (update - warmup_updates + 1) / (total_updates - warmup_updates))
        rate = min_lr + (peak_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2
    return rate / start_lr


def parse_args(argv=None):
    defaults = dict(model_type=MODEL_TYPE, activation=ACTIVATION, negative_slope=NEGATIVE_SLOPE,
                    channels=CHANNELS, hidden_size=HIDDEN_SIZE, batch_norm=BATCH_NORM,
                    grad_clip_norm=GRAD_CLIP_NORM, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    sequence_batch_size=SEQUENCE_BATCH_SIZE, sequence_length=SEQUENCE_LENGTH,
                    learning_rate=LEARNING_RATE, peak_learning_rate=PEAK_LEARNING_RATE,
                    min_learning_rate=MIN_LEARNING_RATE, warmup_updates=WARMUP_UPDATES,
                    schedule=SCHEDULE, weight_decay=WEIGHT_DECAY, beta_1=BETAS[0], beta_2=BETAS[1],
                    epsilon=EPSILON, validation_fraction=VALIDATION_FRACTION, seed=SEED,
                    device=DEVICE, data_dir=DATA_DIR, out_dir=OUT_DIR, output=OUTPUT,
                    train_game_limit=TRAIN_GAME_LIMIT, validation_game_limit=VALIDATION_GAME_LIMIT,
                    probe_max_positions=PROBE_MAX_POSITIONS, threads=None,
                    residual_blocks=0, dropout=0., selection_metric='loss')
    parser = argparse.ArgumentParser(description=__doc__, argument_default=argparse.SUPPRESS)
    for key in defaults:
        flag = '--' + key.replace('_', '-')
        if key == 'batch_norm':
            parser.add_argument(flag, action=argparse.BooleanOptionalAction)
        elif key in ('data_dir', 'out_dir', 'output'):
            parser.add_argument(flag, type=Path)
        elif key in ('model_type', 'activation', 'schedule', 'device', 'selection_metric'):
            choices = {'model_type': ['feedforward', 'recurrent'], 'activation': ['relu', 'leaky_relu'],
                       'schedule': ['warmup_cosine', 'none'], 'device': ['auto', 'cpu', 'cuda', 'mps'],
                       'selection_metric': ['loss', 'accuracy', 'legal_accuracy']}
            parser.add_argument(flag, choices=choices[key])
        elif key in ('negative_slope', 'grad_clip_norm', 'learning_rate', 'peak_learning_rate',
                     'min_learning_rate', 'weight_decay', 'beta_1', 'beta_2', 'epsilon', 'validation_fraction', 'dropout'):
            parser.add_argument(flag, type=float)
        else:
            parser.add_argument(flag, type=int)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--resume', type=Path, help='Resume last_policy.pt after its completed epoch')
    provided = vars(parser.parse_args([] if argv is None else argv))
    if provided.get('resume'):
        saved = torch.load(provided['resume'], map_location='cpu', weights_only=True)
        defaults.update(saved['config']['settings'])
        allowed = {'resume', 'device', 'threads', 'run_dir'}
        for key in provided.keys() - allowed:
            if provided[key] != defaults[key]:
                parser.error(f'Resume preserves the saved {key}; start a new run to change it')
    defaults.update(provided)
    defaults.setdefault('resume', None)
    defaults.setdefault('run_dir', None)
    for key in ('data_dir', 'out_dir', 'output', 'run_dir', 'resume'):
        if defaults[key] is not None:
            defaults[key] = Path(defaults[key])
    args = argparse.Namespace(**defaults)
    if min(args.epochs, args.channels, args.hidden_size, args.batch_size,
           args.sequence_batch_size, args.sequence_length) < 1:
        parser.error('Epoch, batch, and model sizes must be positive')
    for key in ('train_game_limit', 'validation_game_limit', 'threads'):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            parser.error(f'{key} must be positive')
    if args.probe_max_positions is not None and args.probe_max_positions < 0:
        parser.error('Probe positions must be nonnegative; zero means full validation')
    if not (0 < args.learning_rate < float('inf') and
            0 < args.min_learning_rate <= args.peak_learning_rate < float('inf')):
        parser.error('Learning rates must be positive and finite; min <= peak')
    if not (0 <= args.beta_1 < 1 and 0 <= args.beta_2 < 1 and
            0 <= args.weight_decay < float('inf') and 0 < args.epsilon < float('inf') and
            0 < args.grad_clip_norm < float('inf') and 0 < args.validation_fraction < 1):
        parser.error('Invalid AdamW, clipping, or split settings')
    if args.warmup_updates < 0 or (args.schedule == 'warmup_cosine' and
                                  args.learning_rate > args.peak_learning_rate):
        parser.error('Warmup requires a nonnegative count and start LR <= peak LR')
    if args.residual_blocks < 0 or not 0 <= args.dropout < 1:
        parser.error('Residual blocks must be nonnegative; dropout must be in [0, 1)')
    return args


def checkpoint_score(validation, metric):
    """Larger is better; legal accuracy uses the fixed held-out policy probe."""
    if metric == 'loss':
        return -validation['loss']
    if metric == 'accuracy':
        return validation['top1']
    return validation['policy_probes']['legal_moves']['legal_top1']


def main(argv=None):
    args = parse_args(argv)
    started = time.perf_counter()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.output and args.output.exists() and not args.resume:
        raise FileExistsError(f'{args.output} already exists; choose a new --output')
    import matplotlib
    matplotlib.use('Agg')
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use a GPU allocation and CUDA-enabled PyTorch')
    model_config = dict(channels=args.channels, hidden_size=args.hidden_size, batch_norm=args.batch_norm)
    if args.residual_blocks or args.dropout:
        model_config.update(residual_blocks=args.residual_blocks, dropout=args.dropout)
    model = create_policy(args.model_type, args.activation, args.negative_slope, **model_config).to(device)
    saved = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if saved:
        run_dir = args.resume.resolve().parent
        if args.run_dir and args.run_dir.resolve() != run_dir:
            raise ValueError('Resume must use the checkpoint directory')
        split = json.loads((run_dir / 'data_split.json').read_text())
        train_paths, validation_paths = ([Path(p) for p in split[key]] for key in ('train', 'validation'))
    else:
        run_dir = args.run_dir or unused_run_path(args.out_dir, f'train-c{args.channels}-h{args.hidden_size}-res{args.residual_blocks}-{args.epochs}epochs-{args.model_type}')
        if run_dir.exists():
            raise FileExistsError(f'{run_dir} already exists; use --resume or a fresh --run-dir')
        train_paths, validation_paths = split_games(args.data_dir.glob('*.json'), args.validation_fraction, args.seed)
        train_paths = train_paths[:args.train_game_limit]
        validation_paths = validation_paths[:args.validation_game_limit]
    loader_options = dict(batch_size=args.batch_size, sequence_batch_size=args.sequence_batch_size, seed=args.seed)
    train_loader = model.make_loader(train_paths, shuffle=True, **loader_options)
    validation_loader = model.make_loader(validation_paths, **loader_options)
    train_data, validation_data = train_loader.dataset, validation_loader.dataset
    probe_limit = args.probe_max_positions or validation_data.positions
    probes = PolicyProbes(validation_paths, probe_limit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 betas=(args.beta_1, args.beta_2), eps=args.epsilon,
                                 weight_decay=args.weight_decay)
    scheduler = None
    total_updates = args.epochs * len(train_loader) if args.model_type == 'feedforward' else None
    if args.schedule == 'warmup_cosine':
        if total_updates is None:
            raise ValueError('Use --schedule none for recurrent training with variable-length chunks')
        if args.warmup_updates >= total_updates:
            raise ValueError(f'Warmup ({args.warmup_updates}) must be shorter than total updates ({total_updates})')
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda update: warmup_cosine_factor(
                update, args.warmup_updates, total_updates, args.learning_rate,
                args.peak_learning_rate, args.min_learning_rate))
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
                if key not in ('resume', 'run_dir')}
    config = dict(settings=settings, model_type=args.model_type, activation=args.activation,
                  negative_slope=args.negative_slope, model_config=model_config,
                  parameter_count=sum(p.numel() for p in model.parameters()), optimizer='AdamW',
                  learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                  betas=list((args.beta_1, args.beta_2)), epsilon=args.epsilon,
                  gradient_clip_global_l2=args.grad_clip_norm,
                  schedule=args.schedule, warmup_updates=args.warmup_updates,
                  total_optimizer_updates=total_updates, epochs=args.epochs,
                  batch_size=args.batch_size, sequence_batch_size=args.sequence_batch_size,
                  sequence_length=args.sequence_length, seed=args.seed,
                  device=str(device), torch_threads=torch.get_num_threads(),
                  torch_version=str(torch.__version__),
                  train_games=train_data.games, validation_games=validation_data.games,
                  train_positions=train_data.positions, validation_positions=validation_data.positions,
                  probe_max_positions=args.probe_max_positions, probe_positions=probes.positions,
                  probe_definition='Fixed held-out histories, T=1; legality separate from safety; zero limit means full validation',
                  dormant_definition='Relative channel mean absolute activity <= 0.01 on selected validation observations',
                  activation_statistics_definition='Exact reductions over selected observations; histogram values use uniform sampling',
                  update_sample_every=UPDATE_SAMPLE_EVERY,
                  histogram_max_observations=HISTOGRAM_MAX_OBSERVATIONS,
                  histogram_max_values_per_layer=HISTOGRAM_MAX_VALUES_PER_LAYER,
                  histogram_bins=HISTOGRAM_BINS)
    if saved:
        model.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        if scheduler:
            scheduler.load_state_dict(saved['scheduler_state_dict'])
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        if device.type == 'cuda' and saved.get('cuda_rng_state_all'):
            torch.cuda.set_rng_state_all(saved['cuda_rng_state_all'])
        train_loader.generator.set_state(saved['loader_generator_state'].cpu())
        log = saved['training_log']
        # Restore only the last completed epoch if a newer metrics write was interrupted.
        log.update(status='training', config=config)
        first_epoch = saved['epoch'] + 1
        best_loss = log['best_validation_loss']
        if 'best_selection_score' not in log and args.selection_metric == 'loss':
            log['best_selection_score'] = -best_loss
    else:
        run_dir.mkdir(parents=True)
        write_json(run_dir / 'data_split.json', dict(train=list(map(str, train_paths)),
                                                    validation=list(map(str, validation_paths)), probe_games=probes.games))
        log = dict(schema_version=4, status='training', config=config, epochs=[],
                   best_epoch=None, best_validation_loss=None)
        first_epoch, best_loss = 1, float('inf')
    metrics_path = run_dir / 'metrics.json'
    write_json(metrics_path, log)
    print(f'Device: {device}; model: {args.model_type}; BatchNorm: {args.batch_norm}', flush=True)
    print(f'Train: {train_data.games:,} games / {train_data.positions:,} moves; validation: '
          f'{validation_data.games:,} games / {validation_data.positions:,} moves', flush=True)
    print(f'AdamW: betas={optimizer.param_groups[0]["betas"]}, eps={args.epsilon}, decay={args.weight_decay}; '
          f'global gradient clip={args.grad_clip_norm}', flush=True)
    if scheduler:
        print(f'Schedule: {args.schedule}; warmup={args.warmup_updates:,} updates; total={total_updates}; '
              f'LR {args.learning_rate} -> {args.peak_learning_rate} -> {args.min_learning_rate}', flush=True)
    else:
        print(f'Schedule: none; constant LR {args.learning_rate}', flush=True)
    print(f'Diagnostics: {run_dir}; legal probe: {probes.positions:,} held-out moves', flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    activation_report = None
    for epoch in range(first_epoch, args.epochs + 1):
        learning_rates = [group['lr'] for group in optimizer.param_groups]
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, device, f'Epoch {epoch}/{args.epochs} train',
                                  optimizer, scheduler=scheduler, max_grad_norm=args.grad_clip_norm,
                                  sequence_length=args.sequence_length)
        train_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        validation_metrics = run_epoch(model, validation_loader, device, f'Epoch {epoch}/{args.epochs} eval',
                                       sequence_length=args.sequence_length)
        validation_seconds = time.perf_counter() - validation_started
        diagnostics_started = time.perf_counter()
        activation_report = final_histograms(
            model, validation_loader, device, bins=HISTOGRAM_BINS,
            max_observations=HISTOGRAM_MAX_OBSERVATIONS,
            max_values_per_layer=HISTOGRAM_MAX_VALUES_PER_LAYER,
            collect_histograms=epoch == args.epochs, seed=args.seed)
        validation_metrics.update(activation_observations=activation_report['activation_observations'],
                                  activation_stats=activation_report['activation_stats'],
                                  policy_probes=probes.evaluate(model, device, args.sequence_length,
                                                               PROBE_GRADIENT_TRAJECTORIES))
        row = dict(epoch=epoch, learning_rates=learning_rates, train=train_metrics,
                   validation=validation_metrics, train_seconds=train_seconds,
                   validation_seconds=validation_seconds,
                   diagnostics_seconds=time.perf_counter() - diagnostics_started)
        log['epochs'].append(row)
        best_loss = min(best_loss, validation_metrics['loss'])
        log['best_validation_loss'] = best_loss
        score = checkpoint_score(validation_metrics, args.selection_metric)
        previous_score = log.get('best_selection_score')
        improved = previous_score is None or score > previous_score
        if improved:
            log.update(best_epoch=epoch, best_selection_score=score,
                       selection_metric=args.selection_metric,
                       best_selection_value=-score if args.selection_metric == 'loss' else score)
        # Checkpoints resume at epoch boundaries, including shuffle and optimizer state.
        checkpoint = dict(kind=args.model_type, activation=args.activation, negative_slope=args.negative_slope,
                          model_config=model_config,
                          state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                          optimizer_state_dict=optimizer.state_dict(),
                          scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                          torch_rng_state=torch.get_rng_state(),
                          cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                          loader_generator_state=train_loader.generator.get_state(),
                          epoch=epoch, metrics=row, config=config, training_log=log,
                          metrics_json=str(metrics_path.resolve()))
        if improved:
            save_checkpoint(checkpoint, run_dir / 'best_policy.pt')
            if args.output:
                temporary = args.output.with_suffix(args.output.suffix + '.tmp')
                shutil.copyfile(run_dir / 'best_policy.pt', temporary)
                temporary.replace(args.output)
        save_checkpoint(checkpoint, run_dir / 'last_policy.pt')
        write_json(metrics_path, log)
        print(f'Epoch {epoch}/{args.epochs} | train {train_metrics["loss"]:.4f} | '
              f'val {validation_metrics["loss"]:.4f}, top1 {validation_metrics["top1"]:.2%} | '
              f'clipped {train_metrics["gradient_global_l2"]["clipped_fraction"]:.2%}', flush=True)
    if activation_report is None:
        checkpoint = saved
        activation_report = final_histograms(model, validation_loader, device, bins=HISTOGRAM_BINS,
                                              max_observations=HISTOGRAM_MAX_OBSERVATIONS,
                                              max_values_per_layer=HISTOGRAM_MAX_VALUES_PER_LAYER, seed=args.seed)
    save_checkpoint(checkpoint, run_dir / 'final_policy.pt')
    activation_report.update(epoch=args.epochs, model_state='final_epoch')
    write_json(run_dir / 'histograms.json', activation_report)
    plot_diagnostics(log['epochs'], activation_report, run_dir)
    log.update(status='completed', wall_seconds=log.get('wall_seconds', 0.) + time.perf_counter() - started)
    write_json(metrics_path, log)
    print(f'Saved metrics, plots, and checkpoints to {run_dir}', flush=True)
    return run_dir


def save_checkpoint(checkpoint, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


if __name__ == "__main__":
    main(sys.argv[1:])
