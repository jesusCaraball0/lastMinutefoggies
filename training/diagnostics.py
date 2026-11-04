"""Training diagnostics and headless plots; no arena runtime dependencies."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch


def write_json(path, value):
    """Keep the previous complete record if an epoch write is interrupted."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def norm_statistics(values, steps):
    finite = values[np.isfinite(values)].astype(np.float64)
    if not steps:
        return None
    return {'mean': float(finite.mean(dtype=np.float64)) if len(finite) else None,
            'median': float(np.median(finite)) if len(finite) else None,
            'p95': float(np.percentile(finite, 95)) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None,
            'steps': steps, 'nonfinite_count': steps - len(finite)}


class GradientNorms:
    """Exact epoch quantiles using device buffers, with no per-step CPU sync."""

    BUFFER_SIZE = 2048

    def __init__(self, model):
        self.layers = {name: list(module.parameters(recurse=False))
                       for name, module in model.named_modules()
                       if any(p.requires_grad for p in module.parameters(recurse=False))}
        self.buffers = defaultdict(list)
        self.steps = defaultdict(int)
        self.observed = {stage: defaultdict(int) for stage in ('pre_clip', 'post_clip')}

    def record(self, stage):
        norms = []
        active = []
        for name, parameters in self.layers.items():
            squares = [p.grad.detach().float().square().sum()
                       for p in parameters if p.grad is not None]
            norm = torch.stack(squares).sum().sqrt() if squares else None
            norms.append(norm)
            if norm is not None:
                active.append(norm)
                self.observed[stage][name] += 1
        if not active:
            return
        global_norm = torch.stack(active).square().sum().sqrt()
        row = torch.stack([x if x is not None else global_norm.new_tensor(float('nan'))
                           for x in norms] + [global_norm])
        index = self.steps[stage] % self.BUFFER_SIZE
        if index == 0:
            self.buffers[stage].append(row.new_empty((self.BUFFER_SIZE, len(row))))
        self.buffers[stage][-1][index].copy_(row)
        self.steps[stage] += 1

    def arrays(self, stage):
        chunks = self.buffers[stage]
        if not chunks:
            return np.empty((0, len(self.layers) + 1), dtype=np.float32)
        # Trim the final allocation before copying it to the CPU.
        used = (self.steps[stage] - 1) % self.BUFFER_SIZE + 1
        return torch.cat(chunks[:-1] + [chunks[-1][:used]]).cpu().numpy()

    def summary(self):
        arrays = {stage: self.arrays(stage) for stage in ('pre_clip', 'post_clip')}
        return {name: {stage: norm_statistics(values[:, index], self.observed[stage][name])
                       for stage, values in arrays.items()}
                for index, name in enumerate(self.layers)}

    def global_summary(self):
        arrays = {stage: self.arrays(stage)[:, -1] for stage in ('pre_clip', 'post_clip')}
        n = min(map(len, arrays.values()))
        before, after = arrays['pre_clip'][:n], arrays['post_clip'][:n]
        finite = np.isfinite(before) & np.isfinite(after)
        # Measure actual rescaling; an unclipped feedforward step is unchanged.
        clipped = finite & (after < before * (1 - 1e-6))
        result = {stage: norm_statistics(values, len(values)) for stage, values in arrays.items()}
        result.update(compared_steps=int(finite.sum()), clipped_steps=int(clipped.sum()),
                      clipped_fraction=float(clipped.sum() / finite.sum()) if finite.any() else None)
        return result


def distribution(values, bins):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    counts, edges = np.histogram(finite, bins=bins)
    return {'count': len(values), 'nonfinite_count': len(values) - len(finite),
            'mean': float(finite.mean()) if len(finite) else None,
            'std': float(finite.std()) if len(finite) else None,
            'min': float(finite.min()) if len(finite) else None,
            'max': float(finite.max()) if len(finite) else None,
            'counts': counts.tolist(), 'bin_edges': edges.tolist()}


class ActivationStatistics:
    """Exact reductions across all selected validation activations, per channel."""

    def __init__(self):
        self.layers = {}

    def update(self, name, values):
        axis = 1 if values.ndim == 4 else values.ndim - 1
        channels = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
        finite = np.isfinite(channels)
        safe = np.where(finite, channels, 0).astype(np.float64)
        batch = {
            'count': finite.sum(1), 'nonfinite_count': (~finite).sum(1),
            'sum': safe.sum(1), 'squared_sum': np.square(safe).sum(1),
            'absolute_sum': np.abs(safe).sum(1),
            'zero': ((channels == 0) & finite).sum(1),
            'near_zero': ((np.abs(channels) <= 1e-6) & finite).sum(1),
            'negative': ((channels < 0) & finite).sum(1),
            'positive': ((channels > 0) & finite).sum(1),
            'saturated': ((np.abs(channels) >= .95) & finite).sum(1),
        }
        # Channel means per observation distinguish input-dependent activity
        # from a channel that emits the same spatial pattern on every board.
        per_observation = safe.reshape(len(channels), len(values), -1).mean(-1)
        batch.update(observation_count=np.full(len(channels), len(values)),
                     observation_sum=per_observation.sum(1),
                     observation_squared_sum=np.square(per_observation).sum(1))
        if name not in self.layers:
            self.layers[name] = batch
        else:
            for key, value in batch.items():
                self.layers[name][key] += value

    def summary(self):
        def describe(raw):
            count = int(raw['count'])
            mean = float(raw['sum'] / count) if count else None
            variance = max(0, float(raw['squared_sum'] / count - mean ** 2)) if count else None
            result = {'count': count, 'nonfinite_count': int(raw['nonfinite_count']),
                      'mean': mean, 'std': float(np.sqrt(variance)) if count else None,
                      'mean_absolute': float(raw['absolute_sum'] / count) if count else None}
            observations = int(raw['observation_count'])
            observation_variance = max(0., float(raw['observation_squared_sum'] / observations
                                                - (raw['observation_sum'] / observations) ** 2)) if observations else 0.
            result['observation_mean_std'] = float(np.sqrt(observation_variance))
            for field in ('zero', 'near_zero', 'negative', 'positive'):
                result[field + '_fraction'] = float(raw[field] / count) if count else None
            return result

        result = {}
        for name, raw in self.layers.items():
            aggregate = describe({key: value.sum() for key, value in raw.items()})
            observed = raw['count'] > 0
            aggregate['channels'] = len(observed)
            mean_absolute = np.divide(raw['absolute_sum'], raw['count'],
                                      out=np.zeros_like(raw['absolute_sum']), where=observed)
            baseline = mean_absolute[observed].mean() if observed.any() else 0.
            scores = mean_absolute / baseline if baseline else np.zeros_like(mean_absolute)
            aggregate['dormant_threshold'] = 0.01
            aggregate['dormant_fraction'] = float(np.mean(scores[observed] <= .01)) if observed.any() else None
            aggregate['relative_activity_scores'] = scores.tolist()

            aggregate['all_zero_channel_fraction'] = float(np.mean(
                raw['zero'][observed] == raw['count'][observed])) if observed.any() else None
            aggregate['never_positive_channel_fraction'] = float(np.mean(
                raw['positive'][observed] == 0)) if observed.any() else None
            if name == 'memory':
                count = int(raw['count'].sum())
                aggregate['saturation_fraction_abs_ge_0_95'] = float(raw['saturated'].sum() / count) if count else None
            # Conv channels and dense/GRU features are detailed; avoid logging
            # thousands of flatten/logit units per epoch.
            if len(observed) <= 256:
                aggregate['per_channel'] = [describe({key: value[i] for key, value in raw.items()})
                                            for i in range(len(observed))]
                for i, channel in enumerate(aggregate['per_channel']):
                    channel['relative_activity'] = float(scores[i])
            result[name] = aggregate
        return result


class ActivationReservoir:
    """Uniform bounded histogram sample using independent random priorities."""

    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.samples = {}

    def update(self, name, values):
        values = values.reshape(-1)
        priorities = self.rng.random(len(values))
        if name in self.samples:
            previous_priorities, previous_values = self.samples[name]
            priorities = np.concatenate((previous_priorities, priorities))
            values = np.concatenate((previous_values, values))
        if len(values) > self.capacity:
            keep = np.argpartition(priorities, self.capacity - 1)[:self.capacity]
            priorities, values = priorities[keep], values[keep]
        self.samples[name] = priorities, values.copy()


def final_histograms(model, loader, device, *, bins=60, max_observations=512,
                     max_values_per_layer=50000, collect_histograms=True, seed=7980):
    """Validation activation statistics; optionally include final histograms.

    Statistics use every value in the selected, nonpadding observations.
    Histograms use uniform random reservoirs, independent of training RNG.
    """
    from training.datasets import observation_planes, TOKEN

    if min(bins, max_observations, max_values_per_layer) < 1:
        raise ValueError("Histogram bins and sample limits must be positive")
    reservoir = ActivationReservoir(max_values_per_layer, seed)
    statistics = ActivationStatistics()
    active_rows = None
    handles = []

    def capture(name):
        def hook(module, inputs, output):
            if not isinstance(output, torch.Tensor):
                return
            values = output.detach()[active_rows].float().cpu().numpy()
            statistics.update(name, values)
            if collect_histograms:
                reservoir.update(name, values)
        return hook

    was_training = model.training
    observations = 0
    try:
        for name, module in model.named_modules():
            if name and not list(module.children()):
                handles.append(module.register_forward_hook(capture(name)))
        model.eval()
        with torch.inference_mode():
            for tokens, _ in loader:
                hidden = None
                for step in range(tokens.shape[1]):
                    real = (tokens[:, step] != TOKEN['?']).any(dim=-1)
                    rows = real.nonzero().flatten()[:max_observations - observations]
                    if not len(rows):
                        continue
                    active_rows = torch.zeros(tokens.shape[0], dtype=torch.bool, device=device)
                    active_rows[rows.to(device)] = True
                    planes = observation_planes(tokens[:, step:step + 1], device)
                    _, hidden = model.forward_chunk(planes, hidden)
                    observations += len(rows)
                    if observations >= max_observations:
                        break
                if observations >= max_observations:
                    break
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    weights = {}
    if collect_histograms:
        for name, module in model.named_modules():
            parameters = [p.detach().float().cpu().numpy().reshape(-1)
                          for key, p in module.named_parameters(recurse=False) if key.startswith('weight')]
            if parameters:
                weights[name] = distribution(np.concatenate(parameters), bins)
    activation_stats = statistics.summary()
    for name, module in model.named_modules():
        if name in activation_stats:
            activation_stats[name]['module_type'] = type(module).__name__
    return {'model_state': 'final_epoch', 'activation_observations': observations,
            'activation_sample': {'source': 'first nonpadding validation observations',
                                  'max_observations': max_observations,
                                  'max_values_per_layer': max_values_per_layer,
                                  'histogram_method': 'uniform random priority reservoir',
                                  'seed': seed, 'statistics': 'all values in selected observations'},
            'activation_stats': activation_stats, 'weights': weights,
            'activations': {name: distribution(values, bins)
                            for name, (_, values) in reservoir.samples.items()}}


def plot_diagnostics(history, histograms, output, *, x_key="epoch", x_label="Epoch", state_label="Final epoch"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    output = Path(output)
    epochs = [row[x_key] for row in history]

    def save(fig, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        for axis in fig.axes:
            low, high = axis.get_ylim()
            if axis.get_yscale() == 'symlog' and low > 0 and high / low < 10:
                axis.set_yscale('linear')
            if axis.get_xlabel() in ("Epoch", "Optimizer updates"):
                axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for split in ('train', 'validation'):
        ax.plot(epochs, [row[split]['loss'] for row in history], label=split)
    ax.set(xlabel=x_label, ylabel='Cross-entropy loss', title='Training and validation loss')
    ax.legend()
    save(fig, output / 'loss.png')

    fig, ax = plt.subplots(figsize=(8, 4))
    lr_trace = [rates for row in history for rates in row['train'].get('learning_rates_by_step', [])]
    for group in range(len(history[0]['learning_rates'])):
        if lr_trace:
            ax.plot(range(1, len(lr_trace) + 1), [rates[group] for rates in lr_trace], label=f'Group {group}')
        else:
            ax.plot(epochs, [row['learning_rates'][group] for row in history], label=f'Group {group}')
    ax.set(xlabel='Optimizer updates' if lr_trace else x_label, ylabel='Learning rate', title='Learning rate')
    ax.legend()
    save(fig, output / 'learning_rate.png')

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, stage in zip(axes, ('pre_clip', 'post_clip')):
        for layer in history[0]['train']['gradient_l2']:
            values = [row['train']['gradient_l2'][layer][stage] for row in history]
            if all(value is None for value in values):
                continue
            ax.plot(epochs, [value['mean'] if value else np.nan for value in values], label=layer)
        ax.set(ylabel='Mean per-step L2 norm', title=stage.replace('_', ' ').capitalize())
        ax.set_yscale('symlog', linthresh=1e-6)
        if ax.lines:
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    axes[-1].set_xlabel(x_label)
    save(fig, output / 'gradient_l2.png')

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, field in zip(axes, ('median', 'p95', 'max')):
        for layer in history[0]['train']['gradient_l2']:
            values = [row['train']['gradient_l2'][layer]['pre_clip'] for row in history]
            if all(value is None for value in values):
                continue
            ax.plot(epochs, [value[field] if value else np.nan for value in values], label=layer)
        ax.set(ylabel='Pre-clip L2 norm', title=field.capitalize())
        ax.set_yscale('symlog', linthresh=1e-6)
        if ax.lines:
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    axes[-1].set_xlabel(x_label)
    save(fig, output / 'gradient_quantiles.png')

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, stage in zip(axes[:2], ('pre_clip', 'post_clip')):
        for field in ('mean', 'median', 'p95', 'max'):
            ax.plot(epochs, [row['train']['gradient_global_l2'][stage][field]
                            for row in history], label=field)
        ax.set(title=stage.replace('_', ' ').capitalize(), ylabel='Global gradient L2')
        ax.set_yscale('symlog', linthresh=1e-6)
        ax.legend()
    axes[2].plot(epochs, [row['train']['gradient_global_l2']['clipped_fraction']
                         for row in history])
    axes[2].set(xlabel=x_label, ylabel='Fraction of updates clipped', ylim=(-.02, 1.02))
    save(fig, output / 'gradient_global_l2.png')

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name in history[0]['train']['update_to_weight']['layers']:
        measurements = [row['train']['update_to_weight']['layers'][name] for row in history]
        if all(value is None for value in measurements):
            continue
        for ax, statistic in zip(axes, ('median', 'p95')):
            ax.plot(epochs, [value['relative_l2'][statistic] if value else np.nan
                            for value in measurements], label=name)
            ax.set(title=f'Actual update / parameter L2: {statistic}', ylabel='Ratio')
            ax.set_yscale('symlog', linthresh=1e-9)
            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    axes[-1].set_xlabel(x_label)
    save(fig, output / 'update_to_weight.png')

    fig, ax = plt.subplots(figsize=(10, 4))
    for name, layer in histograms['activation_stats'].items():
        if layer['module_type'] in ('ReLU', 'LeakyReLU', 'GRUCell'):
            ax.plot(epochs, [row['validation']['activation_stats'][name]['dormant_fraction']
                            for row in history], label=name)
    ax.set(xlabel=x_label, ylabel='Fraction', ylim=(-.02, 1.02),
           title='Relative mean-absolute activity <= 0.01 of layer average')
    if ax.lines:
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
    save(fig, output / 'dormant_features.png')

    probes = [row['validation']['policy_probes'] for row in history]
    legal = [probe['legal_moves'] for probe in probes]
    fig, axes = plt.subplots(4, 1, figsize=(9, 12), sharex=True)
    axes[0].plot(epochs, [value['legal_probability_mass'] for value in legal])
    axes[0].set(ylabel='Probability', title='Raw probability assigned to game-legal actions', ylim=(0, 1))
    axes[1].plot(epochs, [probe['raw_nll'] for probe in probes], label='Raw')
    axes[1].plot(epochs, [value['legal_nll'] for value in legal], label='Legal-masked')
    axes[1].set(ylabel='NLL', title='Fixed held-out probe moves (temperature 1)')
    axes[1].legend()
    axes[2].plot(epochs, [value['legal_top1'] for value in legal], label='Legal-masked accuracy')
    axes[2].plot(epochs, [probe['raw_top1'] for probe in probes], label='Raw accuracy')
    axes[2].set(ylabel='Accuracy', ylim=(0, 1))
    axes[2].legend()
    for key in ('preferred_legal_move_removed_by_safety', 'probability_mass_removed_by_safety'):
        axes[3].plot(epochs, [value[key] for value in legal], label=key)
    axes[3].set(xlabel=x_label, ylabel='Fraction', ylim=(0, 1), title='Safety heuristic intervention')
    axes[3].legend(fontsize=7)
    save(fig, output / 'legal_policy.png')

    if probes[0]['memory'] is not None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for mode in ('normal', 'reset'):
            axes[0].plot(epochs, [probe['memory'][mode + '_nll'] for probe in probes], label=mode)
        axes[0].set(ylabel='NLL', title='Does recurrent history help on fixed validation trajectories?')
        axes[0].legend()
        axes[1].plot(epochs, [probe['memory']['delta_nll_reset_minus_normal'] for probe in probes])
        axes[1].axhline(0, color='gray', linestyle='--')
        axes[1].set(xlabel=x_label, ylabel='Reset NLL - normal NLL', title='Positive means history helps')
        save(fig, output / 'memory_usefulness.png')
        fig, ax = plt.subplots(figsize=(9, 4))
        for row in (history[0], history[-1]):
            lags = row['validation']['policy_probes']['gradient_by_lag']['lags']
            ax.plot([int(k) for k in lags], [v['mean'] for v in lags.values()],
                    label=f"{x_label} {row[x_key]}")
        ax.set(xlabel='Lag (observations)', ylabel='L2 of loss gradient w.r.t. hidden state',
               title='Temporal gradient flow inside the truncated window')
        ax.set_yscale('symlog', linthresh=1e-12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        save(fig, output / 'gradient_by_lag.png')

    for kind in ('weights', 'activations'):
        for layer, stats in histograms[kind].items():
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.stairs(stats['counts'], stats['bin_edges'], fill=True)
            ax.set(xlabel='Value', ylabel='Count', title=f'{state_label} {kind}: {layer}')
            save(fig, output / kind / f'{layer}.png')
