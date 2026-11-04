#!/usr/bin/env python3
"""Match the residual stress test while completing cosine decay in 80k updates."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from starters.shared.model import create_policy
from training.datasets import observation_planes
from training.artifacts import EXPERIMENTS, unused_run_path, append_findings
from training.diagnostics import write_json
from training.probes import PolicyProbes
from training.size_experiment import warmup_cosine_lr
from training.stability_experiment import args_from_cli, run_case

REFERENCE = ROOT / 'out/experiments/residual-reduced-head-ablations-2560games-80k'


def calibration(checkpoint, probes):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = create_policy(saved['kind'], saved['activation'], **saved['model_config'])
    model.load_state_dict(saved['state_dict']); model.eval()
    confidence, correct, nll, entropy, brier = [], [], [], [], []
    with torch.inference_mode():
        for trajectory in probes.trajectories:
            scores = model.forward_chunk(observation_planes(trajectory['tokens'][None], torch.device('cpu')))[0][0]
            for index, (legal, _) in trajectory['masks'].items():
                target = legal.index(int(trajectory['targets'][index]))
                logp = scores[index, legal].double().log_softmax(0); p = logp.exp()
                confidence.append(float(p.max())); correct.append(int(p.argmax()) == target)
                nll.append(float(-logp[target])); entropy.append(float(-(p*logp).sum()))
                brier.append(float(p.square().sum()-2*p[target]+1))
    c = torch.tensor(confidence, dtype=torch.float64); a = torch.tensor(correct, dtype=torch.float64)
    ece = 0.
    for index in range(15):
        selected = (c >= index/15) & (c < (index+1)/15 if index < 14 else c <= 1)
        if selected.any(): ece += float(selected.double().mean()*(c[selected].mean()-a[selected].mean()).abs())
    mean = lambda values: sum(values)/len(values)
    return dict(positions=len(c), legal_accuracy=mean(correct), legal_nll=mean(nll),
                legal_mean_confidence=mean(confidence), legal_entropy=mean(entropy),
                legal_ece_15=ece, legal_brier=mean(brier))


def pooled_interval(report, start=40000):
    rows = [r for r in report['checkpoints'] if r['updates'] > start]
    stats = [r['train']['gradient_global_l2'] for r in rows]
    steps = sum(g['compared_steps'] for g in stats)
    return dict(start_exclusive=start, end=rows[-1]['updates'], steps=steps,
                clipped_fraction=sum(g['clipped_steps'] for g in stats)/steps,
                gradient_mean=sum(g['pre_clip']['mean']*g['pre_clip']['steps'] for g in stats)/steps,
                gradient_max=max(g['pre_clip']['max'] for g in stats))


def finish(destination, reference):
    old = json.loads((reference/'residual_3/metrics.json').read_text())
    new = json.loads((destination/'residual_3/metrics.json').read_text())
    assert new['status'] == 'completed'
    assert old['config']['model_config'] == new['config']['model_config']
    for key in ('seed','channels','hidden_size','batch_size','train_games','val_games','threads',
                'warmup_updates','optimizer','betas','epsilon','weight_decay','global_clip'):
        assert old['config'][key] == new['config'][key], key
    by_update = {r['updates']:r for r in new['checkpoints']}
    assert list(by_update) == list(range(5000,80001,5000))
    for row in old['checkpoints']:
        assert row['batch_stream_sha256'] == by_update[row['updates']]['batch_stream_sha256']
    old_lr = [v[0] for r in old['checkpoints'] for v in r['train']['learning_rates_by_step']]
    new_lr = [v[0] for r in new['checkpoints'] for v in r['train']['learning_rates_by_step']]
    assert len(new_lr) == len(old_lr) == 80000
    assert old_lr[:5820] == new_lr[:5820]
    assert all(abs(v-warmup_cosine_lr(i,5820,80000-5820,1e-4,1e-3,1e-5))<1e-15 for i,v in enumerate(new_lr))
    assert abs(new_lr[-1]-1e-5)<1e-15
    first_old = torch.load(reference/'residual_3/policy-005000.pt', map_location='cpu', weights_only=False)
    first_new = torch.load(destination/'residual_3/policy-005000.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(v,first_new['state_dict'][k]) for k,v in first_old['state_dict'].items())
    for row in new['checkpoints']:
        g = row['train']['gradient_global_l2']
        assert g['pre_clip']['nonfinite_count'] == g['post_clip']['nonfinite_count'] == 0
        assert g['post_clip']['max'] <= 5.0001
    manifest = json.loads((destination/'data_split.json').read_text())
    probes = PolicyProbes([Path(p) for p in manifest['validation']], new['validation_positions'])
    calibration_results = {'old':{}, 'fast_decay':{}}
    for step in (5000,20000,40000,80000):
        for label, path in (('old',reference),('fast_decay',destination)):
            calibration_results[label][str(step)] = calibration(path/f'residual_3/policy-{step:06d}.pt',probes)
    write_json(destination/'calibration.json',calibration_results)
    summary = dict(status='completed', reference=str(reference),
                   audit=dict(common_batches_match=True, warmup_lr_matches=True,
                              five_k_weights_identical=True, cosine_endpoint_verified=True,
                              finite_gradients=True, clipping_bound_verified=True), cases={})
    for label,r in (('old',old),('fast_decay',new)):
        final = r['checkpoints'][-1]
        summary['cases'][label] = dict(best_raw_nll=r['best_validation_loss'],best_raw_nll_updates=r['best_updates'],
                                      final_raw_nll=final['validation']['loss'],final_raw_accuracy=final['validation']['top1'],
                                      final_legal=r['full_validation_policy_probe']['legal_moves'],
                                      late_interval=pooled_interval(r), final_health=final['health'],
                                      calibration=calibration_results[label])
    summary['cases']['fast_decay'].update(best_legal_nll=new['best_legal_nll'],
                                         best_legal_nll_updates=new['best_legal_nll_updates'],
                                         best_legal_accuracy=new['best_legal_accuracy'],
                                         best_legal_accuracy_updates=new['best_legal_accuracy_updates'])
    common = (5000,20000,40000,80000)
    summary['cases']['fast_decay']['best_raw_nll_common_checkpoints'] = min(by_update[s]['validation']['loss'] for s in common)
    summary['cases']['old']['best_legal_nll_common_checkpoints'] = min(v['legal_nll'] for v in calibration_results['old'].values())
    summary['cases']['fast_decay']['best_legal_nll_common_checkpoints'] = min(v['legal_nll'] for v in calibration_results['fast_decay'].values())
    write_json(destination/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    for label,r in (('Previous long decay',old),('Faster decay',new)):
        rows=r['checkpoints']; x=[v['updates'] for v in rows]
        values=[[v['validation']['loss'] for v in rows],
                [100*v['validation']['top1'] for v in rows],
                [v['train']['learning_rates_by_step'][-1][0] for v in rows],
                [v['validation']['policy_probes']['legal_moves']['legal_nll'] for v in rows],
                [100*v['train']['gradient_global_l2']['clipped_fraction'] for v in rows],
                [100*v['health']['encoder.6']['top4_activity_share'] for v in rows]]
        if label == 'Previous long decay':
            values[3]=[calibration_results['old'][str(v['updates'])]['legal_nll'] for v in rows]
        for ax,y in zip(axes.flat,values): ax.plot(x,y,marker='o',label=label)
    for ax,title in zip(axes.flat,('Raw validation NLL','Raw validation accuracy (%)','Learning rate',
                                  'Full validation legal NLL','Updates clipped (%), preceding interval',
                                  'Top 4 dense features: activity share (%)')):
        ax.set(title=title,xlabel='Optimizer updates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Matched residual CNN: faster decay, one seed; old interval widths differ')
    fig.tight_layout();fig.savefig(destination/'comparison.png',dpi=150);plt.close(fig)
    lines=['## Residual CNN: faster learning-rate decay experiment','',
           f'Results directory: `{destination}`. Same cached 2,560 training games (248,396 moves), '
           '64 validation games (6,147 moves), seed 7980, batch 128, 16 channels / 64 hidden features, '
           'three residual blocks, default initialization, convolutional BN, dropout 0.1, AdamW '
           '(betas 0.9/0.99, eps 1e-8, weight decay 0.01), and global clip 5. '
           'Only the cosine horizon changes from 334,440 to 80,000 updates; warmup stays 5,820 steps '
           '(1e-4 to 1e-3), followed by decay to 1e-5.','',
           'Full raw and legal validation is evaluated every 5,000 updates. Additional evaluations '
           'preserve training RNG state; original 5k/20k/40k/80k evaluation points consume RNG as '
           'in the reference. All input hashes match at shared checkpoints, and the 5k checkpoint '
           'weights exactly reproduce the old run. No label smoothing or normalization changes.','',
           '| Schedule | Best saved raw NLL | Update | Final raw NLL | Final raw accuracy | Final legal NLL | Final legal accuracy |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,r in (('Previous long decay',old),('Faster decay',new)):
        f=r['checkpoints'][-1]; legal=r['full_validation_policy_probe']['legal_moves']
        lines.append(f"| {label} | {r['best_validation_loss']:.4f} | {r['best_updates']:,} | {f['validation']['loss']:.4f} | {f['validation']['top1']:.2%} | {legal['legal_nll']:.4f} | {legal['legal_top1']:.2%} |")
    lines+=['','The new run has denser checkpoint selection. For a like-for-like check among the original '
            'four checkpoints, its best raw NLL is '
            f"{summary['cases']['fast_decay']['best_raw_nll_common_checkpoints']:.4f}; best legal NLL is "
            f"{summary['cases']['fast_decay']['best_legal_nll_common_checkpoints']:.4f} versus "
            f"{summary['cases']['old']['best_legal_nll_common_checkpoints']:.4f} in the reference.",'',
            '| Schedule | Clipped updates, 40k–80k | Mean pre-clip gradient L2 | Max pre-clip gradient L2 | Final legal ECE |',
            '|---|---:|---:|---:|---:|']
    for label,key in (('Previous long decay','old'),('Faster decay','fast_decay')):
        r=summary['cases'][key];g=r['late_interval'];ece=r['calibration']['80000']['legal_ece_15']
        lines.append(f"| {label} | {g['clipped_fraction']:.2%} | {g['gradient_mean']:.3f} | {g['gradient_max']:.3f} | {ece:.4f} |")
    lines+=['','Clipping counts and gradient means are pooled across identical 40,001–80,000-update '
            'windows. Medians/p95 are not averaged across intervals. The plots show interval statistics '
            'with different old/new interval widths; they do not establish exact onset times.','',
            '| Update | LR used | Raw NLL | Raw accuracy | Full legal NLL | Full legal accuracy | Clipped updates in preceding 5k |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for x in new['checkpoints']:
        l=x['validation']['policy_probes']['legal_moves']
        lines.append(f"| {x['updates']:,} | {x['train']['learning_rates_by_step'][-1][0]:.3g} | {x['validation']['loss']:.4f} | {x['validation']['top1']:.2%} | {l['legal_nll']:.4f} | {l['legal_top1']:.2%} | {x['train']['gradient_global_l2']['clipped_fraction']:.2%} |")
    lines+=['','Best raw NLL remains `best_policy.pt`. Best legal NLL is `best_legal_nll_policy.pt` '
            f"at {new['best_legal_nll_updates']:,} updates ({new['best_legal_nll']:.4f}); "
            'best legal accuracy is `best_accuracy_policy.pt` '
            f"at {new['best_legal_accuracy_updates']:,} updates ({new['best_legal_accuracy']:.2%}). "
            'Best raw accuracy is saved separately as `best_raw_accuracy_policy.pt`.','',
            'Calibration uses T=1 and the complete legal action masks on the same validation games '
            '(15 equal-width ECE bins, no fitted temperature). One seed and 64 validation games limit '
            'generalization claims. No arena evaluation or production-default changes.','',
            f'![Residual decay comparison]({destination}/comparison.png)','']
    section='\n'.join(lines)
    append_findings(section)
    print(section,flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',type=Path,default=REFERENCE)
    parser.add_argument('--out-dir',type=Path,default=EXPERIMENTS)
    options=parser.parse_args()
    reference=options.reference_run.resolve()
    source=json.loads((reference/'residual_3/metrics.json').read_text())
    if source['status']!='completed':raise ValueError('Reference must be complete')
    destination=unused_run_path(options.out_dir.resolve(), 'residual-faster-cosine-decay-2560games-80k')
    destination.mkdir(parents=True)
    shutil.copy2(reference/'data_split.json',destination/'data_split.json')
    for split in ('train','validation'):
        for kind in ('observations','targets'):
            name=f'{split}-{kind}.npy';os.link(reference/name,destination/name)
    command=['--case','residual_3','--run-dir',str(destination),'--schedule-updates','80000',
             '--checkpoints',*map(str,range(5000,80001,5000)), '--full-validation-probes',
             '--preserve-legacy-eval-rng']
    for key in ('train_games','val_games','channels','hidden_size','batch_size','warmup_updates','seed','threads'):
        command += ['--'+key.replace('_','-'),str(source['config'][key])]
    args=args_from_cli(command)
    write_json(destination/'experiment.json',dict(status='running',reference=str(reference),arguments=command))
    print(f'Experiment: {destination}',flush=True)
    try:
        run_case(args)
        finish(destination,reference)
        write_json(destination/'experiment.json',dict(status='completed',reference=str(reference),arguments=command))
    except BaseException as error:
        write_json(destination/'experiment.json',dict(status='failed',reference=str(reference),error=str(error),arguments=command))
        raise
    print(f'Completed experiment and appended findings to EXPERIMENT_HISTORY.md: {destination}',flush=True)


if __name__=='__main__':main()
