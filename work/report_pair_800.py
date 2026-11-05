"""Record the completed historical-value warmup and the subsequent paired PPO phase."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json, hashlib
import torch
from training.diagnostics import write_json

warm=json.loads((ROOT/'out/rl/value-pretrain-after-400.json').read_text())
a=json.loads((ROOT/'out/rl/paired-after-value-warmup-800-analysis.json').read_text())
prior=json.loads((ROOT/'out/rl/paired-400-analysis.json').read_text())
fresh=json.loads((ROOT/'out/rl/paired-after-value-warmup-800-evaluation.json').read_text())
oldfresh=json.loads((ROOT/'out/rl/paired-400-fresh-evaluation.json').read_text())
assert warm['status']=='completed' and len(warm['models'])==2 and warm['epochs']==1
sources={'cnn-c64-h256-8ebd1cf653':ROOT/'out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt',
         'residual-c32-h128-518d814dce':ROOT/'out/experiments/residual-medium-27890681/best_policy.pt'}
raw={}
for name,d in a.items():
 folder=ROOT/'out/rl'/name;raw[name]=json.loads((folder/'metrics.json').read_text())
 saved=torch.load(folder/'policy.pt',map_location='cpu',weights_only=True)
 original=torch.load(sources[name],map_location='cpu',weights_only=True)
 assert saved['update']==800 and saved['training_log']['status']=='completed'
 assert len(list(folder.glob('*.pt')))==1 and not (folder/'.paired-commit.tmp').exists()
 assert saved['training_log']['value_pretraining'][0]['after_ppo_update']==400
 assert saved['training_log']['value_pretraining'][0]['actor_and_buffers_unchanged']
 assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
 sha=hashlib.sha256(sources[name].read_bytes()).hexdigest()
 assert sha==saved['training_log']['source_sha256']
 buffers=[k for k in saved['state_dict'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
 assert all(torch.equal(saved['state_dict'][k],original['state_dict'][k]) for k in buffers)
 d['verification']=dict(source_sha256_unchanged=sha,all_checkpoint_tensors_finite=True,unchanged_bn_buffers=len(buffers),
                        top_level_rl_model_files=1,best_update=saved['best_policy']['update'])
 d['phase2_termination_reasons']={}
 for row in raw[name]['updates'][400:]:
  for k,v in row['rollout']['reasons'].items():d['phase2_termination_reasons'][k]=d['phase2_termination_reasons'].get(k,0)+v
write_json(ROOT/'out/rl/paired-after-value-warmup-800-analysis.json',a)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(13,9))
for name,d in a.items():
 log=raw[name];label='Full CNN' if name.startswith('cnn') else 'Residual'
 ev=log['evaluations'];rows=log['updates'];x=[r['update'] for r in rows]
 axes[0,0].plot([e['update'] for e in ev],[e['mean_score'] for e in ev],label=label)
 def rolling(v,n=25):return [sum(v[max(0,i-n+1):i+1])/len(v[max(0,i-n+1):i+1]) for i in range(len(v))]
 axes[1,0].plot(x,rolling([r['ppo']['value_fit']['monte_carlo_before']['rmse'] for r in rows]),label=label)
 axes[1,1].plot(x,rolling([r['ppo']['value_fit']['monte_carlo_before']['explained_variance'] for r in rows]),label=label)
for i,stem in enumerate(['full_cnn','residual']):
 for j,(data,key,label) in enumerate([(oldfresh,stem+'_ppo_latest','Update400'),(fresh,stem+'_ppo_latest','Warmup + update800'),(fresh,stem+'_ppo_best','Selected best')]):
  axes[0,1].bar(i+(j-1)*.24,data['policies'][key]['score'],width=.23,label=label if i==0 else None)
axes[0,0].set(title='Fixed-seed evaluation (40 games/policy)',xlabel='PPO update',ylabel='Score')
axes[0,1].set(title='Repeated benchmark (100 games/policy)',xticks=[0,1],xticklabels=['Full CNN','Residual'],ylim=(0,1),ylabel='Score')
axes[1,0].set(title='Fresh-rollout MC RMSE, rolling25 updates',xlabel='PPO update')
axes[1,1].set(title='Fresh-rollout explained variance, rolling25',xlabel='PPO update')
for i,ax in enumerate(axes.flat):
 if i!=1:ax.axvline(400,color='gray',linestyle='--',label='Historical value warmup')
 ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'out/rl/paired-after-value-warmup-800-comparison.png',dpi=150);plt.close(fig)

lines=[]
lines.append('''<a id="stage-39"></a>

## 39. Completed historical value warmup and paired PPO through update800

This completes the experiment authorized in stage38: exactly one historical training epoch fitting only both value heads, followed by400 additional paired PPO updates (401–800). Existing actor architectures, rewards, opponent pairing, PPO recipe and evaluation protocol were retained. No un-warmed400→800 control was run, so subsequent policy gains cannot be attributed solely to pretraining. All preceding400-update evidence is preserved in its original analysis/evaluation files.

### Historical head-only fitting

The existing full-data split supplied55060 usable training games/5332962 positions and6114 validation games/592666 positions, after excluding241/30 non-gameplay-termination files. Historical inputs were the player's own pre-decision fog views. Terminal return targets used +1/−1/+0.1 and gamma0.99 per own decision. See stage38 and the raw phase report for result/reason distributions, filtering and split identity.

One shuffled epoch had5208 optimizer batches, batch1024, CPU4threads, Adam LR0.001/eps1e-5/default betas/no decay, half-MSE and head-only gradient clip0.5. Encoders, policy heads and BN buffers remained frozen, with a bitwise assertion before commit. Actor Adam moments/RNG and earlier evaluated best snapshots were preserved; critic moments were cleared before on-policy continuation. Both heads committed to the same persistent policy.pt files. Warmup is recorded at PPO update400 and does not increment its PPO counter.

| Model | Historical validation RMSE before / after | Validation explained variance before / after | Validation MAE before / after | Online train RMSE | Head gradient clip fraction |
| --- | --- | --- | --- | --- | --- |''')
for name,d in warm['models'].items():
 b=d['validation_before'];c=d['validation_after']
 lines.append(f"| {name} | {b['rmse']:.6f} / {c['rmse']:.6f} | {b['explained_variance']:.6f} / {c['explained_variance']:.6f} | {b['mae']:.6f} / {c['mae']:.6f} | {d['train_online']['rmse']:.6f} | {d['gradient_clipped_fraction']:.3%} |")
lines.append(f"\nWarmup wall time: {warm['wall_seconds']:.2f}s. Historical error is on a different state/policy/episode-length distribution than current-peer PPO error; compare before versus after on the same historical validation set, not historical RMSE directly against PPO RMSE. One epoch does not prove convergence or head-capacity sufficiency.\n")
lines.append('''### Continued paired PPO

The additional6400 shared training games bring total shared PPO experience to12800 games, not25600. Each model learns from its own decisions in the same games. Fixed-baseline evaluations remain40 games/model every10 updates, using the existing isolated seed9137. Best selection spans the entire run; it was not reset by warmup. No historical opponents were added to PPO training.

| Model | Phase401–800 W / L / D | Score | Mean terminal reward/game | Phase decisions | Selected best update | Selected fixed-seed score | Final fixed-seed score |
| --- | --- | --- | --- | --- | --- | --- | --- |''')
for name,d in a.items():
 q=d['phase2'];dec=sum(r['ppo']['transitions'] for r in raw[name]['updates'][400:])
 lines.append(f"| {name} | {q['wins']} / {q['losses']} / {q['draws']} | {q['score']:.3%} | {q['mean_reward']:.6f} | {dec:,} | {d['best_eval']['update']} | {d['best_eval']['mean_score']:.2%} | {d['final_eval']['mean_score']:.2%} |")
for name,d in a.items():lines.append(f"\n{name} phase2 reasons: {d['phase2_termination_reasons']}.")
lines.append('''

### Critic transfer, losses and update stability

Window measurements use pre-update predictions on fresh rollouts. RMSE pools squared errors by decision; explained variance and PPO loss/health values are update means, not pooled variance ratios. First10 post-warmup updates contain160 shared games, each50-update window800 games. Opponents/state distributions evolve, so comparisons are descriptive.

| Model | Window | MC RMSE | Mean MC explained variance | Critic half-MSE loss | Actor loss | Endpoint KL | PPO ratio clip | Gradient clip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |''')
for name,d in a.items():
 for label,q in [('pre-warmup last50',prior[name]['last50']),('post-warmup first10',d['phase2_first10']),('post-warmup first50',d['phase2_first50']),('post-warmup last50',d['last50'])]:
  v=q['value_fit']['monte_carlo_before'];m=q['ppo']
  lines.append(f"| {name} | {label} | {v['rmse']:.6f} | {v['mean_update_explained_variance']:.6f} | {m['value_loss']:.6f} | {m['policy_loss']:.6f} | {m['endpoint_kl']:.6f} | {m['ratio_clip_fraction']:.3%} | {m['gradient_clipped_fraction']:.3%} |")
lines.append('''

Large raw gradients and frequent global clipping must be interpreted together with actual policy KL, finite weights/losses and update ratios. No causal diagnosis of gradient explosion follows solely from the clipping frequency. Full norms/layer updates, entropy, throughput, reward/outcome curves and fixed-probe activation histograms remain in the canonical per-model metrics/plots. PPO figures mark the value-warmup boundary at400.

### Repeated trajectory benchmark

Evaluated latest and selected-best policies against the same two frozen original baselines,50 games/opponent,100 games/policy, temperature0.9,max256plies,seed572940. This repeats the earlier400-update benchmark protocol/seed; it is not a new unseen opponent distribution or untouched sampling-seed test. Reused update400 scores below are references, not new games. Differences of a few percentage points are unresolved by100 games per policy.

| Policy | Update400 reference score | Current selected update | Current W / L / D | Current score | Current reward/game |
| --- | --- | --- | --- | --- | --- |''')
for name,d in fresh['policies'].items():
 lines.append(f"| {name} | {d['previous_400_score']:.2%} | {d['selected_update']} | {d['wins']} / {d['losses']} / {d['draws']} | {d['score']:.2%} | {d['mean_reward']:.4f} |")
lines.append('''

Selected-best peaks are optimistic because the same small frozen benchmark chooses among many checkpoints. Fresh/repeated benchmark scores should not silently replace the stored best state. More PPO updates and warmup were combined here; a matched no-warmup continuation is needed to isolate whether pretraining caused policy improvement or degradation. Historical held-out head fitting and current-policy value transfer are distinct questions.

### Artifact and lifecycle verification

Source supervised checkpoint hashes remain unchanged; latest checkpoint tensors are finite; BN running buffers still match their inherited supervised values. Both checkpoints/logs complete update800, record the warmup after400, and have no unfinished transaction journal. Each variant still contains one top-level RL model file, with latest optimizer state plus independently preserved best weights. Existing first400 histories were extended rather than replaced. The full test suite passed62 tests during implementation; source compilation/diff checks passed. No additional large architecture or PSRO machinery was introduced.

Artifacts: out/rl/value-pretrain-after-400.json/.png; out/rl/paired-after-value-warmup-800-analysis.json; out/rl/paired-after-value-warmup-800-evaluation.json; out/rl/paired-after-value-warmup-800-comparison.png; per-model canonical metrics/plots/diagnostics. Logs are logs/value-pretrain-after-400.out/.err, logs/ppo-pair-401-800.out/.err and logs/ppo-pair-800-analysis.out/.err.

The two canonical PPO histories are extensions of existing training records, not two independent new experiments. The historical one-epoch warmup has its own combined summary artifact. Further interpretation and next recommendations are stated below after reviewing the measured values and charts.
''')
p=ROOT/'EXPERIMENT_HISTORY.md';text=p.read_text();marker='<a id="stage-39"></a>'
if marker in text:raise ValueError('Stage39 already exists; edit it in place rather than duplicate findings')
a0=text.index('Current update:');b0=text.index('\n',a0)
text=text[:a0]+'Current update: [Completed historical value warmup and PPO through800](#stage-39). Earlier pending status is historical.'+text[b0:]
text=text.replace('- [38. Historical value-head warmup, then PPO updates401–800](#stage-38)','- [38. Historical value-head warmup, then PPO updates401–800](#stage-38)\n- [39. Completed historical value warmup and PPO through800](#stage-39)')
p.write_text(text+'\n\n'+'\n'.join(lines)+'\n')
print('Verified checkpoints, saved comparison plot, appended measured stage39. Review results and add interpretation before final delivery.')
