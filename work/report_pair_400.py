import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json, hashlib, math
import torch
from training.diagnostics import write_json

analysis=json.loads((ROOT/'out/rl/paired-400-analysis.json').read_text())
fresh=json.loads((ROOT/'out/rl/paired-400-fresh-evaluation.json').read_text())
sources={'cnn-c64-h256-8ebd1cf653':ROOT/'out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt',
         'residual-c32-h128-518d814dce':ROOT/'out/experiments/residual-medium-27890681/best_policy.pt'}
for name,d in analysis.items():
 folder=ROOT/'out/rl'/name
 saved=torch.load(folder/'policy.pt',map_location='cpu',weights_only=True)
 original=torch.load(sources[name],map_location='cpu',weights_only=True)
 assert saved['update']==400 and saved['training_log']['status']=='completed'
 assert len(list(folder.glob('*.pt')))==1
 assert not (folder/'.paired-commit.tmp').exists()
 assert saved['best_policy']['update']==d['best_eval']['update']
 assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
 assert d['status']=='completed'
 sha=hashlib.sha256(sources[name].read_bytes()).hexdigest()
 assert sha==saved['training_log']['source_sha256']
 buffers=[k for k in saved['state_dict'] if k.endswith(('running_mean','running_var','num_batches_tracked'))]
 assert all(torch.equal(saved['state_dict'][k],original['state_dict'][k]) for k in buffers)
 d['verification']=dict(source_sha256_unchanged=sha,all_checkpoint_tensors_finite=True,unchanged_bn_buffers=len(buffers),
                        top_level_rl_model_files=1,best_update=saved['best_policy']['update'],
                        policy_head_delta_l2=float((saved['state_dict']['policy.weight']-original['state_dict']['policy.weight']).norm()))
 raw=json.loads((folder/'metrics.json').read_text())
 d['activation_probe']={phase:raw['activation_diagnostics'][i] for phase,i in [('initial',0),('final',-1)]}
write_json(ROOT/'out/rl/paired-400-analysis.json',analysis)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(13,9))
for name,d in analysis.items():
 raw=json.loads((ROOT/'out/rl'/name/'metrics.json').read_text())
 label='Full CNN' if name.startswith('cnn') else 'Medium residual'
 ev=raw['evaluations'];rows=raw['updates'];x=[r['update'] for r in rows]
 axes[0,0].plot([e['update'] for e in ev],[e['mean_score'] for e in ev],label=label)
 def rolling(values,n=25):return [sum(values[max(0,i-n+1):i+1])/len(values[max(0,i-n+1):i+1]) for i in range(len(values))]
 axes[1,0].plot(x,rolling([r['ppo']['value_fit']['monte_carlo_before']['rmse'] for r in rows]),label=label)
 axes[1,1].plot(x,rolling([r['ppo']['endpoint_kl'] for r in rows]),label=label)
colors=['#4c78a8','#f58518','#54a24b']
for i,stem in enumerate(['full_cnn','residual']):
 for j,(suffix,label) in enumerate([('initial','Original'),('ppo_best','Selected best'),('ppo_latest','Update 400')]):
  q=fresh['policies'][stem+'_'+suffix]
  axes[0,1].bar(i+(j-1)*.23,q['score'],width=.22,color=colors[j],label=label if i==0 else None)
axes[0,0].set(title='Fixed-seed benchmark during training (40 games/policy)',xlabel='PPO update',ylabel='Game score')
axes[0,1].set(title='Fresh-seed benchmark (100 games/policy)',ylabel='Game score',xticks=[0,1],xticklabels=['Full CNN','Medium residual'],ylim=(0,1))
axes[1,0].set(title='Fresh-rollout value RMSE (25-update rolling mean)',xlabel='PPO update',ylabel='RMSE vs discounted terminal reward')
axes[1,1].set(title='Endpoint KL (25-update rolling mean)',xlabel='PPO update',ylabel='Approximate KL')
for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(ROOT/'out/rl/paired-400-comparison.png',dpi=150);plt.close(fig)

p=ROOT/'EXPERIMENT_HISTORY.md';s=p.read_text();a=s.index('Current update:');b=s.index('\n',a)
s=s[:a]+'Current update: [Completed 400-update paired PPO experiment](#stage-36). Earlier pending status is historical.'+s[b:]
s=s.replace('- [35. Minimal paired PPO: both CNNs learn from the same games](#stage-35)','- [35. Minimal paired PPO: both CNNs learn from the same games](#stage-35)\n- [36. Full CNN and medium residual: 400 paired PPO updates](#stage-36)')
s+='''

<a id="stage-36"></a>

## 36. Full CNN and medium residual: 400 paired PPO updates

**Request:** run400 steps for each model, monitor, briefly summarize results/metrics and next recommendations. The stated interpretation was400 completed PPO updates per model, not400 individual optimizer minibatches. Both completed without training errors. This is the first sustained paired PPO run, not the earlier short-cap smoke checks.

### Recipe, scope and verification

Sources are the fullCNN64/256 supervised best from job27800942 and medium residual32/128 supervised best from job27890681. Canonical output locations are out/rl/cnn-c64-h256-8ebd1cf653/ and out/rl/residual-c32-h128-518d814dce/. Each keeps one policy.pt containing latest training state and the independently selected best; metrics, eight learning curves and fixed-probe histogram/statistic JSON are separate evidence. No original supervised file was overwritten. Source hashes match the hashes recorded at initialization; all latest checkpoint tensors are finite, all inherited BN running buffers are unchanged, both checkpoints/logs confirm completed update400, and no unfinished pair transaction journal remains.

Paired collection used16 shared color-balanced complete games/update, max256plies, terminal rewards+1/-1/+0.1, gamma0.99/lambda0.95 over learner decisions, temperature0.9, seed7980. Actor/encoder Adam LR1e-5; critic LR1e-3; Adam eps1e-5, default betas(.9,.999), no weight decay;4 optimization epochs, minibatch128, probability ratio clip0.2, value coefficient0.5 applied to half-MSE, entropy coefficient0.01, global gradient clip0.5, approximate-KL stop threshold1.5*0.02. No schedule, historical opponent sampling, population solver or architecture modification was added. Dropout remained off and BN running statistics stayed fixed consistently during collection and optimization. Actual device was local CPU with one Torch thread.

Each model evaluated20 color-balanced games against each of the two frozen supervised baselines at update0 and every10 updates:40 games per evaluation,41 evaluation events per model,3280 evaluation games total. Evaluation seed9137 was fixed and isolated from training RNG. A thread monitor was registered and paused after training completed; no retraining or hyperparameter changes occurred during monitoring. Logs: logs/ppo-pair-400.out/.err; stderr was empty. Per-model wall timers were1677.55s (fullCNN) and1663.23s (residual), approximately28minutes plus initial startup; these overlap and must not be summed. Shared collection used465.33s total; model optimization timers totaled241.42s and426.51s, excluding most evaluation/diagnostic overhead.

Analysis artifacts: out/rl/paired-400-analysis.json, out/rl/paired-400-fresh-evaluation.json, out/rl/paired-400-comparison.png. Analysis/evaluation scripts are retained in work/. The comparison chart summarizes the fixed-seed benchmark, fresh-seed benchmark, value RMSE and KL; full per-update/layer evidence remains in each variant's metrics/diagnostics.

### Shared training outcomes and actual update counts

There were **6400 distinct training games**, yielding6400 own-color episodes per learner. Summing both model logs as12800 games would double-count the same games. Terminations:6339 King capturable and61 threefold repetitions; draw fraction0.953125%. No training game reached the ply cap, so this run gives no evidence of exploiting cap draws. There were no bot/infrastructure failures. Mean reward is terminal reward per game, not discounted return; critic targets are discounted separately.

| Model | Wins / losses / draws | Standard score | Mean terminal reward | Learner decisions | Optimizer minibatch steps | KL early-stop updates |
| --- | --- | --- | --- | --- | --- | --- |
'''
for name,d in analysis.items():
 q=d['all'];s+=f"| {name} | {q['wins']} / {q['losses']} / {q['draws']} | {q['score']:.4%} | {q['mean_reward']:.8f} | {d['total_decisions']:,} | {d['optimizer_steps']:,} | {d['early_stop_updates']} |\n"
s+='''
Head-to-head score over the first50 updates was48.0625% fullCNN /51.9375% residual; the last50 updates was54.9375% /45.0625%. That is a moving matchup, not absolute playing strength. Both models can improve against the original baselines while their mutual score remains close to50%.

### Checkpoint-selection benchmark (same fixed seed used during training)

| Model | Initial score | Selected best update | Selected best score | Final update400 score |
| --- | --- | --- | --- | --- |
'''
for name,d in analysis.items():
 s+=f"| {name} | {d['initial_eval']['mean_score']:.2%} | {d['best_eval']['update']} | {d['best_eval']['mean_score']:.2%} | {d['final_eval']['mean_score']:.2%} |\n"
s+='''
These peaks were selected across41 repeated40-game benchmarks and are optimistic performance estimates. The residual's initial fixed-seed benchmark was40%, not55%; an intermediate commentary misstated it and was corrected from the stored metrics. Do not conflate these initial fixed-seed measurements with the separate fresh-seed baseline values below.

### Fresh-seed benchmark after completion

To check whether apparent gains survive new game trajectories, evaluated original policies, selected-best policies and latest update400 policies on **100 games each**,50 against each frozen original baseline, alternating colors. Seed572940, temperature0.9 and max256plies. Total600 additional evaluation games. Initial actor weights were unchanged; initial critics were zeroed as in PPO initialization. Neither training nor checkpoint selection consumed these games. This is a fresh trajectory sample against the same known opponent pool, not an independent opponent distribution.

| Policy | Selected update | Wins / losses / draws | Score | Terminal reward/game | Approximate descriptive score SE |
| --- | --- | --- | --- | --- | --- |
'''
for name,d in fresh['policies'].items():
 s+=f"| {name} | {d['selected_update'] if d['selected_update'] is not None else 'supervised source'} | {d['wins']} / {d['losses']} / {d['draws']} | {d['score']:.2%} | {d['mean_reward']:.4f} | {d['score_standard_error']:.4f} |\n"
s+='''
Update400 gains were+12percentage points for fullCNN (54→66%) and+15.5points for residual (45→60.5%) on this new seed. Selected-best scores were65% and59%. Their difference from the latest scores (1 and1.5points) is much smaller than the sampling uncertainty; this does not establish which checkpoint is superior. Each100-game score has roughly±9–10percentage points of nominal95% sampling uncertainty, and initial-versus-trained differences also have uncertainty. Common sampling seeds and opponent strata complicate independence assumptions; no formal paired significance or exploitability claim is made. Nevertheless, both trained models improving on new trajectories is more encouraging than the selected75/80% fixed-seed peaks alone. Original selected-best checkpoints remain selected by the original protocol; fresh evaluation did not silently replace them.

Per-opponent fresh scores:

| Policy | Frozen opponent | Score | W / L / D |
| --- | --- | --- | --- |
'''
for name,d in fresh['policies'].items():
 for opponent,v in d['opponents'].items():
  s+=f"| {name} | {Path(opponent).parent.name} | {v['score']:.2%} | {v['wins']} / {v['losses']} / {v['draws']} |\n"
s+='''
### Losses, critic quality and policy stability

Window comparisons below use first50 versus last50 updates (800 shared games/window). Losses are means over updates of the per-update minibatch-sample-weighted loss. Monte Carlo RMSE pools squared error by learner decisions; explained variance is an update-average, not a pooled variance ratio. Opponents and state distributions change during training, so these are descriptive trends rather than fixed-dataset learning curves. GAE critic loss is distinct from error against independently discounted terminal outcomes.

| Model | Window | Actor loss | Critic half-MSE loss | MC RMSE before update | Mean MC explained variance before update | Entropy (nats) | Endpoint KL | PPO ratio clipping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
'''
for name,d in analysis.items():
 for label in ['first50','last50']:
  q=d[label];m=q['ppo'];v=q['value_fit']['monte_carlo_before']
  s+=f"| {name} | {label} | {m['policy_loss']:.6f} | {m['value_loss']:.6f} | {v['rmse']:.6f} | {v['mean_update_explained_variance']:.6f} | {m['endpoint_entropy']:.6f} | {m['endpoint_kl']:.6f} | {m['ratio_clip_fraction']:.3%} |\n"
s+='''
Both critics improved RMSE only modestly (about0.846→0.822 fullCNN and0.846→0.818 residual); fresh-rollout explained variance remained near0.01–0.02 late in training. They are weak outcome predictors despite actor performance gains. This is not a converged critic, but does not prove PPO cannot improve with it or that a larger architecture is needed. Fixed-baseline evaluation has an additional opponent/state-distribution shift relative to current-peer training.

| Model | Optimizer-step-weighted gradient clipping fraction (whole run) | Mean pre-clip gradient L2 (last50 update means) | Largest pre-clip gradient L2 (whole run) | Largest endpoint KL (whole run) |
| --- | --- | --- | --- | --- |
'''
for name,d in analysis.items():
 s+=f"| {name} | {d['weighted_gradient_clip_fraction']:.5%} | {d['last50']['ppo']['pre_clip_gradient_l2']:.6f} | {d['largest_gradient']:.6f} | {d['largest_kl']:.6f} |\n"
s+='''
Both clipped every optimizer step in the last50 updates. Residual raw gradients were substantially larger, but endpoint policy changes stayed modest, no losses/checkpoint tensors became nonfinite, and the KL guard stopped further residual minibatches on10/400 updates. Frequent clipping is a monitored scale concern, not sufficient evidence of numerical divergence; low KL also does not mean the inherited feature concentration is fixed. The global gradient threshold was kept at0.5 throughout; it was not raised during the run.

### Fixed-probe feature health

Each model's512-observation probe was sampled from its own first training rollout, then held fixed. Probes are not held-out validation and differ between models, so within-model initial/final changes are more interpretable than direct architecture comparisons. BN running statistics were verified unchanged. Values below are activation mass shares and relative dormancy, not feature-importance proofs.

| Model | Layer | Update | Zero fraction | Relative dormant fraction | Top1 activity share | Top4 activity share |
| --- | --- | --- | --- | --- | --- | --- |
'''
for name,d in analysis.items():
 for phase in ['initial','final']:
  q=d['activation_probe'][phase]
  for layer in ['encoder.1','encoder.3','encoder.2.2.relu2','encoder.6']:
   if layer not in q['features']:continue
   v=q['features'][layer]
   s+=f"| {name} | {layer} | {q['update']} | {v['zero_fraction']:.4%} | {v['dormant_fraction']:.4%} | {v['top1_activity_share']:.4%} | {v['top4_activity_share']:.4%} |\n"
s+='''
Residual final-trunk top1 share remained about89% (89.60→89.16%); this run did not fix the inherited concentration. Dense activity became less sparse and less concentrated in both models, particularly the residual. The fullCNN second-trunk top4 share remained about79%. Those modest internal changes coexist with benchmark improvement and weak critics; no causal architecture/norm conclusion follows from this single paired run.

### Takeaways and next recommendations

1. The minimal two-current-policy PPO pipeline completed a meaningful run and both actors improved against the original baselines on a fresh trajectory sample. This is encouraging first evidence, not equilibrium convergence or general strength against an unseen opponent population. Keep the persistent latest states for continuation and preserve the independently selected bests already embedded in each file.
2. Improve evaluation reliability before judging small gains or selecting another long run: use more games and additional sampling seeds for checkpoint comparison, and eventually an additional opponent outside the two training lineages. The75/80% selection peaks substantially overstated this fresh sample's65/59% selected-best scores. Latest versus selected-best is unresolved here.
3. Prioritize a small rollout-size experiment (e.g.16 versus64 games/update at matched total game budget), holding architecture and optimizer recipe fixed. Its purpose is to test whether larger batches reduce noisy policy/value estimates; improvement is a hypothesis, not established by this run. More updates alone have not demonstrated a well-fitted critic. If value fit remains weak, isolate a critic-learning-rate/fit-budget ablation rather than changing several components together.
4. Do not enlarge either CNN or loosen clipping solely because gradients were clipped. There was no numerical failure, and policy KL stayed controlled. Inherited concentration and low critic explained variance remain open issues to measure; raw norm magnitude by itself does not identify a remedy.

No PSRO or historical-opponent scheme was introduced. Total primary training-metrics coverage is now **48 records**:46 previous records plus these two canonical sustained-PPO model logs. This is one paired experiment with6400 training games, not two independent6400-game datasets. The600 fresh evaluation games and3280 repeated fixed-seed evaluation games are evaluations, not additional training experience.
'''
p.write_text(s)
print('Verified checkpoints/source hashes, saved comparison chart, and appended stage36.')
