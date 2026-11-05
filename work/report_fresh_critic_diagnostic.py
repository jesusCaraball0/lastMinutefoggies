import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fow-matplotlib')
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'out/experiments/fresh-update800-critic-head-vs-independent'
d=json.loads((out/'results.json').read_text());assert d['status']=='completed' and d['canonical_checkpoints_unchanged']
fig,axes=plt.subplots(2,2,figsize=(12,8))
for i,(name,m) in enumerate(d['models'].items()):
 for arm,color in [('head_only','tab:blue'),('independent_critic','tab:orange')]:
  h=m['arms'][arm]['history']
  for j,metric in enumerate(['rmse','explained_variance']):
   for split,style in [('train','--'),('validation','-')]:
    axes[i,j].plot([r['epoch'] for r in h],[r[split][metric] for r in h],style,label=arm+' '+split,color=color)
   axes[i,j].set_title(name+' '+metric);axes[i,j].set_xlabel('Epoch');axes[i,j].legend(fontsize=8)
fig.tight_layout();fig.savefig(out/'train-validation-curves.png',dpi=160);plt.close(fig)
lines=['\n## Stage 41 — Frozen update800 fresh-game critic diagnostic (2026-11-05; completed)\n',
'Executed the first experiment proposed in stage40. This is supervised critic regression with frozen game-generating policies, not a PPO continuation or an actor-strength experiment. Both canonical RL checkpoint file hashes were unchanged before/after. No fitted experimental weights were installed or persisted as additional RL checkpoints.\n',
'Protocol: 512 unique games from the latest update800 fullCNN/residual pair, alternating colors, temperature0.9, referee max256 plies, seed831705. Both player trajectories come from the same games and use their own existing fog observations. Targets are actual discounted terminal outcomes (+1/-1/draw0.1, gamma0.99 per own decision), not bootstrapped GAE. FullCNN outcomes273W225L14D. All 512 games and both players\' return arrays were independently checked against the recorded results and remaining-decision counts.\n',
'Game-level split:320 train/96 validation/96 untouched test, balanced by model color, shared game IDs across models; disjoint game IDs verified. FullCNN positions11014/3598/3392; residual11030/3607/3392. Total18004/18029 positions respectively. Related positions within a game are correlated, so game counts are the more relevant independent sample size.\n',
'For each model compare: unchanged update800 critic; constant prediction equal to training-set mean return; head-only training with frozen encoder at Adam LR1e-3; independently trainable copy of encoder+value head at Adam LR1e-4. Both fitting arms start from identical update800 weights, use the same minibatch shuffle seed, batch256, Adam eps1e-5/default betas, no weight decay, half-MSE, global clip0.5, 12 epochs. BN running buffers remain frozen and dropout stays off; independent critic BN affine parameters are trainable. Buffer equality and head-only encoder equality verified. Validation MSE selects among epochs1–12; test data are excluded from fitting/selection. This compares two practical fitting recipes, not a pure architecture ablation: their LRs differ and there is one collection/split/training seed.\n',
'| Model | Arm | Selected epoch | Train RMSE | Validation RMSE | Test RMSE | Test explained variance |\n| --- | --- | --- | --- | --- | --- | --- |']
for name,m in d['models'].items():
 for arm,a in m['arms'].items():
  lines.append(f"| {name} | {arm} | {a['selected_epoch']} | {a['train']['rmse']:.6f} | {a['validation']['rmse']:.6f} | {a['test']['rmse']:.6f} | {a['test']['explained_variance']:.6f} |")
lines+=['\n### Training trajectories and generalization\n']
for name,m in d['models'].items():
 for arm in ['head_only','independent_critic']:
  a=m['arms'][arm];h=a['history'];r=h[-1]
  lines.append(f"- {name}, {arm}: epoch1 train/validation RMSE {h[0]['train']['rmse']:.6f}/{h[0]['validation']['rmse']:.6f}; epoch12 {r['train']['rmse']:.6f}/{r['validation']['rmse']:.6f}. Epoch12 train/validation EV {r['train']['explained_variance']:.6f}/{r['validation']['explained_variance']:.6f}; clipping fraction {r['clip_fraction']:.4%}, mean/max raw gradient norm {r['gradient_l2_mean']:.6f}/{r['gradient_l2_max']:.6f}.")
lines+=['\nIndependent critics show a clear train/validation divergence on this small fixed dataset. Best epochs are2 fullCNN and1 residual. Better training fit does not produce a demonstrated test improvement over the original critic. This weakens the claim that simply freeing the encoder and adding value epochs is sufficient. It does not disprove separate critics in online PPO with continually refreshed data.\n',
'### Paired uncertainty\n',
'Paired game-cluster bootstrap,3000 resamples of the96 test games, position-weighted MSE difference (independent minus reference; negative favors independent). These intervals condition on this training run, do not include training-seed variability, and are not adjusted for multiple comparisons.\n',
'| Model | Reference | MSE difference | 95% interval |\n| --- | --- | --- | --- |']
for name,m in d['models'].items():
 for ref,x in m['independent_minus_reference_test'].items():
  lo,hi=x['game_bootstrap_95_ci'];lines.append(f"| {name} | {ref} | {x['mse_difference']:.6f} | [{lo:.6f}, {hi:.6f}] |")
lines+=['\nBoth independent critics beat the training-mean constant on this test split, but their intervals versus unchanged and head-only critics include zero. FullCNN test results numerically favor the unchanged critic; residual unchanged and independent are almost tied. Validation/test ranking reversals emphasize the uncertainty from96 games per held-out split.\n',
'### Distance-to-terminal breakdown\n',
'Remaining own decisions are used only for retrospective analysis, never as model input. "last6" includes remaining counts0–5; other bins6–20 and>20. Discounting reduces target magnitudes far from termination, so lower RMSE in distant positions does not imply better discrimination.\n',
'| Model | Arm | Remaining decisions | Test positions | RMSE | Explained variance |\n| --- | --- | --- | --- | --- | --- |']
for name,m in d['models'].items():
 for arm,a in m['arms'].items():
  for stage,x in a['test']['by_remaining_decisions'].items():
   lines.append(f"| {name} | {arm} | {stage} | {x['samples']} | {x['rmse']:.6f} | {x['explained_variance']:.6f} |")
lines+=['\nPredictive signal is stronger near game termination: unchanged full/residual EV about0.192/0.096 in the last6 decisions, versus approximately zero >20 decisions away. Independent fits reach0.224/0.121 near termination but have no established overall benefit. This pattern is consistent with long-horizon outcome uncertainty/credit-assignment difficulty and limited data; it is not proof of an irreducible information ceiling or a diagnosis of hidden-state insufficiency.\n',
'### Takeaways and next recommendations\n',
'1. Keep the current actors/checkpoints. This experiment did not train actors and cannot establish a win-rate improvement. Neither more fixed-data critic epochs nor replacing the canonical critic is justified by the observed test results.\n',
'2. If pursuing critic fitting, prioritize more fresh independent games rather than replaying320 games for additional epochs. A useful follow-up is4–8x the game count with the same frozen actors, recipe and an untouched enlarged test set, training briefly/early-stopping. A data-scaling improvement would support a sample-size explanation; failure would motivate revisiting representation/targets. This is proposed, not launched.\n',
'3. The smaller algorithmic next test remains a matched PPO continuation with lambda1 versus0.95, starting from identical update800 states. Complete-game lambda1 uses observed discounted returns and reduces reliance on intermediate value predictions, at higher variance. Keep other knobs equal, judge fixed-opponent playing strength with sufficient evaluation games, and monitor fresh value prediction rather than only fitting loss. Do not combine it immediately with critic architecture changes. This is proposed, not launched.\n',
'4. Longer-term separate-critic optimization remains a hypothesis supported by literature but not validated here. The fixed-data experiment omits actor-gradient interference by freezing the actor in both arms, so it cannot measure the benefit of removing such interference during PPO.\n',
f"Wall time {d['wall_seconds']:.2f}s (~2.1min). Artifacts: out/experiments/fresh-update800-critic-head-vs-independent/results.json, fresh_games.npz, split.json, game_reports.json, per-model prediction arrays, validation-curves.png and train-validation-curves.png. Reproduction: work/fresh_critic_diagnostic.py; report generation: work/report_fresh_critic_diagnostic.py. Logs: logs/fresh-critic-diagnostic.out/.err (error log empty). Existing canonical checkpoints unchanged. No background task remains.\n"]
p=root/'EXPERIMENT_HISTORY.md';old=p.read_text();assert '## Stage 41 —' not in old
p.write_text(old+'\n'.join(lines)+'\n')
print('Appended stage41 and plotted training/validation curves')
