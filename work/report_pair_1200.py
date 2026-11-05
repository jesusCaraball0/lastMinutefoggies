import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fow-matplotlib')
from pathlib import Path
import json,hashlib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'out/rl/paired-lambda1-1200-analysis.json').read_text())
e=json.loads((ROOT/'out/rl/paired-lambda1-1200-evaluation.json').read_text());assert len(e['policies'])==4
out=ROOT/'out/experiments/paired-ppo-lambda1-updates801-1200'
verification={};logs={}
for name,x in a.items():
 folder=ROOT/'out/rl'/name;s=torch.load(folder/'policy.pt',map_location='cpu',weights_only=True)
 assert s['update']==1200 and s['training_log']['status']=='completed'
 assert list(folder.glob('*.pt'))==[folder/'policy.pt']
 assert all(torch.isfinite(t).all() for t in s['state_dict'].values())
 old=json.loads((out/(name+'-before.json')).read_text());log=s['training_log'];logs[name]=log
 assert log['updates'][:800]==old['updates']
 assert all(r['ppo']['gae_lambda']==1 for r in log['updates'][800:])
 source=Path(s['source_checkpoint']);source=source if source.is_absolute() else ROOT/source
 digest=hashlib.sha256(source.read_bytes()).hexdigest()
 assert digest==log['source_sha256']
 verification[name]=dict(update=1200,checkpoint_sha256=hashlib.sha256((folder/'policy.pt').read_bytes()).hexdigest(),source_sha256=digest,finite=True,history_preserved=True,recipe_changes=log['recipe_changes'])
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(12,8))
for name,log in logs.items():
 label='Full CNN' if name.startswith('cnn') else 'Residual'
 rows=log['updates'];indices=list(range(749,1200,10))
 for ax,fn,title in [(axes[0,0],lambda r:r['ppo']['value_fit']['monte_carlo_before']['rmse'],'Fresh-rollout MC RMSE'),(axes[0,1],lambda r:r['ppo']['value_fit']['monte_carlo_before']['explained_variance'],'Fresh-rollout MC explained variance'),(axes[1,0],lambda r:r['ppo']['endpoint_kl'],'Policy KL')]:
  ys=[sum(fn(r) for r in rows[max(0,k-9):k+1])/len(rows[max(0,k-9):k+1]) for k in indices]
  ax.plot([k+1 for k in indices],ys,label=label);ax.axvline(800,color='gray',linestyle='--');ax.set_title(title+' (10-update mean)');ax.legend();ax.set_xlabel('Update')
keys=list(e['policies']);axes[1,1].bar(range(4),[e['policies'][k]['score'] for k in keys]);axes[1,1].set_xticks(range(4));axes[1,1].set_xticklabels(['Full latest','Full best','Residual latest','Residual best'],rotation=15);axes[1,1].set_ylim(0,1);axes[1,1].set_title('Repeated fixed-opponent benchmark score')
fig.tight_layout();fig.savefig(ROOT/'out/rl/paired-lambda1-1200-comparison.png',dpi=160);plt.close(fig)
lines=['\n## Stage 43 — Paired PPO lambda1 continuation through1200 (2026-11-05; completed)\n',
'Both canonical policies completed400 additional updates (801–1200),6400 unique shared games,19200 cumulative unique training games. All other saved settings were preserved. The boundary at800 changes lambda0.95→1 and is recorded in both logs. No unchanged-lambda continuation control exists; actor gains cannot be attributed specifically to lambda.\n',
'### Fresh-rollout value quality and paired outcomes\n',
'RMSE below pools squared errors by decision count; EV is the mean of per-update EV. Predictions are measured before updating on the batch. Opponents evolve throughout, so these are changing data distributions, not fixed validation sets.\n',
'| Model | Window | W/L/D | Score | Mean reward | MC RMSE | MC EV |\n| --- | --- | --- | --- | --- | --- | --- |']
for name,x in a.items():
 for window in ['pre_change_last50','phase2_first10','phase2_first50','last50','phase2']:
  z=x[window];v=z['value_fit']['monte_carlo_before']
  lines.append(f"| {name} | {window} | {z['wins']}/{z['losses']}/{z['draws']} | {z['score']:.4%} | {z['mean_reward']:.6f} | {v['rmse']:.6f} | {v['mean_update_explained_variance']:.6f} |")
lines+=['\nHere phase2 means801–1200; pre_change_last50 means751–800; first10/50 means801–810/850; last50 means1151–1200.\n','### Repeated benchmark\n','Same seed572940, temperature0.9, max256 plies,50 games against each original frozen baseline (100 per candidate). This is the established benchmark, not an untouched opponent population or seed. Scores count draws0.5, training rewards count draws0.1.\n','| Candidate | Selected update | W/L/D | Score | Previous800 score | Standard error |\n| --- | --- | --- | --- | --- | --- |']
for name,x in e['policies'].items():lines.append(f"| {name} | {x['selected_update']} | {x['wins']}/{x['losses']}/{x['draws']} | {x['score']:.1%} | {x['previous_800_score']:.1%} | {x['score_standard_error']:.4f} |")
lines+=['\n### Detailed optimization and feature evidence\n']
for name,x in a.items():
 log=logs[name];old=json.loads((out/(name+'-before.json')).read_text());rows=log['updates'][800:]
 reasons={}
 for r in rows:
  for k,v in r['rollout']['reasons'].items():reasons[k]=reasons.get(k,0)+v
 evidence=dict(phase_mean_ppo=x['phase2']['ppo'],last50_ppo=x['last50']['ppo'],early_stop_updates=sum(r['ppo']['early_stop_kl'] for r in rows),termination_reasons=reasons,optimizer_steps=sum(r['ppo']['optimizer_steps'] for r in rows),decisions=sum(r['ppo']['transitions'] for r in rows),best_eval=x['best_eval'],final_eval=x['final_eval'],features800=old['activation_diagnostics'][-1]['features'],features1200=log['activation_diagnostics'][-1]['features'],verification=verification[name])
 lines+=['\n#### '+name+'\n','```json\n'+json.dumps(evidence,indent=2)+'\n```\n']
lines+=['\n### Interpretation and next recommendations\n',
'The value issue is not resolved. FullCNN late fresh-rollout EV decreased0.0420→0.0253 and RMSE increased0.8104→0.8376; residual EV increased slightly0.0107→0.0174 but RMSE increased0.8213→0.8305. Different return variance and evolving policies matter. The loss increase to roughly0.33 is not directly comparable with the previous roughly0.11: lambda1 trains on full discounted MC returns, a higher-variance target. It is not by itself evidence of divergence.\n',
'Both models still clip every optimizer step, yet late policy KL remains modest (~0.00393/full,0.00912/residual); tensors remain finite. Feature concentration persists: full second-conv top4 about79.7%, dense top4 about90.8%; residual final-trunk top1 about89.9%. Lambda1 did not repair concentrated representations or create a strong value predictor.\n',
'Actor strength should be judged using the benchmark table rather than paired reward alone. FullCNN won55.19% score within this training phase; residual44.81%. Both opponents change, so this is relative performance. The40-game in-loop best selector remains noisy; a selected-best checkpoint need not outperform the latest on the100-game repeated benchmark.\n',
'Keep lambda1 provisionally if continuing, but do not call it superior to0.95 from this uncontrolled continuation. Prioritize a larger multi-seed fixed-opponent evaluation before treating modest score differences or best/latest rankings as reliable. For further critic work, use more independent fresh games and short/early-stopped fitting; stage41 already showed overfitting with320 training games. Do not increase model size or value epochs solely to drive down training loss. A matched lambda comparison remains the clean way to attribute changes if needed. No further training is launched by this report.\n',
'Artifacts: out/rl/paired-lambda1-1200-analysis.json, paired-lambda1-1200-evaluation.json, paired-lambda1-1200-comparison.png; out/experiments/paired-ppo-lambda1-updates801-1200/verification.json and preserved800 logs. Checkpoints contain latest optimizer/RNG state and embedded best actors; original supervised source hashes verified unchanged. Older400/800 evidence is preserved.\n']
p=ROOT/'EXPERIMENT_HISTORY.md';assert '## Stage 43 —' not in p.read_text();p.write_text(p.read_text()+'\n'.join(lines)+'\n')
print('Saved stage43, verification and comparison plot')
