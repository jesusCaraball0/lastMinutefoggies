import json, hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rl.policy import load_policy
from training.probes import PolicyProbes

ROOT=Path.cwd()
B=ROOT/'out/experiments/boosted-cnn-28018186'
R=ROOT/'out/experiments/residual-medium-27890681'
OUT=B/'completed-run-analysis'
OUT.mkdir(exist_ok=True)
b=json.load(open(B/'metrics.json')); r=json.load(open(R/'metrics.json'))
splits=[json.load(open(p/'data_split.json')) for p in (B,R)]
equal={k:[Path(x).name for x in splits[0][k]]==[Path(x).name for x in splits[1][k]] for k in ('train','validation')}
keys=['optimizer','learning_rate','weight_decay','betas','epsilon','gradient_clip_global_l2','schedule','warmup_updates','total_optimizer_updates','epochs','batch_size','seed','train_games','validation_games','train_positions','validation_positions','probe_positions']
matching={k:b['config'][k]==r['config'][k] for k in keys}
def activation(s):
 vals=np.array([v['mean_absolute'] for v in s['per_channel']])
 return {k:s[k] for k in ('mean','std','zero_fraction','dormant_fraction','nonfinite_count')} | {'top1_activity_share':float(vals.max()/vals.sum()),'top4_activity_share':float(np.sort(vals)[-4:].sum()/vals.sum()),'largest_channel':int(vals.argmax()),'largest_channel_stats':s['per_channel'][int(vals.argmax())]}
def epoch(e):
 return {'epoch':e['epoch'],'lr':e['learning_rates'],'train_loss':e['train']['loss'],'validation':{k:e['validation'][k] for k in ('loss','top1','top5','positions')},'legal':e['validation']['policy_probes']['legal_moves'],'gradient_global':e['train']['gradient_global_l2'],'activations':{k:activation(e['validation']['activation_stats'][k]) for k in ('encoder.1','encoder.2.0.relu2','encoder.2.1.relu2','encoder.2.2.relu2','encoder.6')},'layer_gradients':e['train']['gradient_l2'],'update_ratios':e['train']['update_to_weight']}
result={'scope':'Completed 50-epoch boosted training run','boosted_status':b['status'],'completed_epochs':len(b['epochs']),'planned_epochs':b['config']['epochs'],'ordered_split_filenames_equal':equal,'matching_recipe':matching,'boosted_parameter_count':b['config']['parameter_count'],'residual_parameter_count':r['config']['parameter_count'],'boosted_best':epoch(b['epochs'][b['best_epoch']-1]),'boosted_latest':epoch(b['epochs'][-1]),'residual_best':epoch(r['epochs'][r['best_epoch']-1]),'residual_epoch48':epoch(r['epochs'][47]),'curves':{'boosted':[epoch(e) for e in b['epochs']],'residual':[epoch(e) for e in r['epochs']]}}
# Lightweight JSON first, retaining epoch diagnostics but not 2M per-step LR entries.
(OUT/'snapshot-analysis.json').write_text(json.dumps(result,indent=2))
print('Splits',equal,'recipe',all(matching.values()),flush=True)
print('gradient schema',result['boosted_latest']['gradient_global'],flush=True)
# Finite checkpoint verification and policy-column characterization.
torch.set_num_threads(2)
model,saved=load_policy(B/'best_policy.pt')
result['checkpoint_checks']={}
for name in ('best_policy.pt','last_policy.pt'):
 ck=saved if name=='best_policy.pt' else torch.load(B/name,map_location='cpu',weights_only=True)
 result['checkpoint_checks'][name]={'epoch':ck.get('epoch'),'finite':all(torch.isfinite(v).all().item() for v in ck['state_dict'].values() if v.is_floating_point()),'sha256':hashlib.sha256((B/name).read_bytes()).hexdigest()}
col=model.policy.weight[:,72].detach()
result['dominant_unit_policy_column']={'mean':col.mean().item(),'std_across_actions':col.std().item(),'other_columns_median_std':model.policy.weight.detach().std(0).median().item()}
paths=[ROOT/'game_data'/Path(x).name for x in splits[0]['validation']]
print('Building fixed 8192-position probe',flush=True)
probes=PolicyProbes(paths,8192,boosted=True)
print('Evaluating intact selected best',flush=True)
result['functional_probe']={'positions':probes.positions,'intact':probes.evaluate(model,'cpu')}
def remove_unit(module,inputs,output):
 value=output.clone(); value[:,72]=0; return value
handle=model.encoder[6].register_forward_hook(remove_unit)
print('Evaluating with dense channel72 zeroed (inference only)',flush=True)
try: result['functional_probe']['zero_dense_channel72']=probes.evaluate(model,'cpu')
finally: handle.remove()
(OUT/'snapshot-analysis.json').write_text(json.dumps(result,indent=2))
print('Functional probe',result['functional_probe'],flush=True)
# Derived plots; these are not the missing end-of-training histograms.
fig,axes=plt.subplots(2,3,figsize=(16,9))
for label,m in [('boosted',b),('residual',r)]:
 es=m['epochs']; x=[e['epoch'] for e in es]
 axes[0,0].plot(x,[e['validation']['loss'] for e in es],label=label+' validation')
 axes[0,0].plot(x,[e['train']['loss'] for e in es],ls='--',alpha=.5,label=label+' train')
 axes[0,1].plot(x,[100*e['validation']['top1'] for e in es],label=label+' full validation')
 axes[0,1].plot(x,[100*e['validation']['policy_probes']['legal_moves']['legal_top1'] for e in es],ls='--',label=label+' legal probe')
 for stat in ('median','p95'):
  axes[0,2].plot(x,[e['train']['gradient_global_l2']['pre_clip'][stat] for e in es],ls='-' if stat=='median' else '--',label=label+' '+stat)
 axes[1,0].plot(x,[e['train']['gradient_global_l2']['pre_clip']['max'] for e in es],label=label)
 for key,title,ax in [('encoder.2.2.relu2','Largest final-trunk channel activity share',axes[1,1]),('encoder.6','Largest dense channel activity share',axes[1,2])]:
  ax.plot(x,[100*activation(e['validation']['activation_stats'][key])['top1_activity_share'] for e in es],label=label)
for ax,title in zip(axes.flat,['Cross entropy','Move prediction accuracy (%)','Global gradient L2 before clipping','Maximum global gradient per epoch','Largest final-trunk channel activity share (%)','Largest dense channel activity share (%)']):
 ax.set_title(title);ax.set_xlabel('Epoch');ax.legend(fontsize=8);ax.grid(alpha=.2)
axes[0,2].axhline(5,color='gray',ls=':');axes[1,0].set_yscale('log')
fig.suptitle('Boosted CNN and residual baseline: completed 50 epochs each')
fig.tight_layout();fig.savefig(OUT/'training-comparison.png',dpi=150);plt.close(fig)
print('Saved',OUT,flush=True)
