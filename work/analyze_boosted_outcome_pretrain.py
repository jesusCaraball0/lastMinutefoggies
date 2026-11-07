"""Completed medium joint-pretraining comparison; no arena or optimizer steps."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
NEW=ROOT/'out/experiments/boosted-outcome-cnn-28094904'
OLD=ROOT/'out/experiments/boosted-cnn-28018186'
OUT=NEW/'analysis';OUT.mkdir(exist_ok=True)
a=json.loads((NEW/'metrics.json').read_text());b=json.loads((OLD/'metrics.json').read_text())
assert a['status']=='completed' and len(a['epochs'])==50
splits=[json.loads((p/'data_split.json').read_text()) for p in [NEW,OLD]]
equal={k:[Path(p).name for p in splits[0][k]]==[Path(p).name for p in splits[1][k]] for k in ['train','validation']}
assert all(equal.values())
recipe=['parameter_count','learning_rate','weight_decay','betas','epsilon','gradient_clip_global_l2','schedule','warmup_updates','total_optimizer_updates','epochs','batch_size','seed','train_positions','validation_positions','probe_positions']
assert all(a['config'][k]==b['config'][k] for k in recipe)
def summarize(epoch):
 v=epoch['validation'];t=epoch['train'];acts={}
 for name,stats in v['activation_stats'].items():
  values=sorted([x['mean_absolute'] for x in stats.get('per_channel',[])],reverse=True);total=sum(values)
  acts[name]={k:stats[k] for k in ['mean','std','zero_fraction','nonfinite_count','dormant_fraction'] if k in stats}
  acts[name].update(top1_share=values[0]/total if total else None,top4_share=sum(values[:4])/total if total else None)
 return dict(epoch=epoch['epoch'],train={k:t[k] for k in ['loss','top1','top5','total_loss','outcome'] if k in t},validation={k:v[k] for k in ['loss','top1','top5','total_loss','outcome'] if k in v},legal=v['policy_probes']['legal_moves'],gradients=t['gradient_global_l2'],layer_gradients=t['gradient_l2'],actual_update_ratios=t['update_to_weight'],activations=acts)
result=dict(run=NEW.name,status=a['status'],best_epoch=a['best_epoch'],baseline_best_epoch=b['best_epoch'],ordered_split_filenames_equal=equal,matched_recipe_keys=recipe,config=a['config'],wall_seconds=a['wall_seconds'],baseline_wall_seconds=b['wall_seconds'],best=summarize(a['epochs'][a['best_epoch']-1]),final=summarize(a['epochs'][-1]),baseline_best=summarize(b['epochs'][b['best_epoch']-1]),baseline_epoch43=summarize(b['epochs'][42]),curves={'joint':[summarize(e) for e in a['epochs']],'move_only':[summarize(e) for e in b['epochs']]},best_outcome_epoch=min(a['epochs'],key=lambda e:e['validation']['outcome']['mse'])['epoch'])
result['checkpoint_checks']={}
torch.set_num_threads(1)
for name,expected in [('best_policy.pt',43),('last_policy.pt',50),('final_policy.pt',50)]:
 s=torch.load(NEW/name,map_location='cpu',weights_only=True)
 assert s['epoch']==expected and s['value_pretrained'] and s['model_config']['outcome_supervision']
 assert all(torch.isfinite(v).all().item() for v in s['state_dict'].values() if v.is_floating_point())
 result['checkpoint_checks'][name]=dict(epoch=s['epoch'],finite=True,value_pretrained=s['value_pretrained'],sha256=hashlib.sha256((NEW/name).read_bytes()).hexdigest())
write=lambda:(OUT/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
write()
fig,axs=plt.subplots(3,3,figsize=(16,12))
for label,m in [('Joint',a),('Move-only',b)]:
 es=m['epochs'];xs=[e['epoch'] for e in es]
 axs[0,0].plot(xs,[e['validation']['loss'] for e in es],label=label)
 axs[0,1].plot(xs,[100*e['validation']['top1'] for e in es],label=label+' full raw')
 axs[0,1].plot(xs,[100*e['validation']['policy_probes']['legal_moves']['legal_top1'] for e in es],ls='--',alpha=.6,label=label+' legal probe')
 compact=result['curves']['joint' if label=='Joint' else 'move_only']
 axs[1,1].plot(xs,[100*e['activations']['encoder.6']['top1_share'] for e in compact],label=label)
 axs[1,2].plot(xs,[e['activations']['encoder.6']['std'] for e in compact],label=label)
 for key,style in [('median','-'),('p95','--')]:
  axs[2,0].plot(xs,[e['train']['gradient_global_l2']['pre_clip'][key] for e in es],ls=style,label=label+' '+key)
 axs[2,1].plot(xs,[e['train']['gradient_global_l2']['pre_clip']['max'] for e in es],label=label)
xs=list(range(1,51))
for split in ['train','validation']:
 es=[e[split]['outcome'] for e in a['epochs']]
 axs[0,2].plot(xs,[e['rmse'] for e in es],label=split)
 axs[1,0].plot(xs,[e['explained_variance'] for e in es],label=split)
 axs[2,2].plot(xs,[100*e['win_loss_accuracy'] for e in es],label=split)
axs[0,2].axhline(a['epochs'][-1]['validation']['outcome']['training_mean_constant_mse']**.5,ls=':',color='gray',label='Train-mean constant, validation')
axs[2,1].set_yscale('log')
for ax,title in zip(axs.flat,['Full-validation move CE','Move accuracy (%)','Historical outcome RMSE','Historical outcome EV','Largest dense-unit activity share (%)','Dense activation standard deviation','Global pre-clip gradient L2','Largest global gradient per epoch','Historical win/loss accuracy (%)']):
 ax.set(title=title,xlabel='Epoch');ax.legend(fontsize=7);ax.grid(alpha=.2)
fig.suptitle('Same 32/128 boosted architecture and data: joint outcome supervision vs move-only')
fig.tight_layout();fig.savefig(OUT/'joint-vs-move-only.png',dpi=140);plt.close(fig)
print('Verified matched split/recipe and three finite checkpoints; wrote',OUT,flush=True)
