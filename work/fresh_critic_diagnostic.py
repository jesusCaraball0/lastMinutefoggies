"""Frozen update800 game-level critic comparison; never writes canonical policies."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fow-matplotlib')
import sys, json, time, random, hashlib, copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rl.policy import load_policy
from rl.rollout import collect_pair
from rl.ppo import value_quality
from training.datasets import observation_planes
from training.diagnostics import write_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'out/experiments/fresh-update800-critic-head-vs-independent'
OUT.mkdir(parents=True,exist_ok=True)
PATHS=[ROOT/'out/rl'/v/'policy.pt' for v in ['cnn-c64-h256-8ebd1cf653','residual-c32-h128-518d814dce']]
NAMES=['full_cnn','residual']
SEED=831705
GAMES=512
EPOCHS=12
BATCH=256
torch.set_num_threads(2)
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes=[sha(p) for p in PATHS]
models=[]
for p in PATHS:
 m,s=load_policy(p);assert s['update']==800
 models.append(m)
started=time.time()
result=dict(status='collecting',seed=SEED,games=GAMES,epochs=EPOCHS,batch_size=BATCH,
 source_hashes=hashes,source_paths=[str(p) for p in PATHS],temperature=.9,gamma=.99,
 protocol='Frozen actors; identical game split; train320/validation96/test96; select epoch by validation MSE; BN buffers frozen, dropout off; head Adam1e-3 vs independent encoder+head Adam1e-4, eps1e-5, clip0.5; no policy updates.',models={})
write_json(OUT/'results.json',result)
cache=OUT/'fresh_games.npz'
if not cache.exists():
 rows=[dict(tokens=[],targets=[],game=[],remaining=[]) for _ in models]
 reports=[]
 for g in range(0,GAMES,2):
  pairs=collect_pair(models,NAMES,2,max_plies=256)
  reports.extend(pairs[0][1])
  for i,(batch,reps) in enumerate(pairs):
   # Restart points in discounted returns identify games only ambiguously; use
   # recorded plies and alternating colors to recover exact decision lengths.
   lengths=[(r['plies']+1)//2 if r['color']=='white' else r['plies']//2 for r in reps]
   assert sum(lengths)==len(batch)
   rows[i]['tokens'].append(batch.tokens.numpy()); rows[i]['targets'].append(batch.discounted_returns.numpy())
   rows[i]['game'].append(np.repeat([g,g+1],lengths))
   rows[i]['remaining'].append(np.concatenate([np.arange(n-1,-1,-1) for n in lengths]))
  if (g+2)%32==0: print('Collected',g+2,'/',GAMES,'games',round(time.time()-started,1),'s',flush=True)
 arrays={f'{i}_{k}':np.concatenate(v) for i,row in enumerate(rows) for k,v in row.items()}
 np.savez_compressed(cache,**arrays)
 write_json(OUT/'game_reports.json',reports)
data=np.load(cache)
# Split separately within colors; shared IDs across both model datasets.
rng=np.random.default_rng(SEED+1)
ids=[rng.permutation(np.arange(c,GAMES,2)) for c in (0,1)]
splits={key:np.concatenate([x[a:b] for x in ids]) for key,a,b in [('train',0,160),('validation',160,208),('test',208,256)]}
assert not set(splits['train'])&set(splits['test'])
write_json(OUT/'split.json',{k:v.tolist() for k,v in splits.items()})

@torch.no_grad()
def predict(enc,head,tokens):
 return torch.cat([head(enc(observation_planes(t,'cpu'))).tanh().flatten() for t in tokens.split(BATCH)])

def score(pred,y,gids,remaining):
 out=value_quality(pred,y)
 out['by_remaining_decisions']={}
 for label,mask in [('last_6',remaining<=5),('6_to_20', (remaining>5)&(remaining<=20)),('over_20',remaining>20)]:
  if mask.any():out['by_remaining_decisions'][label]=value_quality(pred[mask],y[mask])
 out['games']=len(torch.unique(gids));return out

for i,name in enumerate(NAMES):
 tokens=torch.from_numpy(data[f'{i}_tokens']); targets=torch.from_numpy(data[f'{i}_targets'])
 games=torch.from_numpy(data[f'{i}_game']);remaining=torch.from_numpy(data[f'{i}_remaining'])
 ix={k:torch.from_numpy(np.flatnonzero(np.isin(games.numpy(),v))) for k,v in splits.items()}
 model=models[i]
 baseline=predict(model.encoder,model.value,tokens)
 constant=float(targets[ix['train']].mean())
 entry=dict(positions={k:len(v) for k,v in ix.items()},constant=constant,arms={})
 result['models'][name]=entry
 preds={}
 for arm in ['unchanged','constant','head_only','independent_critic']:
  if arm in ['unchanged','constant']:
   prediction=baseline if arm=='unchanged' else torch.full_like(targets,constant)
   history=[];best_epoch=0
  else:
   torch.manual_seed(SEED+2)
   enc=copy.deepcopy(model.encoder).eval();head=copy.deepcopy(model.value).eval()
   for p in enc.parameters():p.requires_grad_(arm=='independent_critic')
   params=list(head.parameters())+(list(enc.parameters()) if arm=='independent_critic' else [])
   opt=torch.optim.Adam(params,lr=1e-3 if arm=='head_only' else 1e-4,eps=1e-5)
   buffers={k:v.clone() for k,v in enc.named_buffers()}
   initial_params={k:v.clone() for k,v in enc.state_dict().items()}
   history=[];best=float('inf');best_state=None;best_epoch=0
   # Cache features only for the frozen-encoder arm.
   with torch.no_grad():
    features=torch.cat([enc(observation_planes(t,'cpu')) for t in tokens.split(BATCH)]) if arm=='head_only' else None
   for epoch in range(1,EPOCHS+1):
    norms=[]
    for inds in ix['train'][torch.randperm(len(ix['train']))].split(BATCH):
     feat=features[inds] if features is not None else enc(observation_planes(tokens[inds],'cpu'))
     pred=head(feat).tanh().flatten();loss=.5*(pred-targets[inds]).square().mean()
     opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,.5,error_if_nonfinite=True);opt.step();norms.append(float(norm))
    with torch.no_grad():
     def subset_predict(key):
      inds=ix[key]
      return head(features[inds]).tanh().flatten() if features is not None else predict(enc,head,tokens[inds])
     train=value_quality(subset_predict('train'),targets[ix['train']])
     val=value_quality(subset_predict('validation'),targets[ix['validation']])
    row=dict(epoch=epoch,train=train,validation=val,gradient_l2_mean=float(np.mean(norms)),gradient_l2_max=max(norms),clip_fraction=float(np.mean(np.array(norms)>.5)))
    history.append(row)
    if val['mse']<best:
     best=val['mse'];best_epoch=epoch;best_state=(copy.deepcopy(enc.state_dict()),copy.deepcopy(head.state_dict()))
    print(name,arm,epoch,'train RMSE',round(train['rmse'],4),'val',round(val['rmse'],4),'EV',round(val['explained_variance'],4),flush=True)
    entry['running']=dict(arm=arm,history=history);result['status']='fitting';write_json(OUT/'results.json',result)
   assert all(torch.equal(v,buffers[k]) for k,v in enc.named_buffers())
   if arm=='head_only':assert all(torch.equal(v,initial_params[k]) for k,v in enc.state_dict().items())
   enc.load_state_dict(best_state[0]);head.load_state_dict(best_state[1]);prediction=predict(enc,head,tokens)
  preds[arm]=prediction
  entry['arms'][arm]=dict(selected_epoch=best_epoch,history=history,**{k:score(prediction[v],targets[v],games[v],remaining[v]) for k,v in ix.items()})
  write_json(OUT/'results.json',result)
 # Paired game cluster bootstrap preserves within-game correlations. Position-weighted MSE.
 test=ix['test']; gids=games[test].unique();rng=np.random.default_rng(SEED+3)
 comparisons={}
 for reference in ['head_only','unchanged','constant']:
  diffs=(preds['independent_critic'][test]-targets[test]).square()-(preds[reference][test]-targets[test]).square()
  sums=np.array([float(diffs[games[test]==g].sum()) for g in gids]);counts=np.array([int((games[test]==g).sum()) for g in gids])
  draws=rng.integers(0,len(gids),(3000,len(gids)))
  boot=sums[draws].sum(1)/counts[draws].sum(1)
  comparisons[reference]=dict(mse_difference=float(diffs.mean()),game_bootstrap_95_ci=np.quantile(boot,[.025,.975]).tolist())
 entry['independent_minus_reference_test']=comparisons
 entry.pop('running',None)
 np.savez_compressed(OUT/(name+'-predictions.npz'),targets=targets.numpy(),games=games.numpy(),remaining=remaining.numpy(),**{k:v.numpy() for k,v in preds.items()})
 result['status']='fitting';write_json(OUT/'results.json',result)
assert hashes==[sha(p) for p in PATHS]
result.update(status='completed',canonical_checkpoints_unchanged=True,wall_seconds=time.time()-started)
write_json(OUT/'results.json',result)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(12,8))
for i,name in enumerate(NAMES):
 for arm,color in [('head_only','tab:blue'),('independent_critic','tab:orange')]:
  h=result['models'][name]['arms'][arm]['history']
  for j,metric in enumerate(['rmse','explained_variance']):
   axes[i,j].plot([r['epoch'] for r in h],[r['validation'][metric] for r in h],label=arm,color=color)
   axes[i,j].set_title(name+' validation '+metric);axes[i,j].set_xlabel('Epoch');axes[i,j].legend()
fig.tight_layout();fig.savefig(OUT/'validation-curves.png',dpi=160);plt.close(fig)
print('COMPLETED',result['wall_seconds'],flush=True)
