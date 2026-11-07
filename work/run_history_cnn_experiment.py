"""Matched imitation pilot -> frozen-opponent PPO -> multi-seed cross-play.

Reuses training/train.py, rl/train.py and referee-backed tournament. No game rules here.
"""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fow-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/fow-cache')
import sys,json,time,subprocess,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from training.datasets import ReplayDataset
from training.diagnostics import write_json
from rl.storage import variant_name
from rl.policy import load_policy
from rl.opponents import OpponentPool
from rl.evaluation import evaluate
from work.checkpoint_round_robin import run as tournament
OUT=ROOT/'out/experiments/history4-cnn-imitation-then-ppo-pilot'
PYTHON=ROOT/'.venv/bin/python'
SOURCES=[ROOT/'out/rl/cnn-c64-h256-8ebd1cf653/policy.pt',ROOT/'out/rl/residual-c32-h128-518d814dce/policy.pt']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def execute(args,name):
 with (ROOT/'logs'/(name+'.out')).open('a') as stdout,(ROOT/'logs'/(name+'.err')).open('a') as stderr:
  print('Starting',name,flush=True)
  subprocess.run([str(PYTHON),'-u',*map(str,args)],cwd=ROOT,env=os.environ,stdout=stdout,stderr=stderr,check=True)

def main():
 OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
 manifest=OUT/'pipeline.json'
 if manifest.exists():m=json.loads(manifest.read_text())
 else:
  m=dict(status='setup',started_unix=time.time(),seed=7980,history_frames=4,epochs=8,ppo_target_updates=400,
   note='Local controlled pilot, not full-data training. Current-only imitation control; PPO only history model. Existing full/residual actors frozen. No PSRO.',
   original_checkpoint_hashes={str(p):sha(p) for p in SOURCES})
  split=json.loads((ROOT/'out/experiments/residual-reduced-head-ablations-2560games-80k/data_split.json').read_text())
  assert len(split['train'])==2560 and len(split['validation'])==64
  assert not set(split['train'])&set(split['validation'])
  write_json(OUT/'split.json',split)
  counts=ReplayDataset([Path(p) for p in split['train']]);warmup=3*math.ceil(counts.positions/128)
  m.update(train_games=2560,validation_games=64,train_positions=counts.positions,warmup_updates=warmup,
   model=dict(channels=32,hidden_size=128,batch_norm=True,residual_blocks=3,dropout=.1),
   imitation_recipe=dict(optimizer='AdamW',start_lr=1e-4,peak_lr=3e-4,min_lr=1e-5,betas=[.9,.99],eps=1e-8,weight_decay=.01,clip=5,batch=128,selection='legal_accuracy'))
  opponent_folder=OUT/'frozen-opponents';opponent_folder.mkdir()
  for p,label,best in zip(SOURCES,['full-latest2200','residual-best1200'],[False,True]):
   s=torch.load(p,map_location='cpu',weights_only=True);assert s['update']==2200
   state=s['best_policy'] if best else {k:s[k] for k in ['kind','activation','negative_slope','model_config','state_dict','inference_temperature','algorithm','update'] if k in s}
   torch.save(state,opponent_folder/(label+'.pt'))
  m['opponents']=[str(opponent_folder/(label+'.pt')) for label in ['full-latest2200','residual-best1200']]
  m['opponent_hashes']={p:sha(Path(p)) for p in m['opponents']}
  write_json(manifest,m)
 for frames,label in [(1,'current-control'),(4,'history4')]:
  run=OUT/label;metrics=run/'metrics.json';saved=None
  if metrics.exists():
   d=json.loads(metrics.read_text())
   if d['status']=='completed':continue
   if (run/'last_policy.pt').exists():saved=run/'last_policy.pt'
   else:raise RuntimeError('Incomplete imitation setup without resumable checkpoint: '+str(run))
  m.update(status='imitation',current_case=label);write_json(manifest,m)
  args=['training/train.py','--run-dir',run,'--device','cpu','--threads','2']
  if saved:args+=['--resume',saved]
  else:args+=['--split-manifest',OUT/'split.json','--channels','32','--hidden-size','128','--residual-blocks','3','--dropout','.1','--batch-norm','--history-frames',frames,'--epochs','8','--batch-size','128','--learning-rate','1e-4','--peak-learning-rate','3e-4','--min-learning-rate','1e-5','--warmup-updates',m['warmup_updates'],'--beta-1','.9','--beta-2','.99','--weight-decay','.01','--epsilon','1e-8','--grad-clip-norm','5','--selection-metric','legal_accuracy','--probe-max-positions','2048','--seed','7980']
  execute(args,'history-pilot-'+label+'-imitation')
 control=OUT/'current-control/best_policy.pt';history=OUT/'history4/best_policy.pt'
 # Independent pre-PPO evaluation files are completed once; continuation never overwrites them.
 before=OUT/'imitation-arena.json'
 if not before.exists():
  m.update(status='pre_ppo_evaluation');write_json(manifest,m)
  result={}
  opponents=OpponentPool([Path(p) for p in m['opponents']])
  for label,path in [('current_control',control),('history4',history)]:
   model,_=load_policy(path)
   result[label]={str(seed):evaluate(model,opponents.static,games_per_opponent=50,seed=seed) for seed in [743291,743292]}
  write_json(before,result)
 init=torch.load(history,map_location='cpu',weights_only=True);variant=variant_name(init);run=ROOT/'out/rl'/variant
 m.update(status='ppo',rl_variant=variant,rl_dir=str(run));write_json(manifest,m)
 completed=0
 if (run/'policy.pt').exists():
  s=torch.load(run/'policy.pt',map_location='cpu',weights_only=True);completed=s['update']
  if s['training_log']['source_sha256']!=sha(history):raise RuntimeError('History canonical checkpoint belongs to a different imitation source')
 if completed<400:
  args=['rl/train.py','--init-checkpoint',history,'--updates','400','--games-per-update','16','--gae-lambda','1','--device','cpu','--threads','1','--eval-every','20']
  for p in m['opponents']:args+=['--opponent',p,'--eval-opponent',p]
  execute(args,'history-pilot-ppo')
 m.update(status='post_ppo_evaluation');write_json(manifest,m)
 entries=[['History4 imitation',str(history.relative_to(ROOT)),False],['Current-only imitation control',str(control.relative_to(ROOT)),False],['History4 PPO latest400',str((run/'policy.pt').relative_to(ROOT)),False],['History4 PPO best',str((run/'policy.pt').relative_to(ROOT)),True],['Full CNN RL frozen2200',str(Path(m['opponents'][0]).relative_to(ROOT)),False],['Residual RL frozen1200',str(Path(m['opponents'][1]).relative_to(ROOT)),False]]
 roster=OUT/'tournament-roster.json';write_json(roster,entries)
 results=OUT/'tournament/results.json'
 if not results.exists() or json.loads(results.read_text())['status']!='completed':tournament(entries,OUT/'tournament')
 assert all(sha(Path(p))==h for p,h in m['opponent_hashes'].items())
 # Original canonical policies are never written by this pipeline; check external modifications explicitly.
 m.update(status='completed',wall_seconds=time.time()-m['started_unix'],original_checkpoint_hashes_after={str(p):sha(p) for p in SOURCES})
 write_json(manifest,m);print('PIPELINE COMPLETED',OUT,flush=True)
if __name__=='__main__':main()
