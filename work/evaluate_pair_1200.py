"""Repeat the established benchmark after value warmup and PPO continuation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json, time, math
import torch
from rl.policy import load_policy
from rl.opponents import OpponentPool
from rl.evaluation import evaluate
from training.diagnostics import write_json

torch.set_num_threads(1)
folders=[ROOT/'out/rl/cnn-c64-h256-8ebd1cf653',ROOT/'out/rl/residual-c32-h128-518d814dce']
for folder in folders:
 s=torch.load(folder/'policy.pt',map_location='cpu',weights_only=True)
 assert s['update']==1200 and s['training_log']['status']=='completed'
 assert any(p['after_ppo_update']==400 for p in s['training_log'].get('value_pretraining',[]))
pool=OpponentPool([ROOT/'out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt',
                   ROOT/'out/experiments/residual-medium-27890681/best_policy.pt'])
prior=json.loads((ROOT/'out/rl/paired-after-value-warmup-800-evaluation.json').read_text())
result=dict(seed=572940,games_per_opponent=50,max_plies=256,temperature=.9,
            reference_800='out/rl/paired-after-value-warmup-800-evaluation.json',
            description='Same benchmark seed/protocol as update800, not an untouched opponent or sampling-seed test.',policies={})
started=time.perf_counter()
for folder,stem in zip(folders,['full_cnn','residual']):
 for best,suffix in [(False,'ppo_latest'),(True,'ppo_best')]:
  name=stem+'_'+suffix;model,saved=load_policy(folder/'policy.pt',best=best)
  scores=evaluate(model,pool.static,games_per_opponent=50,temperature=.9,max_plies=256,seed=572940)
  n=sum(v['games'] for v in scores.values());w=sum(v['wins'] for v in scores.values());l=sum(v['losses'] for v in scores.values());d=n-w-l
  score=(w+.5*d)/n
  se=math.sqrt((w*(1-score)**2+d*(.5-score)**2+l*score**2)/(n*(n-1)))
  result['policies'][name]=dict(selected_update=saved.get('update'),games=n,wins=w,losses=l,draws=d,score=score,
                                score_standard_error=se,mean_reward=(w-l+.1*d)/n,opponents=scores,
                                previous_800_score=prior['policies'][name]['score'])
  result['wall_seconds']=time.perf_counter()-started
  write_json(ROOT/'out/rl/paired-lambda1-1200-evaluation.json',result)
  print(name,'update',saved.get('update'),f'W/L/D {w}/{l}/{d}',f'score {score:.1%}',flush=True)
