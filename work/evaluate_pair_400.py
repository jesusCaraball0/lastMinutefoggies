"""Fresh-seed benchmark of selected PPO policies versus their initial sources."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import json, math, time
import torch
from rl.evaluation import evaluate
from rl.opponents import OpponentPool
from rl.policy import load_policy, initialize_value
from training.diagnostics import write_json

torch.set_num_threads(1)
items=[('full_cnn_initial',ROOT/'out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt',False),
       ('residual_initial',ROOT/'out/experiments/residual-medium-27890681/best_policy.pt',False),
       ('full_cnn_ppo_best',ROOT/'out/rl/cnn-c64-h256-8ebd1cf653/policy.pt',True),
       ('residual_ppo_best',ROOT/'out/rl/residual-c32-h128-518d814dce/policy.pt',True),
       ('full_cnn_ppo_latest',ROOT/'out/rl/cnn-c64-h256-8ebd1cf653/policy.pt',False),
       ('residual_ppo_latest',ROOT/'out/rl/residual-c32-h128-518d814dce/policy.pt',False)]
for _,path,best in items:
 if 'out/rl' in str(path):
  saved=torch.load(path,map_location='cpu',weights_only=True)
  assert saved['update']==400 and saved['training_log']['status']=='completed','Training must complete first'
pool=OpponentPool([items[0][1],items[1][1]],device='cpu')
output=ROOT/'out/rl/paired-400-fresh-evaluation.json'
result=dict(seed=572940,games_per_opponent=50,max_plies=256,temperature=.9,
            description='Fresh sampling seed, same frozen opponent pool; not a held-out opponent population.',policies={})
started=time.perf_counter()
for name,path,best in items:
 model,saved=load_policy(path,best=best)
 if saved.get('algorithm')!='ppo':initialize_value(model,saved)
 opponents=evaluate(model,pool.static,games_per_opponent=50,temperature=.9,max_plies=256,seed=572940)
 games=sum(r['games'] for r in opponents.values())
 wins=sum(r['wins'] for r in opponents.values());losses=sum(r['losses'] for r in opponents.values());draws=games-wins-losses
 score=(wins+.5*draws)/games
 # Descriptive sampling SE; opponent strata/common seeds do not imply independent performance estimates.
 mean_se=math.sqrt((wins*(1-score)**2+draws*(.5-score)**2+losses*score**2)/(games*(games-1)))
 result['policies'][name]=dict(source=str(path),selected_update=saved.get('update'),games=games,wins=wins,losses=losses,draws=draws,
                              score=score,score_standard_error=mean_se,mean_reward=(wins-losses+.1*draws)/games,opponents=opponents)
 result['wall_seconds']=time.perf_counter()-started
 write_json(output,result)
 print(name,'update',saved.get('update'),f'W/L/D {wins}/{losses}/{draws}',f'score {score:.1%}',flush=True)
print(output,flush=True)
