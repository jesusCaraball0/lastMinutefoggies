"""Frozen checkpoint round robin using the existing referee and policy agents."""
import sys,json,time,hashlib,itertools,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rl.policy import load_policy,PolicyAgent
from rl.game import play_game
from rl.rollout import outcome
from collections import Counter
from training.diagnostics import write_json
ROOT=Path(__file__).resolve().parents[1]
def run(entries,out):
 torch.set_num_threads(1);out.mkdir(parents=True,exist_ok=True)
 models={};roster={};hashes={}
 for label,path,best in entries:
  m,s=load_policy(ROOT/path,best=best)
  h=hashlib.sha256()
  for k,v in sorted(m.state_dict().items()):
   if not k.startswith('value.'):
    h.update(k.encode());h.update(v.numpy().tobytes())
  digest=h.hexdigest()
  if digest in hashes:
   roster[hashes[digest]]['aliases'].append(label);continue
  hashes[digest]=label;models[label]=m
  roster[label]=dict(path=path,best=best,update=s.get('update'),actor_sha256=digest,aliases=[])
 result=dict(status='running',temperature=.9,max_plies=256,games_per_pair=100,seeds=[743291,743292,743293,743294,743295],roster=roster,matches=[])
 started=time.time()
 for a,b in itertools.combinations(models,2):
  counts=Counter(wins=0,losses=0,draws=0);colors={c:Counter(wins=0,losses=0,draws=0) for c in ['white','black']};reasons=Counter();lengths=[];seed_scores=[]
  for seed in result['seeds']:
   torch.manual_seed(seed);random.seed(seed);scores=[]
   for g in range(20):
    white=g%2==0
    fa=lambda:PolicyAgent(models[a],.9);fb=lambda:PolicyAgent(models[b],.9)
    replay=play_game(fa if white else fb,fb if white else fa,max_plies=256,white_name=a if white else b,black_name=b if white else a)
    if replay.get('failedSide') or replay['result']=='*':raise RuntimeError(str(replay))
    r=outcome(replay['result'],white);counts[r]+=1;colors['white' if white else 'black'][r]+=1
    reasons[replay['reason']]+=1;lengths.append(len(replay['frames'])-1);scores.append({'wins':1,'losses':0,'draws':.5}[r])
   seed_scores.append(dict(seed=seed,score=sum(scores)/len(scores)))
  row=dict(a=a,b=b,**counts,score=(counts['wins']+.5*counts['draws'])/100,by_color={k:dict(v) for k,v in colors.items()},reasons=dict(reasons),mean_plies=sum(lengths)/len(lengths),seed_scores=seed_scores)
  result['matches'].append(row);result['wall_seconds']=time.time()-started;write_json(out/'results.json',result)
  print(a,'vs',b,dict(counts),'score',row['score'],flush=True)
 standings={k:dict(wins=0,losses=0,draws=0,games=0) for k in models}
 for r in result['matches']:
  for name,reverse in [(r['a'],False),(r['b'],True)]:
   x=standings[name];x['wins']+=r['losses' if reverse else 'wins'];x['losses']+=r['wins' if reverse else 'losses'];x['draws']+=r['draws'];x['games']+=100
 for x in standings.values():x['score']=(x['wins']+.5*x['draws'])/x['games']
 result.update(status='completed',standings=standings);write_json(out/'results.json',result)
 return result
if __name__=='__main__':
 entries=json.loads(Path(sys.argv[1]).read_text());run(entries,Path(sys.argv[2]))
