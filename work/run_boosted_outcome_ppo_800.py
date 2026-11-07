"""Resume the authorized fixed-opponent outcome PPO experiment, updates401–800."""
import sys,json,time,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from work import run_boosted_outcome_ppo_400 as base
from training.diagnostics import write_json
from rl.train import parse_args,training_steps
from rl.storage import training_lock,require_completed_pair
from rl.opponents import OpponentPool
OLD=base.OUT
OUT=ROOT/'out/experiments/boosted-outcome-cnn-ppo-updates401-800-balanced-frozen-opponents'

def main():
 torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
 with training_lock(OUT):
  original=json.loads((OLD/'pipeline.json').read_text());canonical=Path(original['canonical_checkpoint'])
  manifest=OUT/'pipeline.json'
  if not manifest.exists():
   saved=torch.load(canonical,map_location='cpu',weights_only=True)
   assert saved['update']==400
   assert original['status']=='completed'
   # Preserve complete prior metrics and benchmark evidence; no duplicate RL model file.
   write_json(OUT/'update400-metrics.json',saved['training_log'])
   shutil.copy2(OLD/'analysis.json',OUT/'prior400-analysis.json')
   shutil.copy2(OLD/'latest400-benchmark.json',OUT/'before-continuation-benchmark.json')
   original.update(status='setup',start_update=400,target_updates=800,started_unix=time.time(),starting_checkpoint_sha256=base.sha(canonical),starting_best_update=saved['best_policy']['update'])
   write_json(manifest,original)
  meta=json.loads(manifest.read_text())
  for p,h in meta['source_hashes'].items():assert base.sha(p)==h,p
  for opp in meta['opponents']:assert base.sha(opp['path'])==opp['sha256']
  current=torch.load(canonical,map_location='cpu',weights_only=True)['update']
  if current<800:
   assert 400<=current<800
   a=parse_args(['--resume',str(canonical),'--updates','800','--device','cpu','--threads','1'])
   assert a.gae_lambda==1 and a.games_per_update==16
   meta['status']='PPO training';write_json(manifest,meta)
   with training_lock(canonical.parent):
    require_completed_pair(canonical.parent)
    pool=OpponentPool([Path(p['path']) for p in meta['opponents']]);steps=training_steps(a)
    try:
     request=next(steps)
     while True:
      start=time.perf_counter();batch,reports=base.collect_balanced(request,pool.static)
      for opp in meta['opponents']:
       games=[g for g in reports if g['opponent']==opp['path']]
       assert len(games)==8 and sum(g['color']=='white' for g in games)==4
      request=steps.send((batch,reports,time.perf_counter()-start))
    except StopIteration:pass
    except BaseException as error:
     try:steps.throw(error)
     except (StopIteration,GeneratorExit):pass
     raise
    finally:steps.close()
  assert torch.load(canonical,map_location='cpu',weights_only=True)['update']==800
  meta['status']='after benchmarks';write_json(manifest,meta)
  # benchmark() intentionally uses the original immutable opponent paths.
  for best,label in [(False,'latest800'),(True,'selected-best')]:
   dest=OUT/(label+'-benchmark.json')
   if not dest.exists():
    print('Starting benchmark:',label,flush=True)
    result=base.benchmark(canonical,best=best);write_json(dest,result)
    print(label,result['wins'],result['losses'],result['draws'],result['score'],flush=True)
  for p,h in meta['source_hashes'].items():assert base.sha(p)==h,p
  for opp in meta['opponents']:assert base.sha(opp['path'])==opp['sha256']
  meta.update(status='completed',finished_unix=time.time(),wall_seconds=time.time()-meta['started_unix']);write_json(manifest,meta)
  subprocess.run([sys.executable,'work/report_boosted_outcome_ppo_800.py'],cwd=ROOT,check=True)
if __name__=='__main__':main()
