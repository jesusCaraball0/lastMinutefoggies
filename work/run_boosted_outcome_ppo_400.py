"""Outcome-pretrained boosted PPO, exact 50/50 frozen-opponent collection, canonical trainer lifecycle."""
import sys, json, hashlib, random, time, dataclasses
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rl.train import parse_args, training_steps, policy_weights
from rl.storage import training_lock, require_completed_pair, variant_name
from rl.rollout import collect, Batch
from rl.policy import load_policy, initialize_value
from rl.opponents import OpponentPool
from rl.evaluation import evaluate
from training.diagnostics import write_json

OUT=ROOT/'out/experiments/boosted-outcome-cnn-ppo-400-balanced-frozen-opponents'
SOURCE=ROOT/'out/experiments/boosted-outcome-cnn-28094904/best_policy.pt'
SOURCES=[ROOT/'out/rl/cnn-c64-h256-8ebd1cf653/policy.pt',ROOT/'out/rl/residual-c32-h128-518d814dce/policy.pt']
LABELS=['full-latest2200','residual-best1200']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

from work.run_boosted_ppo_400 import collect_balanced

def benchmark(path,best=False):
    model,saved=load_policy(path,best=best)
    pool=OpponentPool([OUT/'frozen-opponents'/(label+'.pt') for label in LABELS])
    scores=evaluate(model,pool.static,games_per_opponent=50,temperature=.9,max_plies=256,seed=572940,gamma=.99)
    n=sum(v['games'] for v in scores.values());w=sum(v['wins'] for v in scores.values());l=sum(v['losses'] for v in scores.values());d=n-w-l
    return dict(selected_update=saved.get('update',0),source_epoch=saved.get('epoch'),seed=572940,games=n,wins=w,losses=l,draws=d,score=(w+.5*d)/n,mean_reward=(w-l+.1*d)/n,opponents=scores,critic_zero_initialized=False,critic_source="joint historical outcome pretraining")

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    with training_lock(OUT):
        manifest=OUT/'pipeline.json'
        if manifest.exists():meta=json.loads(manifest.read_text())
        else:
            source=torch.load(SOURCE,map_location='cpu',weights_only=True)
            assert source['epoch']==43 and source['model_config']['boosted'] and source['model_config']['outcome_supervision'] and source['value_pretrained']
            canonical=ROOT/'out/rl'/variant_name(source)/'policy.pt'
            if canonical.exists():raise RuntimeError('Unexpected existing boosted run; inspect before choosing a continuation')
            (OUT/'frozen-opponents').mkdir(exist_ok=True)
            meta=dict(status='setup',started_unix=time.time(),source=str(SOURCE),canonical_checkpoint=str(canonical),target_updates=400,
                      collection='exact 8 games/opponent/update, 4 white/4 black per opponent, shuffled opponent color-pairs',
                      source_hashes={str(p):sha(p) for p in [SOURCE,*SOURCES]},opponents=[])
            for p,label,best,expected in zip(SOURCES,LABELS,[False,True],[2200,1200]):
                s=torch.load(p,map_location='cpu',weights_only=True);assert s['update']==2200
                weights=s['best_policy'] if best else policy_weights(s)
                assert weights['update']==expected
                path=OUT/'frozen-opponents'/(label+'.pt');torch.save(weights,path)
                meta['opponents'].append(dict(label=label,path=str(path),update=expected,sha256=sha(path)))
            files=['rl/train.py','rl/ppo.py','rl/rollout.py','rl/policy.py','rl/evaluation.py','starters/shared/boosted_features.py','work/run_boosted_ppo_400.py','work/run_boosted_outcome_ppo_400.py']
            meta['implementation_hashes']={f:sha(ROOT/f) for f in files}
            write_json(manifest,meta)
        for p,h in meta['source_hashes'].items():assert sha(p)==h, 'Source changed: '+p
        for p in meta['opponents']:assert sha(p['path'])==p['sha256']
        canonical=Path(meta['canonical_checkpoint'])
        if not (OUT/'before-ppo-benchmark.json').exists():
            meta['status']='before benchmark';write_json(manifest,meta)
            print('Starting 100-game pre-PPO benchmark against the two frozen RL incumbents',flush=True)
            before=benchmark(SOURCE);write_json(OUT/'before-ppo-benchmark.json',before)
            print('Pre-PPO:',before['wins'],before['losses'],before['draws'],'score',before['score'],flush=True)
        completed=0
        if canonical.exists():
            saved=torch.load(canonical,map_location='cpu',weights_only=True);completed=saved['update']
        if completed<400:
            args=['--init-checkpoint',str(SOURCE),'--updates','400','--games-per-update','16','--gae-lambda','1','--device','cpu','--threads','1','--eval-every','20','--eval-games','20','--checkpoint-every','20']
            for p in meta['opponents']:args+=['--opponent',p['path'],'--eval-opponent',p['path']]
            a=parse_args(args)
            meta.update(status='PPO training',settings=vars(a).copy())
            meta['settings']={k:[str(p) for p in v] if isinstance(v,list) else str(v) if isinstance(v,Path) else v for k,v in meta['settings'].items()}
            write_json(manifest,meta)
            with training_lock(canonical.parent):
                require_completed_pair(canonical.parent)
                pool=OpponentPool([Path(p['path']) for p in meta['opponents']])
                steps=training_steps(a)
                try:
                    request=next(steps)
                    if completed==0:
                        initial=torch.load(SOURCE,map_location='cpu',weights_only=True)
                        assert all(torch.equal(v,initial['state_dict']['value.'+k]) for k,v in request['model'].value.state_dict().items())
                        meta['pretrained_value_preserved_at_PPO_start']=True
                        write_json(manifest,meta)
                    while True:
                        start=time.perf_counter();batch,reports=collect_balanced(request,pool.static)
                        for opponent in meta['opponents']:
                            games=[g for g in reports if g['opponent']==opponent['path']]
                            assert len(games)==8 and sum(g['color']=='white' for g in games)==4
                        request=steps.send((batch,reports,time.perf_counter()-start))
                except StopIteration:pass
                except BaseException as error:
                    try:steps.throw(error)
                    except (StopIteration,GeneratorExit):pass
                    raise
                finally:steps.close()
        meta['status']='after benchmarks';write_json(manifest,meta)
        for best,label in [(False,'latest400'),(True,'selected-best')]:
            path=OUT/(label+'-benchmark.json')
            if path.exists():continue
            print('Starting post-PPO benchmark:',label,flush=True)
            result=benchmark(canonical,best=best);write_json(path,result)
            print(label,result['wins'],result['losses'],result['draws'],'score',result['score'],flush=True)
        # Only the canonical boosted RL file was updated; originals stay immutable.
        for p,h in meta['source_hashes'].items():assert sha(p)==h,'Source changed: '+p
        meta.update(status='completed',finished_unix=time.time(),wall_seconds=time.time()-meta['started_unix'])
        write_json(manifest,meta)
        import subprocess
        subprocess.run([sys.executable,'work/report_boosted_outcome_ppo_400.py'],cwd=ROOT,check=True)
        print('Experiment, benchmarks and report complete:',OUT,flush=True)
if __name__=='__main__':main()
