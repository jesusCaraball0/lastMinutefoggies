"""Measured summaries for updates1201–2200, preserving older evidence."""
import json,math,hashlib,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def window(rows):
 total=sum(r['ppo']['transitions'] for r in rows);n=len(rows)
 games=[g for r in rows for g in r['rollout']['games']]
 w=sum(g['outcome']=='wins' for g in games);l=sum(g['outcome']=='losses' for g in games);d=len(games)-w-l
 reasons={}
 for g in games:reasons[g['reason']]=reasons.get(g['reason'],0)+1
 vf={}
 for mode in ['monte_carlo_before','monte_carlo_after','gae_before','gae_after']:
  vs=[r['ppo']['value_fit'][mode] for r in rows]
  mse=sum(v['mse']*v['samples'] for v in vs)/sum(v['samples'] for v in vs)
  vf[mode]=dict(rmse=math.sqrt(mse),mse=mse,mae=sum(v['mae']*v['samples'] for v in vs)/sum(v['samples'] for v in vs),mean_update_explained_variance=statistics.mean(v['explained_variance'] for v in vs if v['explained_variance'] is not None))
 keys=['policy_loss','value_loss','total_loss','endpoint_kl','endpoint_entropy','ratio_clip_fraction','pre_clip_gradient_l2']
 steps=sum(r['ppo']['optimizer_steps'] for r in rows)
 return dict(updates=n,games=len(games),wins=w,losses=l,draws=d,score=(w+.5*d)/len(games),reward=(w-l+.1*d)/len(games),mean_plies=sum(g['plies'] for g in games)/len(games),reasons=reasons,decisions=total,optimizer_steps=steps,ppo={k:sum(r['ppo'][k] for r in rows)/n for k in keys},value_fit=vf,gradient_clip_fraction=sum(r['ppo']['gradient_clipped_fraction']*r['ppo']['optimizer_steps'] for r in rows)/steps,gradient_max=max(r['ppo']['gradient_global_max'] for r in rows),kl_max=max(r['ppo']['endpoint_kl'] for r in rows),kl_early_stop_updates=sum(r['ppo']['early_stop_kl'] for r in rows),rollout_seconds=sum(r['rollout_seconds'] for r in rows),optimization_seconds=sum(r['optimization_seconds'] for r in rows))

def main():
 result={}
 for p in sorted((ROOT/'out/rl').glob('*/policy.pt')):
  s=torch.load(p,map_location='cpu',weights_only=True);log=s['training_log'];rows=log['updates']
  assert s['update']==2200 and log['status']=='completed' and len(rows)==2200
  assert list(p.parent.glob('*.pt'))==[p]
  assert all(torch.isfinite(v).all() for v in s['state_dict'].values())
  assert all(torch.isfinite(v).all() for v in s['best_policy']['state_dict'].values())
  assert all(r['ppo']['gae_lambda']==1 for r in rows[1200:])
  src=Path(s['source_checkpoint']);src=src if src.is_absolute() else ROOT/src
  digest=hashlib.sha256(src.read_bytes()).hexdigest();assert digest==log['source_sha256']
  old=json.loads((ROOT/'out/rl/paired-lambda1-1200-analysis.json').read_text())[p.parent.name]
  result[p.parent.name]=dict(completed=2200,status=log['status'],settings=s['settings'],checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original_source_sha256=digest,finite_latest_and_best=True,canonical_model_files=1,best_update=s['best_policy']['update'],wall_seconds=log['wall_seconds'],phase=window(rows[1200:]),previous50=window(rows[1150:1200]),first50=window(rows[1200:1250]),last50=window(rows[-50:]),last200=window(rows[-200:]),chunks100=[dict(end_update=1300+i*100,**window(rows[1200+i*100:1300+i*100])) for i in range(10)],evaluations=log['evaluations'],features1200=next(x for x in log['activation_diagnostics'] if x['update']==1200),features2200=log['activation_diagnostics'][-1],prior1200_best=old['best_eval'],recipe_changes=log.get('recipe_changes',[]))
 path=ROOT/'out/rl/paired-lambda1-2200-analysis.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 for name,x in result.items():print(name,'phase',x['phase']['wins'],x['phase']['losses'],x['phase']['draws'],'best',x['best_update'],'last50',x['last50']['value_fit']['monte_carlo_before'])
if __name__=='__main__':main()
