"""Preserve prior400 evidence and report the authorized401–800 continuation."""
import sys,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from work.analyze_pair_2200 import window
from work.run_boosted_outcome_ppo_800 import OUT
from work.run_boosted_outcome_ppo_400 import sha
from training.diagnostics import write_json

def main():
 meta=json.loads((OUT/'pipeline.json').read_text());assert meta['status']=='completed'
 path=Path(meta['canonical_checkpoint']);s=torch.load(path,map_location='cpu',weights_only=True)
 log=s['training_log'];rows=log['updates'];assert s['update']==800 and len(rows)==800 and log['status']=='completed'
 assert len(list(path.parent.glob('*.pt')))==1
 for state in [s['state_dict'],s['best_policy']['state_dict']]:
  assert all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
 for p,h in meta['source_hashes'].items():assert sha(p)==h,p
 for opp in meta['opponents']:assert sha(opp['path'])==opp['sha256']
 source=torch.load(meta['source'],map_location='cpu',weights_only=True)
 bn=[k for k in s['state_dict'] if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
 assert all(torch.equal(source['state_dict'][k],s['state_dict'][k]) for k in bn)
 for row in rows[400:]:
  assert row['ppo']['gae_lambda']==1 and len(row['rollout']['games'])==16
  for opp in meta['opponents']:
   games=[g for g in row['rollout']['games'] if g['opponent']==opp['path']]
   assert len(games)==8 and sum(g['color']=='white' for g in games)==4
 benches={k:json.loads((OUT/f).read_text()) for k,f in [('before','before-continuation-benchmark.json'),('latest','latest800-benchmark.json'),('best','selected-best-benchmark.json')]}
 prior=json.loads((OUT/'prior400-analysis.json').read_text())
 result=dict(status='completed',start_update=400,updates=800,best_update=s['best_policy']['update'],benchmarks=benches,prior400_benchmarks=prior['benchmarks'],pre400_last50=window(rows[350:400]),first10=window(rows[400:410]),first50=window(rows[400:450]),last50=window(rows[750:800]),whole=window(rows[400:800]),chunks50=[dict(end_update=i+50,**window(rows[i:i+50])) for i in range(400,800,50)],features_before=[x for x in log['activation_diagnostics'] if x['update']==400],features_after=log['activation_diagnostics'][-1],settings=s['settings'],checkpoint_sha256=sha(path),finite_latest_best=True,source_opponent_hashes_verified=True,BN_buffers_unchanged=True,one_canonical_rl_file=True,evaluations=log['evaluations'],pipeline_wall_seconds=meta['wall_seconds'])
 write_json(OUT/'analysis.json',result)
 section='\n<a id="stage-62"></a>\n\n## 62. Outcome-pretrained boosted PPO continuation completed: updates401–800\n\n'
 section+='400 additional updates,6400 fresh games, exact8/opponent/update and4/color, same frozen full latest2200/residual best1200. Restored latest400 optimizer/RNG; same recipe, no boosted opponent or PSRO. Earlier400 evidence preserved. Score=(W+0.5D)/N; reward win1/loss-1/draw0.1.\n\n|Actor|Update|W/L/D|Score|Reward|\n|---|---:|---:|---:|---:|\n'
 for label,z in [('Prior latest400',benches['before']),('Prior best',prior['benchmarks']['best']),('Latest800',benches['latest']),('Selected best',benches['best'])]:
  section+=f"|{label}|{z['selected_update']}|{z['wins']}/{z['losses']}/{z['draws']}|{z['score']:.2%}|{z['mean_reward']:.5f}|\n"
 section+='\nBenchmarks:100 games/actor,50/opponent,seed572940,temperature0.9,balanced colors; checkpoint selection uses a different40-game sample every20updates. Reused benchmark seed is not untouched after inspection.\n\n|Actor|Opponent|Score|MC RMSE|MC EV|\n|---|---|---:|---:|---:|\n'
 for label,z in benches.items():
  for opponent,g in z['opponents'].items():
   v=g['value_monte_carlo'];section+=f"|{label}|{Path(opponent).stem}|{g['score']:.2%}|{v['rmse']:.5f}|{v['explained_variance']:.5f}|\n"
 section+='\n|Window|Score|Reward|Critic loss|Fresh MC RMSE|Fresh mean EV|KL|Entropy|Gradient clip|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for key in ['pre400_last50','first10','first50','last50','whole']:
  w=result[key];v=w['value_fit']['monte_carlo_before'];p=w['ppo']
  section+=f"|{key}|{w['score']:.2%}|{w['reward']:.5f}|{p['value_loss']:.5f}|{v['rmse']:.5f}|{v['mean_update_explained_variance']:.5f}|{p['endpoint_kl']:.5f}|{p['endpoint_entropy']:.5f}|{w['gradient_clip_fraction']:.2%}|\n"
 section+='\nFresh critic metrics are collected before fitting each rollout; same-batch post-fit metrics in analysis.json are not held-out generalization. EV ignores additive calibration bias. Changes in actor trajectories also change critic target distributions. One seed and100-game benchmarks do not prove checkpoint superiority or continuation benefits. Finite latest/best, unchanged original source/opponents and BN buffers, exact opponent balance, and one canonical RL model verified. Full window loss/KL/clipping/outcome summaries and activation probes are retained in analysis.json. Review interpretation before final delivery; no training beyond800 was launched.\n'
 report=ROOT/'RL_EXPERIMENT_HISTORY.md';text=report.read_text()
 if '<a id="stage-62"></a>' not in text:report.write_text(text+section)
 print('Stage62 measured report complete; review analysis and add interpretation.',flush=True)
if __name__=='__main__':main()
