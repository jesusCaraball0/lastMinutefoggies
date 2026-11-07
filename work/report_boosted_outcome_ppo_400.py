"""Measured analysis for the completed balanced boosted PPO experiment only."""
import sys,json,hashlib,collections,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from work.analyze_pair_2200 import window
from training.diagnostics import write_json
OUT=ROOT/'out/experiments/boosted-outcome-cnn-ppo-400-balanced-frozen-opponents'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pct(x):return f'{100*x:.2f}%'
def num(x):return 'unavailable' if x is None else f'{x:.5f}'

def main():
 meta=json.loads((OUT/'pipeline.json').read_text());assert meta['status']=='completed'
 path=Path(meta['canonical_checkpoint']);s=torch.load(path,map_location='cpu',weights_only=True);log=s['training_log'];rows=log['updates']
 assert s['update']==400 and log['status']=='completed' and len(rows)==400
 assert len(list(path.parent.glob('*.pt')))==1
 assert all(torch.isfinite(v).all().item() for v in s['state_dict'].values() if v.is_floating_point())
 assert all(torch.isfinite(v).all().item() for v in s['best_policy']['state_dict'].values() if v.is_floating_point())
 for p,h in meta['source_hashes'].items():assert sha(p)==h,'Source changed: '+p
 for p in meta['opponents']:assert sha(p['path'])==p['sha256']
 assert log['source_sha256']==meta['source_hashes'][meta['source']]
 assert meta['pretrained_value_preserved_at_PPO_start']
 assert 'preserved outcome-supervised' in log['critic_initialization']
 for row in rows:
  assert row['ppo']['gae_lambda']==1 and len(row['rollout']['games'])==16
  for opp in meta['opponents']:
   games=[g for g in row['rollout']['games'] if g['opponent']==opp['path']]
   assert len(games)==8 and sum(g['color']=='white' for g in games)==4
 outcomes={}
 for opp in meta['opponents']:
  games=[g for row in rows for g in row['rollout']['games'] if g['opponent']==opp['path']]
  from rl.metrics import game_summary
  outcomes[opp['label']]=game_summary(games)
 benches={label:json.loads((OUT/file).read_text()) for label,file in [('before','before-ppo-benchmark.json'),('latest','latest400-benchmark.json'),('best','selected-best-benchmark.json')]}
 initial=torch.load(meta['source'],map_location='cpu',weights_only=True)
 bnkeys=[k for k in initial['state_dict'] if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]
 bn_unchanged=all(torch.equal(initial['state_dict'][k],s['state_dict'][k]) for k in bnkeys)
 actor_delta=math.sqrt(sum(float((s['state_dict'][k]-v).float().square().sum()) for k,v in initial['state_dict'].items() if k.startswith(('encoder.','policy.')) and v.is_floating_point()))
 prior=json.loads((ROOT/'out/experiments/boosted-cnn-ppo-400-balanced-frozen-opponents/analysis.json').read_text())
 result=dict(pretrained_value_preserved=True,previous_move_only_boosted=dict(benchmarks=prior['benchmarks'],first50=prior['first50'],last50=prior['last50']),status='completed',updates=400,best_update=s['best_policy']['update'],settings=s['settings'],checkpoint_sha256=sha(path),source_hashes_verified=True,frozen_opponents_verified=True,finite_latest_best=True,one_canonical_rl_file=True,BN_buffers_unchanged=bn_unchanged,actor_l2_change=actor_delta,
             whole=window(rows),first10=window(rows[:10]),first50=window(rows[:50]),last50=window(rows[-50:]),chunks50=[dict(end_update=(i+1)*50,**window(rows[i*50:(i+1)*50])) for i in range(8)],training_by_opponent=outcomes,evaluations=log['evaluations'],benchmarks=benches,initial_features=log['activation_diagnostics'][0],final_features=log['activation_diagnostics'][-1],wall_seconds=log['wall_seconds'],pipeline_wall_seconds=meta['wall_seconds'])
 write_json(OUT/'analysis.json',result)
 def row(label,z):
  w=z['whole'] if 'whole' in z else z
  return f"|{label}|{w['wins']}/{w['losses']}/{w['draws']}|{pct(w['score'])}|{num(w['reward'])}|{num(w['ppo']['value_loss'])}|{num(w['value_fit']['monte_carlo_before']['rmse'])}|{num(w['value_fit']['monte_carlo_before']['mean_update_explained_variance'])}|{num(w['ppo']['endpoint_kl'])}|{num(w['ppo']['endpoint_entropy'])}|{pct(w['gradient_clip_fraction'])}|"
 section='''\n<a id="stage-60"></a>\n\n## 60. Completed outcome-pretrained boosted PPO400 against an exactly balanced frozen mixture (2026-11-06)\n\n'''
 section+=f"Completed400 PPO **updates** (each up to4 minibatch passes),6400 fresh training games,3200 against each frozen opponent. Each update contains8 games/opponent,4 per color. Total learner decisions={result['whole']['decisions']}; actual optimizer minibatch steps={result['whole']['optimizer_steps']}. Best selected update={result['best_update']}; latest=400. Only boosted updates; full/residual actors stay frozen. Canonical file: `{path.relative_to(ROOT)}`. Training wall time={log['wall_seconds']/3600:.3f}hours; total pipeline including benchmarks={meta['wall_seconds']/3600:.3f}hours.\n\n"
 section+='### Recipe and opponent identity\n\n'
 section+='Actor starts from jointly outcome-supervised boosted imitation epoch43. The pretrained value head is preserved, not zero initialized; there is no extra head-only warmup. Historical pretraining targets are undiscounted outcomes, while PPO targets discounted current-policy returns. Opponents are full CNN latest2200 and residual saved-best1200 exported as weights-only immutable evaluation/training snapshots. These are stronger RL incumbents, not the old supervised baselines. Thus absolute scores are not interchangeable with stages36–45 benchmarks against supervised actors. Adam actor LR1e-5,value-head LR1e-3,epsilon1e-5;4PPO passes,minibatch128,ratio clip.2,value coefficient.5,entropy coefficient.01,gradient clip.5,target KL.02,temperature.9,gamma.99,lambda1,256-ply cap,seed7980,CPU1thread. Dropout off/BNbuffers fixed for both collection and optimization; encoder and policy weights trainable. Outcomes win+1/loss−1/draw+.1; score=(W+.5D)/N. This is one training seed, not a controlled feature ablation or PSRO.\n\n'
 section+='### Independent before/after benchmark\n\n100games/policy,50/opponent,balanced colors,seed572940,temperature.9,max256plies. Same protocol before/after, separate from the40-game checkpoint selector (seed9137,every20updates). The pre-PPO actor retains its learned historical outcome head. Its fresh-game value scores directly test transfer to discounted current-policy outcomes. Repeated benchmark is not an untouched seed once examined.\n\n|Actor|Update|W/L/D|Score|Mean reward|\n|---|---:|---:|---:|---:|\n'
 for key,label in [('before','Imitation starting actor'),('latest','Latest PPO'),('best','Selected PPO best')]:
  z=benches[key];section+=f"|{label}|{z['selected_update']}|{z['wins']}/{z['losses']}/{z['draws']}|{pct(z['score'])}|{num(z['mean_reward'])}|\n"
 section+='\n|Actor|Frozen opponent|W/L/D|Score|MC value RMSE|MC value EV|Mean plies|\n|---|---|---:|---:|---:|---:|---:|\n'
 for key,label in [('before','Imitation'),('latest','Latest400'),('best','Selected best')]:
  for name,z in benches[key]['opponents'].items():
   v=z['value_monte_carlo'];section+=f"|{label}|{Path(name).stem}|{z['wins']}/{z['losses']}/{z['draws']}|{pct(z['score'])}|{num(v['rmse'])}|{num(v['explained_variance'])}|{z['mean_plies']:.2f}|\n"
 section+='\n### On-policy learning and critic diagnostics\n\nMC RMSE is decision-weighted pooled squared error against full discounted outcomes; EV below is the mean of per-update EVs, not a pooled cross-update EV. Fresh means rollout predictions before fitting that batch. Same-batch after-update results are in analysis.json and must not be presented as held-out generalization. Lambda1 GAE targets equal discounted MC returns up to rounding.\n\n|Window|W/L/D|Score|Reward|Critic loss|Fresh MC RMSE|Fresh mean-update EV|Mean endpoint KL|Mean entropy|Gradient clipped|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
 for label,key in [('All400','whole'),('First10','first10'),('First50','first50'),('Last50','last50')]:section+=row(label,result[key])+'\n'
 section+='\nTraining outcomes by opponent:\n\n|Opponent|Games|W/L/D|Score|Mean reward|Mean plies|\n|---|---:|---:|---:|---:|---:|---:|\n'
 for label,z in outcomes.items():section+=f"|{label}|{z['game_count']}|{z['wins']}/{z['losses']}/{z['draws']}|{pct(z['score'])}|{num(z['mean_reward'])}|{z['mean_plies']:.2f}|\n"
 w=result['whole'];section+=f"\nMaximum endpoint KL={num(w['kl_max'])}; KL early-stop updates={w['kl_early_stop_updates']}; maximum pre-clipping global gradient={num(w['gradient_max'])}. Mean PPO ratio-clipped fraction={pct(w['ppo']['ratio_clip_fraction'])}; mean actor loss={num(w['ppo']['policy_loss'])};mean total loss={num(w['ppo']['total_loss'])}. Termination reason totals: `{json.dumps(w['reasons'],sort_keys=True)}`. Games finish through the canonical referee; any protocol/bot failure aborts rather than becoming a training win.\n\n"
 section+='### Representation diagnostics and integrity\n\nProbe sampled from first training rollout and held fixed; this is not the imitation validation probe or held-out rollout set.\n\n|Layer|Initial zero fraction|Final zero fraction|Initial largest activity share|Final largest activity share|\n|---|---:|---:|---:|---:|\n'
 for layer in ['encoder.1','encoder.2.2.relu2','encoder.6']:
  a=result['initial_features']['features'][layer];b=result['final_features']['features'][layer]
  section+=f"|{layer}|{pct(a['zero_fraction'])}|{pct(b['zero_fraction'])}|{pct(a['top1_activity_share'])}|{pct(b['top1_activity_share'])}|\n"
 section+=f"\nFinite latest/best tensors verified; original supervised source and both original full/residual canonical RL files match launch SHA256s. Frozen opponent snapshots also match. One canonical boosted RL model file with embedded best, no duplicate latest/best files. BN running buffers unchanged={bn_unchanged}; actor/encoder state L2 change={actor_delta:.6f}. Initial source hash recorded in canonical training log matches launch source.\n\n"
 delta=benches['latest']['score']-benches['before']['score'];bd=benches['best']['score']-benches['before']['score']
 section+='### Comparison with the prior move-only boosted PPO experiment\n\n'
 section+='Both runs use the same frozen actor identities,100-game benchmark seed and400-update recipe. This compares joint-pretraining versus move-only starting models;it does not isolate value-head initialization from altered encoder/policy features. Outcomes within games are correlated and one training seed does not measure run-to-run variation.\n\n|Model|Before score|Latest400 score|Selected-best score|Last50 fresh MC EV|Last50 fresh MC RMSE|\n|---|---:|---:|---:|---:|---:|\n'
 for label,ben,last in [('Prior move-only boosted',prior['benchmarks'],prior['last50']),('Joint outcome-pretrained boosted',benches,result['last50'])]:
  section+=f"|{label}|{pct(ben['before']['score'])}|{pct(ben['latest']['score'])}|{pct(ben['best']['score'])}|{num(last['value_fit']['monte_carlo_before']['mean_update_explained_variance'])}|{num(last['value_fit']['monte_carlo_before']['rmse'])}|\n"
 section+='\n### Measured takeaways and next recommendations\n\n'
 section+=f"Latest benchmark score changes by{100*delta:+.2f}percentage points and selected-best benchmark by{100*bd:+.2f}points versus the starting imitation actor. These are observed100-game changes, not proof of statistical significance or a history-specific PPO effect. The previous move-only boosted experiment used this same400-update recipe/opponent identities/seed but a different imitation-trained actor and initially zero critic. It is a single-seed training-objective comparison, not an isolated value-head-initialization control. No unchanged-training control was run. The best selector maximizes a different40-game sample, so selected-best need not win the100-game benchmark. No benchmark result automatically promotes or overwrites a checkpoint.\n\n"
 if delta>=.10:section+='The latest actor shows a substantial descriptive gain against the same frozen incumbents; verify it across additional seeds before committing a longer continuation. '
 elif delta>0:section+='The latest actor shows a modest descriptive gain; evaluation uncertainty remains large enough that repeated-seed incumbent comparisons should come before a long continuation. '
 else:section+='The latest actor does not improve this benchmark; more updates alone are not supported by this result. Compare saved-best against the starting actor across multiple seeds and inspect critic transfer/policy health before continuing. '
 if result['last50']['value_fit']['monte_carlo_before']['mean_update_explained_variance']<.1:section+='Fresh critic EV remains low in the last50updates; lower same-batch fitting error does not establish a useful generalizing critic. '
 section+='Interpret gradient clipping alongside actual KL, score and finite tensors; do not infer collapse from clipping frequency or activity concentration alone. Keep opponent mixture fixed for any immediate diagnostic continuation, preserve the independently evaluated incumbent, and defer PSRO until useful complementary play is established. No additional training beyond400 is authorized by this report.\n\n'
 section+='Artifacts: `out/experiments/boosted-outcome-cnn-ppo-400-balanced-frozen-opponents/{pipeline.json,analysis.json,before-ppo-benchmark.json,latest400-benchmark.json,selected-best-benchmark.json}`. Canonical metrics/plots in the boosted out/rl directory include rewards,outcomes,losses,value_quality,policy_health,throughput,training,evaluation_opponents; per-checkpoint activation/weight diagnostics in diagnostics/. Master logs: `logs/boosted-outcome-ppo-400-balanced.out/.err`. Analysis retains50-update windows, all selector evaluations, per-opponent results, source verification and features.\n'
 report=ROOT/'RL_EXPERIMENT_HISTORY.md';text=report.read_text()
 if '<a id="stage-60"></a>' not in text:
  text=text.replace('stage numbers30–59','stage numbers30–60');text+=section;report.write_text(text)
 print(json.dumps({k:{z:benches[k][z] for z in ['wins','losses','draws','score','selected_update']} for k in benches},indent=2))
 print('Measured stage60 appended (review interpretation before final user delivery).')
if __name__=='__main__':main()
