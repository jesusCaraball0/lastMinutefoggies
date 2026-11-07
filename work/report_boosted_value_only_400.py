"""Review fresh-data value-only experiment without promoting actors or running more games."""
import sys,json,math,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from training.diagnostics import write_json
from work.run_boosted_value_only_400 import OUT,CANONICAL,PHASE,tensor_hash,sha

def summarize(rows):
    result=dict(rounds=len(rows),games=sum(r['rollout']['game_count'] for r in rows),positions=sum(r['target_samples'] for r in rows),optimizer_steps=sum(r['fit']['optimizer_steps'] for r in rows))
    for label in ['before','after']:
        n=sum(r[label]['samples'] for r in rows)
        mse=sum(r[label]['mse']*r[label]['samples'] for r in rows)/n
        result[label]=dict(mse=mse,rmse=math.sqrt(mse),mae=sum(r[label]['mae']*r[label]['samples'] for r in rows)/n,mean_round_EV=statistics.mean(r[label]['explained_variance'] for r in rows if r[label]['explained_variance'] is not None))
    steps=sum(r['fit']['optimizer_steps'] for r in rows)
    result.update(loss=statistics.mean(r['fit']['loss'] for r in rows),gradient_mean=statistics.mean(r['fit']['gradient_mean'] for r in rows),gradient_max=max(r['fit']['gradient_max'] for r in rows),clipped_fraction=sum(r['fit']['gradient_clipped_fraction']*r['fit']['optimizer_steps'] for r in rows)/steps,training_score=statistics.mean(r['rollout']['score'] for r in rows),training_reward=statistics.mean(r['rollout']['mean_reward'] for r in rows))
    return result

def main():
    meta=json.loads((OUT/'pipeline.json').read_text());report=json.loads((OUT/'metrics.json').read_text())
    assert meta['status']==report['status']=='completed' and len(report['updates'])==400
    saved=torch.load(CANONICAL,map_location='cpu',weights_only=True)
    assert saved['update']==400 and saved['value_head_phases'][PHASE]['completed_steps']==400
    assert tensor_hash(saved['state_dict'],True)==meta['frozen_actor_sha256']
    assert tensor_hash(saved['best_policy']['state_dict'])==meta['best_policy_sha256']
    assert all(torch.isfinite(v).all().item() for v in saved['state_dict'].values() if v.is_floating_point())
    assert len(list(CANONICAL.parent.glob('*.pt')))==1
    for p,h in meta['other_source_hashes'].items():assert sha(p)==h
    for p,h in meta['opponent_hashes'].items():assert sha(p)==h
    test=torch.load(OUT/'test-features.pt',weights_only=True);validation=torch.load(OUT/'validation-features.pt',weights_only=True)
    training_hashes={h for r in report['updates'] for h in r['replay_hashes']}
    assert test['seed']==684102 and validation['seed']==684101
    natural_overlaps=dict(validation_test_unique_trajectories=len(set(test['replay_hashes'])&set(validation['replay_hashes'])),training_holdout_unique_trajectories=len(training_hashes&(set(test['replay_hashes'])|set(validation['replay_hashes']))))
    groups={}
    for i,g in enumerate(test['games']):groups.setdefault(g['opponent'],[]).append(i)
    errors=report['test_game_errors'];rng=np.random.default_rng(684103)
    n=np.array([e['n'] for e in errors],dtype=float);selected=np.array([e['selected_sse'] for e in errors])
    bootstrap={}
    for reference,key in [('unchanged','original_sse'),('constant','constant_sse')]:
        difference=selected-np.array([e[key] for e in errors]);samples=[]
        for i in range(3000):
            ids=np.concatenate([rng.choice(values,size=len(values),replace=True) for values in groups.values()])
            samples.append(float(difference[ids].sum()/n[ids].sum()))
        bootstrap[reference]=dict(selected_minus_reference_MSE=float(difference.sum()/n.sum()),CI95=np.quantile(samples,[.025,.975]).tolist(),resamples=3000,definition='paired game-cluster bootstrap stratified by opponent; fixed trained model/seed, no training variance')
    results=dict(status='completed',selected_step=report['selected_step'],canonical_ppo_update=saved['update'],original_actor_and_BN_bitwise_unchanged=True,embedded_best320_unchanged=True,finite_checkpoint=True,one_canonical_model_file=True,source_opponent_hashes_verified=True,independent_game_draws_by_seed=True,natural_trajectory_overlaps=natural_overlaps,whole=summarize(report['updates']),first50=summarize(report['updates'][:50]),last50=summarize(report['updates'][-50:]),chunks50=[dict(end_round=(i+1)*50,**summarize(report['updates'][i*50:(i+1)*50])) for i in range(8)],initial_test=report['initial_test'],latest_test=report['latest_test'],selected_test=report['selected_test'],constant_test=report['constant_test'],bootstrap=bootstrap,validation=report['validation'],wall_seconds=meta['wall_seconds'],test_games=256,validation_games=128,test_positions=len(test['targets']),validation_positions=len(validation['targets']),constant_prediction=report['constant_prediction'])
    results['test_by_opponent']={}
    for name,indices in groups.items():
        total=sum(errors[i]['n'] for i in indices)
        results['test_by_opponent'][Path(name).stem]={label:math.sqrt(sum(errors[i][key] for i in indices)/total) for label,key in [('selected_RMSE','selected_sse'),('unchanged_RMSE','original_sse'),('constant_RMSE','constant_sse')]}
    write_json(OUT/'analysis.json',results)
    def f(v):return 'unavailable' if v is None else f'{v:.6f}'
    section='''\n<a id="stage-54"></a>\n\n## 54. Completed boosted value-head-only fitting:400 fresh-game rounds (2026-11-06)\n\n'''
    section+=f"400 value-fitting rounds completed,6400 new training games,3200/opponent,4games/color/opponent/round. PPO actor update remains400; this is **not**400 additional actor/PPO updates. Only the linear+tanh value head was fitted,4passes/minibatch128 each round,Adam LR1e-3/eps1e-5,head-only clip.5. Latest400 actor/encoder/BN buffers stayed bitwise unchanged; saved-best320 actor/critic snapshot was untouched. The canonical PPO file retains the validation-selected head from round{results['selected_step']} and matching optimizer state; original/latest/selected small value-head states and phase RNG/metrics are embedded for audit/resume. The400-round fitting phase does not advance PPO update numbering or change actor sampling weights. No historical data,PSRO or extra actor training was used.\n\n"
    section+=f"Game-level holdouts:128 validation games/{results['validation_positions']}positions and256 test games/{results['test_positions']}positions,balanced by opponent/color. Dedicated collection seeds684101/684102; fresh fitting seed684100. Different RNG streams produce independent game draws. Identical trajectories can naturally recur, especially short games; these are retained to avoid conditioning evaluation on novelty/outcome. No held-out game instance enters optimization. Natural trajectory overlap counts are recorded in analysis.json, not falsely described as novel states. Validation measured every20rounds selects minimum MSE,including original round0 as a candidate. Test predictions were measured initially and after fitting,never used for updates/selection. Constant predictor is mean discounted return over all fresh training positions ({report['constant_prediction']:.6f}),not the test mean. Both original and selected heads use the exact same test features/returns,so comparisons do not suffer from changed actor trajectories. Feature caching is valid because the encoder is frozen in eval mode.\n\n"
    section+='### Untouched test returns\n\n|Critic|Test RMSE|Test MAE|Test EV|Prediction mean|Prediction std|\n|---|---:|---:|---:|---:|---:|\n'
    for key,label in [('initial_test','Unchanged PPO400 critic'),('latest_test','Latest value round400'),('selected_test','Validation-selected head'),('constant_test','Training-mean constant')]:
        r=report[key];section+=f"|{label}|{f(r['rmse'])}|{f(r['mae'])}|{f(r['explained_variance'])}|{f(r['prediction_mean'])}|{f(r['prediction_std'])}|\n"
    section+='\n|Reference|Selected minus reference test MSE|Paired95%interval|\n|---|---:|---|\n'
    for name,z in bootstrap.items():section+=f"|{name}|{f(z['selected_minus_reference_MSE'])}|[{f(z['CI95'][0])},{f(z['CI95'][1])}]|\n"
    section+='\nBootstrap resamples256 whole test games3000times,stratified by frozen opponent. Negative difference favors fitted head. These intervals condition on one training run and one held-out sample,exclude training-seed uncertainty,and are not multiplicity-adjusted.\n\n'
    section+='|Frozen opponent|Unchanged test RMSE|Selected test RMSE|Constant test RMSE|\n|---|---:|---:|---:|\n'
    for name,z in results['test_by_opponent'].items():section+=f"|{name}|{f(z['unchanged_RMSE'])}|{f(z['selected_RMSE'])}|{f(z['constant_RMSE'])}|\n"
    section+='\n### Fresh-stream fitting versus generalization\n\n|Window|Fresh pre-fit RMSE|Fresh mean-round EV|Same-batch post-fit RMSE|Value loss|Head clip fraction|Mean pre-clip L2|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for key,label in [('whole','All400'),('first50','First50'),('last50','Last50')]:
        z=results[key];section+=f"|{label}|{f(z['before']['rmse'])}|{f(z['before']['mean_round_EV'])}|{f(z['after']['rmse'])}|{f(z['loss'])}|{100*z['clipped_fraction']:.2f}%|{f(z['gradient_mean'])}|\n"
    section+=f"\nTotal training positions={results['whole']['positions']};actual optimizer minibatch steps={results['whole']['optimizer_steps']};maximum raw head gradient={f(results['whole']['gradient_max'])};elapsed={results['wall_seconds']/60:.2f}minutes. Fresh-stream summary RMSE pools squared errors by decision; EV is a mean of round EVs. Test EV is pooled on the fixed test set. Same-batch post-fit error is not held-out generalization. Fixed actor training scores/rewards fluctuate from sampling and are **not policy improvement**.\n\n"
    delta=bootstrap['unchanged'];improved=delta['CI95'][1]<0
    section+='### Findings and next recommendation\n\n'
    if improved:section+='The validation-selected head improves test MSE over the unchanged critic on this paired game-cluster comparison. This supports a useful critic-fitting benefit for this actor/opponent distribution; it does not establish that the next PPO phase will learn faster or win more games. '
    else:section+='The selected head does not establish a test-MSE improvement over the unchanged critic under the paired game-cluster interval. More fresh-game value-only fitting is therefore not yet demonstrated to solve the critic problem. '
    if report['selected_test']['explained_variance']<.1:section+='Explained variance remains low even after head-only training,so a generalizing outcome predictor is still limited. '
    section+='A frozen shared representation constrains what a linear readout can learn; this experiment does not isolate representational limits from irreducible hidden-information/outcome noise or optimization/conditioning. No separate critic encoder was trained. Compare selected and unchanged critic initialization in a matched subsequent PPO experiment before attributing better actor learning to head fitting. If additional fitting is unsupported,avoid repeating400more rounds by default; a small independent-critic/nonlinear-readout diagnostic is more informative than enlarging the actor. No subsequent actor update or new experiment was launched by this report.\n\n'
    section+='Artifacts: `out/experiments/boosted-value-head-only-400-fresh-games/{pipeline.json,metrics.json,analysis.json,value-only-training.png,validation-features.pt,test-features.pt}`. Canonical boosted `policy.pt` retains one model file, original actor optimizer state/root PPO RNG and embedded best; phase metadata holds the fitted-head lifecycle separately. Original supervised/full/residual checkpoints and frozen opponents match launch hashes. Logs: `logs/boosted-value-head-only-400.out/.err`. Reproduction/analysis scripts in `work/run_boosted_value_only_400.py` and `work/report_boosted_value_only_400.py`.\n'
    p=ROOT/'RL_EXPERIMENT_HISTORY.md';text=p.read_text()
    if '<a id="stage-54"></a>' not in text:
        text=text.replace('stage numbers30–52','stage numbers30–54');text+=section;p.write_text(text)
    print('Test comparisons',json.dumps({k:report[k] for k in ['initial_test','latest_test','selected_test','constant_test']},indent=2),flush=True)
    print('Bootstrap',bootstrap,'selected round',report['selected_step'],flush=True)
    print('Completed stage54 appended; review before final delivery',flush=True)
if __name__=='__main__':main()
