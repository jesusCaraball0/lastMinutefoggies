"""Fresh-rollout head-only value fitting; frozen canonical actor and game-level holdouts."""
import sys,json,time,random,hashlib,copy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from rl.policy import load_policy
from rl.ppo import optimizer_for,PPOConfig,value_quality
from rl.storage import training_lock,require_completed_pair
from training.datasets import observation_planes
from training.train import save_checkpoint
from training.diagnostics import write_json
from work.run_boosted_ppo_400 import collect_balanced
from rl.opponents import OpponentPool
from rl.metrics import game_summary

OUT=ROOT/'out/experiments/boosted-value-head-only-400-fresh-games'
CANONICAL=ROOT/'out/rl/boosted-cnn-c32-h128-f4004a979d/policy.pt'
PRIOR=ROOT/'out/experiments/boosted-cnn-ppo-400-balanced-frozen-opponents'
PHASE='boosted_fresh_value_head_400'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_hash(state,exclude_value=False):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        if exclude_value and k.startswith('value.'):continue
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def head_state(model):return {k:v.detach().cpu().clone() for k,v in model.value.state_dict().items()}
def features_for(model,tokens):
    with torch.no_grad():
        return torch.cat([model.encoder(observation_planes(tokens[i:i+128],'cpu')).detach() for i in range(0,len(tokens),128)])
def predictions(head,features):
    with torch.no_grad():return head(features).tanh()[:,0]
def fit_head(head,optimizer,features,targets,epochs=4,minibatch=128,max_norm=.5):
    norms=[];losses=[];count=0
    for epoch in range(epochs):
        order=torch.randperm(len(features))
        for start in range(0,len(features),minibatch):
            ids=order[start:start+minibatch];pred=head(features[ids]).tanh()[:,0]
            loss=.5*(pred-targets[ids]).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite value-head loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),max_norm,error_if_nonfinite=True)
            optimizer.step();norms.append(float(norm));losses.append(float(loss.detach())*len(ids));count+=len(ids)
    return dict(loss=sum(losses)/count,optimizer_steps=len(norms),gradient_mean=sum(norms)/len(norms),gradient_max=max(norms),gradient_clipped_fraction=sum(n>max_norm for n in norms)/len(norms))
def draw_report(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=report['updates'];xs=[r['step'] for r in rows];fig,axes=plt.subplots(2,3,figsize=(16,8))
    for phase,label in [('before','Fresh training rollout'),('after','Same rollout after fitting')]:
        axes[0,0].plot(xs,[r[phase]['rmse'] for r in rows],label=label)
        axes[0,1].plot(xs,[r[phase]['explained_variance'] for r in rows],label=label)
    ev=report['validation'];ex=[v['step'] for v in ev]
    axes[0,0].plot(ex,[v['value']['rmse'] for v in ev],label='Fixed validation')
    axes[0,1].plot(ex,[v['value']['explained_variance'] for v in ev],label='Fixed validation')
    for key,label in [('loss','Half MSE'),('gradient_mean','Mean pre-clip norm'),('gradient_clipped_fraction','Clipping fraction')]:
        ax=axes[0,2] if key=='loss' else axes[1,0] if key=='gradient_mean' else axes[1,1]
        ax.plot(xs,[r['fit'][key] for r in rows],label=label)
    axes[1,2].plot(xs,[r['rollout']['score'] for r in rows],label='Frozen actor training score')
    for ax,title in zip(axes.flat,['Value RMSE','Value explained variance','Value fitting loss','Head gradient L2','Head clipping fraction','Score (sampling noise; actor frozen)']):
        ax.set(title=title,xlabel='Value-only fitting round');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'value-only-training.png',dpi=150);plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    with training_lock(OUT),training_lock(CANONICAL.parent):
        require_completed_pair(CANONICAL.parent)
        model,saved=load_policy(CANONICAL)
        assert saved['update']==400 and saved['model_config']['boosted']
        model.eval()
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('value.'))
        optimizer=optimizer_for(model,PPOConfig());optimizer.load_state_dict(saved['optimizer_state_dict'])
        launch=OUT/'pipeline.json'
        prior=json.loads((PRIOR/'pipeline.json').read_text())
        if launch.exists():meta=json.loads(launch.read_text())
        else:
            assert PHASE not in saved.get('value_head_phases',{}),'Existing phase requires its manifest'
            meta=dict(status='setup',started_unix=time.time(),phase=PHASE,source_ppo_update=400,target_steps=400,
                      actor='latest400, unchanged throughout; saved best320 preserved',canonical=str(CANONICAL),
                      initial_checkpoint_sha256=sha(CANONICAL),frozen_actor_sha256=tensor_hash(saved['state_dict'],True),
                      best_policy_sha256=tensor_hash(saved['best_policy']['state_dict']),
                      opponent_hashes={o['path']:sha(o['path']) for o in prior['opponents']},
                      other_source_hashes={p:h for p,h in prior['source_hashes'].items()},
                      implementation_hashes={str(p):sha(p) for p in [ROOT/'work/run_boosted_value_only_400.py',ROOT/'work/run_boosted_ppo_400.py',ROOT/'rl/rollout.py',ROOT/'rl/policy.py']},
                      recipe=dict(train_seed=684100,validation_seed=684101,test_seed=684102,bootstrap_seed=684103,validation_games=128,test_games=256,games_per_step=16,games_per_opponent_per_step=8,gamma=.99,temperature=.9,max_plies=256,learning_rate=.001,adam_epsilon=.00001,passes=4,minibatch=128,gradient_clip=.5,validation_every=20,selection='minimum fixed-validation MSE; initial critic is eligible'))
            write_json(launch,meta)
        for p,h in meta['opponent_hashes'].items():assert sha(p)==h
        for p,h in meta['other_source_hashes'].items():assert sha(p)==h
        assert tensor_hash(model.state_dict(),True)==meta['frozen_actor_sha256']
        assert tensor_hash(saved['best_policy']['state_dict'])==meta['best_policy_sha256']
        pool=OpponentPool([Path(o['path']) for o in prior['opponents']])
        args=SimpleNamespace(games_per_update=16,temperature=.9,gamma=.99,gae_lambda=1,max_plies=256,base_time=180.,increment=2.)
        current_hashes=[]
        def replay_id(replay):
            moves=' '.join(str(f.get('move','')) for f in replay['frames'])
            current_hashes.append(hashlib.sha256((moves+' '+replay['result']).encode()).hexdigest())
        request=dict(args=args,model=model,replay=replay_id)
        # Collect holdout games under RNG streams that cannot perturb training.
        for label,games,seed in [('validation',128,684101),('test',256,684102)]:
            path=OUT/(label+'-features.pt')
            if path.exists():continue
            random.seed(seed);torch.manual_seed(seed);fs=[];ys=[];ids=[];reports=[];hashes=[]
            print('Collecting fixed',label,games,'games',flush=True)
            for i in range(games//16):
                current_hashes.clear();batch,records=collect_balanced(request,pool.static)
                fs.append(features_for(model,batch.tokens));ys.append(batch.discounted_returns)
                lens=[]
                # Each individual game's return sequence ends at the terminal reward;
                # collect sizes via a lightweight collector callback below is unnecessary:
                # Trajectories reports contain plies but learner counts follow its color.
                for r in records:
                    n=(r['plies']+1)//2 if r['color']=='white' else r['plies']//2
                    lens.append(n)
                assert sum(lens)==len(batch)
                ids.extend([i*16+j for j,n in enumerate(lens) for _ in range(n)])
                reports.extend(records);hashes.extend(current_hashes)
            data=dict(features=torch.cat(fs),targets=torch.cat(ys),game_ids=torch.tensor(ids),games=reports,replay_hashes=hashes,seed=seed)
            save_checkpoint(data,path)
        validation=torch.load(OUT/'validation-features.pt',weights_only=True)
        test=torch.load(OUT/'test-features.pt',weights_only=True)
        meta['natural_validation_test_repeated_trajectories']=len(set(validation['replay_hashes'])&set(test['replay_hashes']))
        meta['game_split_definition']='Independent game draws under separate RNG seeds; naturally repeated trajectories are retained, not censored by their outcome.'
        meta['current_driver_sha256']=sha(Path(__file__))
        write_json(launch,meta)
        heldout_hashes=set(validation['replay_hashes'])|set(test['replay_hashes'])
        phase=saved.get('value_head_phases',{}).get(PHASE)
        if phase:
            report=phase.get('training_log') or json.loads((OUT/'metrics.json').read_text());completed=phase['completed_steps']
            if report['status']=='completed':
                import subprocess
                subprocess.run([sys.executable,'work/report_boosted_value_only_400.py'],cwd=ROOT,check=True)
                print('Value-only phase already completed; report verified',flush=True);return
            torch.set_rng_state(phase['torch_rng_state']);random.setstate(phase['python_rng_state'])
        else:
            assert not (OUT/'metrics.json').exists(),'Metrics without canonical phase checkpoint; inspect recovery'
            completed=0;random.seed(684100);torch.manual_seed(684100)
            original=head_state(model)
            phase=dict(completed_steps=0,original_value_state_dict=original,best_value_state_dict=original,best_step=0,
                       best_validation_mse=value_quality(predictions(model.value,validation['features']),validation['targets'])['mse'],best_optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
            report=dict(status='training',phase=PHASE,settings=meta['recipe'],updates=[],validation=[],initial_validation=value_quality(predictions(model.value,validation['features']),validation['targets']),initial_test=value_quality(predictions(model.value,test['features']),test['targets']),actor_frozen=True,source_ppo_update=400,original_head=original)
            # JSON contains measurements, checkpoint contains tensors.
            report.pop('original_head')
            report['validation'].append(dict(step=0,value=report['initial_validation']))
            write_json(OUT/'metrics.json',report)
        def commit(step):
            assert tensor_hash(model.state_dict(),True)==meta['frozen_actor_sha256']
            phase.update(completed_steps=step,torch_rng_state=torch.get_rng_state(),python_rng_state=random.getstate())
            phase['training_log']=report
            saved.setdefault('value_head_phases',{})[PHASE]=phase
            saved['state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()}
            saved['optimizer_state_dict']=optimizer.state_dict()
            report['wall_seconds']=time.time()-meta['started_unix']
            saved['training_log'].setdefault('value_head_training',{})[PHASE]=dict(status=report['status'],completed_steps=step,report=str(OUT/'metrics.json'),best_step=phase['best_step'],actor_frozen=True)
            save_checkpoint(saved,CANONICAL);write_json(OUT/'metrics.json',report)
            write_json(CANONICAL.parent/'metrics.json',saved['training_log'])
        # Commit the starting head/RNG before the first fresh training batch.
        if completed==0:commit(0)
        meta['status']='value-only training';write_json(launch,meta)
        for step in range(completed+1,401):
            current_hashes.clear();start=time.perf_counter();batch,games=collect_balanced(request,pool.static)
            natural_repeats=sum(h in heldout_hashes for h in current_hashes)
            for opponent in prior['opponents']:
                group=[g for g in games if g['opponent']==opponent['path']]
                assert len(group)==8 and sum(g['color']=='white' for g in group)==4
            feats=features_for(model,batch.tokens);targets=batch.discounted_returns
            before=value_quality(batch.old_values,targets)
            fit=fit_head(model.value,optimizer,feats,targets)
            after=value_quality(predictions(model.value,feats),targets)
            report['updates'].append(dict(step=step,rollout=game_summary(games),games=games,replay_hashes=current_hashes.copy(),natural_trajectory_repeats_with_holdouts=natural_repeats,before=before,after=after,fit=fit,target_sum=float(targets.double().sum()),target_samples=len(targets),seconds=time.perf_counter()-start))
            if step%20==0 or step==400:
                value=value_quality(predictions(model.value,validation['features']),validation['targets'])
                report['validation'].append(dict(step=step,value=value))
                if value['mse']<phase['best_validation_mse']:
                    phase.update(best_step=step,best_validation_mse=value['mse'],best_value_state_dict=head_state(model),best_optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                print('Validation step',step,'RMSE',round(value['rmse'],5),'EV',round(value['explained_variance'],5),'best',phase['best_step'],flush=True)
            commit(step)
            if step%20==0:draw_report(report)
            print(f'Value round {step}/400 | fresh RMSE {before["rmse"]:.5f} EV {before["explained_variance"]:.5f} | fitted RMSE {after["rmse"]:.5f}',flush=True)
        # Evaluate final latest and validation-selected head on the untouched game split.
        phase['latest_value_state_dict']=head_state(model)
        report['latest_test']=value_quality(predictions(model.value,test['features']),test['targets'])
        model.value.load_state_dict(phase['best_value_state_dict']);optimizer.load_state_dict(phase['best_optimizer_state_dict'])
        report['selected_test']=value_quality(predictions(model.value,test['features']),test['targets'])
        report['selected_validation']=value_quality(predictions(model.value,validation['features']),validation['targets'])
        report.update(status='completed',selected_step=phase['best_step'],committed='validation-selected value head; actor and PPO update count unchanged')
        train_mean=sum(r['target_sum'] for r in report['updates'])/sum(r['target_samples'] for r in report['updates'])
        report['constant_test']=value_quality(torch.full_like(test['targets'],train_mean),test['targets'])
        report['constant_prediction']=train_mean
        # Per-game sufficient error sums for uncertainty; no held-out positions used for optimization.
        original=torch.nn.Linear(128,1);original.load_state_dict(phase['original_value_state_dict'])
        final_predictions=predictions(model.value,test['features']);original_predictions=predictions(original,test['features'])
        y=test['targets'];ids=test['game_ids'];game_count=len(test['games'])
        pergame=[]
        for i in range(game_count):
            mask=ids==i
            pergame.append(dict(game=i,n=int(mask.sum()),selected_sse=float((final_predictions[mask]-y[mask]).double().square().sum()),original_sse=float((original_predictions[mask]-y[mask]).double().square().sum()),constant_sse=float((train_mean-y[mask]).double().square().sum())))
        report['test_game_errors']=pergame
        assert tensor_hash(model.state_dict(),True)==meta['frozen_actor_sha256']
        assert tensor_hash(saved['best_policy']['state_dict'])==meta['best_policy_sha256']
        report.update(actor_and_BN_buffers_bitwise_unchanged=True,saved_best_policy_unchanged=True)
        commit(400);draw_report(report)
        for p,h in meta['other_source_hashes'].items():assert sha(p)==h
        for p,h in meta['opponent_hashes'].items():assert sha(p)==h
        meta.update(status='completed',finished_unix=time.time(),wall_seconds=time.time()-meta['started_unix']);write_json(launch,meta)
        import subprocess
        subprocess.run([sys.executable,'work/report_boosted_value_only_400.py'],cwd=ROOT,check=True)
        print('Value-only experiment and report complete',flush=True)
if __name__=='__main__':main()
