#!/usr/bin/env python3
"""One historical-game epoch fitting only the value heads of persistent PPO models."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from rl.policy import load_policy
from rl.rollout import terminal_reward
from rl.storage import training_lock
from rl.train_pair import commit, recover
from training.datasets import observation_planes, replay_examples
from training.diagnostics import write_json

VALID_REASONS={'King capturable','No legal moves','Threefold repetition','50-move rule','Move limit reached'}


def value_examples(replay,gamma=.99):
    """Own pre-decision fog observations; no hidden-board inputs or engine labels."""
    if replay.get('rulesVersion')!='fow-v1':return [],'incompatible or missing rulesVersion'
    if replay.get('failedSide') or replay.get('reason') not in VALID_REASONS:return [],'non-gameplay termination'
    if replay.get('result') not in ('1-0','0-1','1/2-1/2'):return [],'unfinished result'
    examples=list(replay_examples(replay))
    frames=replay['frames'][:-1]
    if len(examples)!=len(frames):raise ValueError('Move/observation count mismatch')
    counts=Counter(frame['ply']%2==0 for frame in frames)
    seen=Counter();result=[]
    for frame,(tokens,action) in zip(frames,examples):
        white=frame['ply']%2==0
        remaining=counts[white]-seen[white]-1
        target=terminal_reward(replay['result'],white)*gamma**remaining
        result.append((tokens,target));seen[white]+=1
    return result,None if result else 'no decisions'


def dataset(paths,gamma):
    tokens,targets=[],[]
    report=dict(files=len(paths),games=0,positions=0,excluded={},reasons={},results={},bad_examples=[])
    excluded,reasons,results=Counter(),Counter(),Counter()
    for i,path in enumerate(paths):
        try:
            replay=json.loads(path.read_text())
            examples,why=value_examples(replay,gamma)
        except (ValueError,KeyError,TypeError,IndexError) as error:
            examples=[];why='invalid replay'
            if len(report['bad_examples'])<10:report['bad_examples'].append(dict(file=str(path),error=str(error)))
        if examples:
            views,values=zip(*examples)
            tokens.append(np.asarray(views,dtype=np.uint8));targets.append(np.asarray(values,dtype=np.float32))
            report['games']+=1;report['positions']+=len(examples)
            reasons[replay['reason']]+=1;results[replay['result']]+=1
        else:excluded[why]+=1
        if (i+1)%5000==0:print(f'Read {i+1}/{len(paths)} games; {report["positions"]:,} usable positions',flush=True)
    if not tokens:raise ValueError('No usable value examples')
    report.update(excluded=dict(excluded),reasons=dict(reasons),results=dict(results))
    return TensorDataset(torch.from_numpy(np.concatenate(tokens)),torch.from_numpy(np.concatenate(targets))),report


class Moments:
    def __init__(self):self.n=0;self.y=0.;self.yy=0.;self.p=0.;self.pp=0.;self.e=0.;self.ee=0.;self.a=0.
    def add(self,predictions,targets):
        p=predictions.detach().double();y=targets.double();e=p-y
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite value prediction')
        self.n+=len(y);self.y+=float(y.sum());self.yy+=float(y.square().sum())
        self.p+=float(p.sum());self.pp+=float(p.square().sum())
        self.e+=float(e.sum());self.ee+=float(e.square().sum());self.a+=float(e.abs().sum())
    def report(self):
        n=self.n;vy=max(0.,self.yy/n-(self.y/n)**2);ve=max(0.,self.ee/n-(self.e/n)**2)
        return dict(samples=n,mse=self.ee/n,rmse=(self.ee/n)**.5,mae=self.a/n,
                    explained_variance=1-ve/vy if vy>1e-12 else None,
                    prediction_mean=self.p/n,prediction_std=max(0.,self.pp/n-(self.p/n)**2)**.5,
                    target_mean=self.y/n,target_std=vy**.5)


def evaluate_heads(models,loader,device):
    moments=[Moments() for _ in models]
    with torch.no_grad():
        for tokens,targets in loader:
            planes=observation_planes(tokens,device);targets=targets.to(device)
            for model,meter in zip(models,moments):
                meter.add(model.value(model.encoder(planes)).tanh()[:,0],targets)
    return [m.report() for m in moments]


def fit_heads(models,train_loader,validation_loader,device,learning_rates,max_norm=.5,progress=None):
    """One epoch. Encoder/policy weights AND BN buffers stay frozen throughout."""
    frozen=[{k:v.detach().cpu().clone() for k,v in m.state_dict().items() if not k.startswith('value.')} for m in models]
    for model in models:
        model.eval()
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('value.'))
    before=evaluate_heads(models,validation_loader,device)
    optimizers=[torch.optim.Adam(m.value.parameters(),lr=lr,eps=1e-5) for m,lr in zip(models,learning_rates)]
    meters=[Moments() for _ in models];norms=[[] for _ in models];losses=[0. for _ in models];count=0
    for step,(tokens,targets) in enumerate(train_loader,1):
        planes=observation_planes(tokens,device);targets=targets.to(device);count+=len(targets)
        for i,(model,optimizer) in enumerate(zip(models,optimizers)):
            with torch.no_grad():features=model.encoder(planes)
            prediction=model.value(features).tanh()[:,0]
            loss=.5*(prediction-targets).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite value pretraining loss')
            meters[i].add(prediction,targets);losses[i]+=float(loss.detach())*len(targets)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.value.parameters(),max_norm,error_if_nonfinite=True)
            norms[i].append(float(norm));optimizer.step()
        if progress and (step%100==0 or step==len(train_loader)):
            progress(step,len(train_loader),[meter.report() for meter in meters])
    after=evaluate_heads(models,validation_loader,device)
    reports=[]
    for i,model in enumerate(models):
        for key,value in frozen[i].items():
            if not torch.equal(value,model.state_dict()[key].cpu()):raise AssertionError(f'Actor or BN buffer changed: {key}')
        ns=torch.tensor(norms[i])
        reports.append(dict(validation_before=before[i],validation_after=after[i],train_online=meters[i].report(),
                            train_half_mse=losses[i]/count,optimizer_steps=len(norms[i]),learning_rate=learning_rates[i],
                            gradient_clip=max_norm,gradient_clipped_fraction=float((ns>max_norm).float().mean()),
                            gradient_l2_mean=float(ns.mean()),gradient_l2_p95=float(torch.quantile(ns,.95)),
                            gradient_l2_max=float(ns.max()),actor_and_buffers_unchanged=True))
    return reports


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,action='append',required=True)
    parser.add_argument('--data-dir',type=Path,default=ROOT/'game_data')
    parser.add_argument('--split',type=Path,default=ROOT/'out/experiments/residual-medium-27890681/data_split.json')
    parser.add_argument('--after-update',type=int,help='Guard against accidentally repeating a warmup after PPO has advanced')
    parser.add_argument('--batch-size',type=int,default=1024)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args(argv)
    if args.batch_size<1 or args.threads<1:parser.error('Batch size and threads must be positive')
    paths=sorted([p.resolve() for p in args.checkpoint]);folders=[p.parent for p in paths]
    if len(paths)!=2 or len(set(folders))!=2:parser.error('Pass the two distinct persistent paired PPO checkpoint files')
    if any(p.name!='policy.pt' for p in paths):parser.error('Pretraining updates persistent policy.pt files in place')
    torch.set_num_threads(args.threads)
    with ExitStack() as locks:
        for folder in folders:locks.enter_context(training_lock(folder))
        recover(folders)
        loaded=[load_policy(p,args.device) for p in paths]
        models=[m for m,s in loaded];saved=[s for m,s in loaded]
        if any(s.get('algorithm')!='ppo' or s['kind']!='feedforward' for s in saved):raise ValueError('Need feedforward PPO checkpoints')
        update=saved[0]['update'];gamma=saved[0]['settings']['gamma']
        if any(s['update']!=update or s['settings']['gamma']!=gamma for s in saved):raise ValueError('Pair update/discount mismatch')
        expected=update if args.after_update is None else args.after_update
        done=[any(v['after_ppo_update']==expected for v in s['training_log'].get('value_pretraining',[])) for s in saved]
        if all(done):
            print(f'Historical value pretraining is already recorded after update {expected}; skipping.',flush=True)
            return
        if any(done):raise ValueError('Only one member records value pretraining; recover the pair before continuing')
        if update!=expected:raise ValueError(f'Expected PPO update {expected}, found {update}; refusing another warmup')
        manifest=json.loads(args.split.read_text());split={key:[args.data_dir/Path(p).name for p in manifest[key]] for key in ['train','validation']}
        if set(split['train']) & set(split['validation']):raise ValueError('Train/validation game overlap')
        if any(not p.exists() for paths_ in split.values() for p in paths_):raise FileNotFoundError('Missing historical replay in split')
        seed=saved[0]['settings']['seed'];started=time.perf_counter()
        print(f'Historical value-only warmup after PPO update {update}: exactly one epoch',flush=True)
        train_data,train_report=dataset(split['train'],gamma)
        val_data,val_report=dataset(split['validation'],gamma)
        train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=0,
                                generator=torch.Generator().manual_seed(seed))
        val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False,num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
        report=dict(schema_version=1,phase='historical_value_head_only',after_ppo_update=update,epochs=1,
                    gamma=gamma,rewards={'win':1.,'loss':-1.,'draw':.1},batch_size=args.batch_size,device=args.device,threads=args.threads,
                    split_source=str(args.split),split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
                    train=train_report,validation=val_report,models={},status='training',progress=[])
        output=folders[0].parent/f'value-pretrain-after-{update}.json'
        write_json(output,report)
        def progress(step,total,stats):
            report['progress'].append(dict(step=step,total=total,online=stats))
            write_json(output,report)
            print(f'Value epoch 1/1: {step}/{total} batches; online RMSE '+', '.join(f'{s["rmse"]:.4f}' for s in stats),flush=True)
        metrics=fit_heads(models,train_loader,val_loader,args.device,
                          [s['settings']['value_learning_rate'] for s in saved],progress=progress)
        pending={}
        for path,model,state,measurements in zip(paths,models,saved,metrics):
            entry=dict(after_ppo_update=update,epochs=1,train_positions=train_report['positions'],validation_positions=val_report['positions'],
                       source_report=str(output),**measurements)
            state['state_dict']={k:v.detach().cpu() for k,v in model.state_dict().items()}
            # Keep all actor Adam moments; old critic moments reflect a different fitting phase.
            critic_ids=state['optimizer_state_dict']['param_groups'][1]['params']
            for parameter_id in critic_ids:state['optimizer_state_dict']['state'].pop(parameter_id,None)
            entry['critic_optimizer_moments']='reset for on-policy adaptation; actor moments unchanged'
            state['training_log'].setdefault('value_pretraining',[]).append(entry)
            pending[path]=state;report['models'][path.parent.name]=entry
        commit(pending,folders)
        report.update(status='completed',wall_seconds=time.perf_counter()-started)
        write_json(output,report)
        for path,state in zip(paths,saved):write_json(path.parent/'metrics.json',state['training_log'])
        # Only metadata in the latest state changes; best actor/critic remains its evaluated snapshot.
        plot_report(report,output.with_suffix('.png'))
        print('Value pretraining complete; both persistent checkpoints committed.',flush=True)
        return output


def plot_report(report,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(report['models']);fig,axes=plt.subplots(1,2,figsize=(11,4))
    for i,name in enumerate(names):
        d=report['models'][name]
        for j,phase in enumerate(['validation_before','validation_after']):
            axes[0].bar(i+(j-.5)*.3,d[phase]['rmse'],width=.3,label=phase if i==0 else None)
            axes[1].bar(i+(j-.5)*.3,d[phase]['explained_variance'],width=.3,label=phase if i==0 else None)
    for ax,title in zip(axes,['Historical held-out RMSE','Historical held-out explained variance']):
        ax.set(title=title,xticks=range(len(names)),xticklabels=['Full CNN' if n.startswith('cnn') else 'Residual' for n in names]);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':main()
