#!/usr/bin/env python3
"""Fine-tune an existing feedforward FOW checkpoint with on-policy PPO."""
import argparse
import json
import hashlib
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rl.evaluation import evaluate
from rl.diagnostics import capture
from rl.opponents import OpponentPool
from rl.policy import load_policy, initialize_value
from rl.ppo import PPOConfig, optimizer_for, update
from rl.rollout import collect
from rl.metrics import game_summary, plot_metrics
from training.artifacts import EXPERIMENTS
from rl.storage import RL_ROOT, variant_name, training_lock, require_completed_pair
from training.diagnostics import write_json
from training.train import save_checkpoint

BASELINES=[EXPERIMENTS/'residual-medium-27890681/best_policy.pt',
           EXPERIMENTS/'full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt']


def parse_args(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--updates',type=int,default=200,help='Total completed PPO updates; may increase on resume')
    p.add_argument('--games-per-update',type=int,default=16)
    p.add_argument('--max-plies',type=int,default=256)
    p.add_argument('--base-time',type=float,default=180.)
    p.add_argument('--increment',type=float,default=2.)
    p.add_argument('--opponent',type=Path,action='append',default=None)
    p.add_argument('--opponent-command',action='append',default=None,
                   help='Trusted executable command implementing the existing bot protocol')
    p.add_argument('--self-play',action='store_true',help='Add a frozen current-policy copy to training opponents')
    p.add_argument('--self-play-history',type=int,default=4)
    p.add_argument('--snapshot-every',type=int,default=10)
    p.add_argument('--eval-opponent',type=Path,action='append',default=None)
    p.add_argument('--eval-command',action='append',default=None)
    p.add_argument('--eval-every',type=int,default=10,help='Zero disables evaluation; no best checkpoint is then selected')
    p.add_argument('--eval-games',type=int,default=20,help='Games per frozen opponent per evaluation')
    p.add_argument('--eval-seed',type=int,default=9137)
    p.add_argument('--save-replays',action='store_true')
    p.add_argument('--checkpoint-every',type=int,default=10)
    p.add_argument('--seed',type=int,default=7980)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    p.add_argument('--gamma',type=float,default=.99)
    p.add_argument('--gae-lambda',type=float,default=.95)
    defaults=PPOConfig()
    for key in defaults.__dict__:
        value=getattr(defaults,key)
        p.add_argument('--'+key.replace('_','-'),type=int if isinstance(value,int) else float,
                       default=None if key=='temperature' else value)
    args=p.parse_args(argv)
    if args.resume and args.init_checkpoint:
        p.error('Choose --init-checkpoint or --resume, not both')
    if args.init_checkpoint:
        initial=torch.load(args.init_checkpoint,map_location='cpu',weights_only=True)
        if initial['kind']!='feedforward':p.error('Recurrent policies are excluded from PPO training')
        args.run_dir=args.run_dir or RL_ROOT/variant_name(initial)
        canonical=args.run_dir/'policy.pt'
        if canonical.exists():
            args.resume=canonical
            print(f'Continuing existing RL policy: {canonical}',flush=True)
    if args.resume:
        saved_path=args.resume
        saved=torch.load(args.resume,map_location='cpu',weights_only=True)
        if args.init_checkpoint and variant_name(initial)!=variant_name(saved):
            p.error('Existing RL policy has a different architecture')
        if saved.get('algorithm')!='ppo' or 'optimizer_state_dict' not in saved or 'settings' not in saved:
            p.error('--resume needs a PPO checkpoint; use --init-checkpoint for imitation weights')
        explicit={a.dest for a in p._actions if any(flag==word or word.startswith(flag+'=') for word in argv for flag in a.option_strings)}
        if args.init_checkpoint:
            explicit.discard('init_checkpoint')
        if saved['training_log'].get('reward_scheme') != {'win': 1., 'loss': -1., 'draw': .1}:
            p.error('This checkpoint used a different reward scheme. Start a new run with --init-checkpoint to use draw +0.1; historical rewards must not be mixed on resume.')
        previous=saved['settings']
        for key in explicit-{'resume','updates','device','threads','run_dir','gae_lambda'}:
            old=previous[key]
            value=getattr(args,key)
            normalized=lambda v: str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v
            if normalized(value)!=old:
                p.error(f'Resume preserves {key}; start a new initialized run to change the recipe')
        overrides={key:getattr(args,key) for key in explicit}
        vars(args).update(previous)
        vars(args).update(overrides)
        args.resume=Path(saved_path)
        if args.updates<=saved['update']:
            p.error('--updates must exceed the number already completed')
    elif not args.init_checkpoint:
        p.error('Choose --init-checkpoint or --resume')
    for key in ['init_checkpoint','resume','run_dir']:
        if getattr(args,key) is not None:setattr(args,key,Path(getattr(args,key)))
    for key in ['opponent','eval_opponent']:
        if getattr(args,key) is not None:setattr(args,key,[Path(x) for x in getattr(args,key)])
    if min(args.updates,args.max_plies,args.threads,args.snapshot_every,args.checkpoint_every,
           args.epochs,args.minibatch_size)<1 or args.self_play_history<0 or args.eval_every<0:
        p.error('Counts must be positive (history/eval-every may be zero)')
    if args.games_per_update<2 or args.games_per_update%2 or args.eval_games<2 or args.eval_games%2:
        p.error('Training/evaluation game counts must be positive even numbers >=2')
    for key in ['learning_rate','value_learning_rate','max_grad_norm','temperature','base_time']:
        if key=='temperature' and args.temperature is None:continue
        if not 0<getattr(args,key)<float('inf'):p.error(f'{key} must be positive and finite')
    if not (0<=args.gamma<=1 and 0<=args.gae_lambda<=1 and 0<args.clip_ratio<1 and
            0<=args.target_kl<float('inf') and 0<=args.value_coefficient<float('inf') and
            0<=args.entropy_coefficient<float('inf') and 0<=args.increment<float('inf')):
        p.error('Invalid PPO/discount/time settings')
    return args


def settings(args):
    return {k: str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v
            for k,v in vars(args).items() if k not in ['resume','run_dir']}


def policy_weights(saved):
    result={k:saved[k] for k in ['kind','activation','negative_slope','model_config','state_dict',
                                 'inference_temperature','algorithm','update'] if k in saved}
    result['state_dict']={k:v.detach().cpu().clone() for k,v in saved['state_dict'].items()}
    return result


def main(argv=None):
    args=parse_args(argv)
    folder=args.resume.resolve().parent if args.resume else args.run_dir.resolve()
    with training_lock(folder):
        require_completed_pair(folder)
        return train(args)


def train(args):
    steps=training_steps(args)
    try:
        request=next(steps)
        while True:
            started=time.perf_counter()
            try:
                batch,games=request['collect']()
            except BaseException as error:
                steps.throw(error)
                raise
            request=steps.send((batch,games,time.perf_counter()-started))
    except StopIteration as finished:
        return finished.value
    finally:
        steps.close()


def training_steps(args, partner=None, save=save_checkpoint):
    """Yield collection requests so single and paired training share the same lifecycle."""
    torch.set_num_threads(args.threads)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    torch.manual_seed(args.seed);random.seed(args.seed)
    model,source=load_policy(args.resume or args.init_checkpoint,device)
    if source['kind']!='feedforward':
        raise ValueError('PPO actor must be feedforward; recurrent checkpoints can be frozen opponents')
    if args.temperature is None:args.temperature=float(source.get('inference_temperature',.9))
    if not 0<args.temperature<float('inf'):raise ValueError('Invalid checkpoint sampling temperature')
    config=PPOConfig(**{k:getattr(args,k) for k in PPOConfig().__dict__})
    optimizer=optimizer_for(model,config)
    if args.resume:
        run_dir=args.resume.resolve().parent
        if args.run_dir and args.run_dir.resolve()!=run_dir:raise ValueError('Resume stays in its run directory')
        optimizer.load_state_dict(source['optimizer_state_dict'])
        log=source['training_log'];completed=source['update'];history=source.get('opponent_history',[])
        if args.updates<=completed:raise ValueError('--updates must exceed the latest completed update')
        best_score=source.get('best_score',-1.)
        best_policy=source.get('best_policy')
        if best_policy is None and (run_dir/'best_policy.pt').exists():
            best_policy=policy_weights(torch.load(run_dir/'best_policy.pt',map_location='cpu',weights_only=True))
        history=[torch.load(run_dir/p,map_location='cpu',weights_only=True) if isinstance(p,str) else p for p in history]
    else:
        run_dir=args.run_dir.resolve()
        # The lock directory may already exist; never overwrite an existing trainer's policy.
        if (run_dir/'policy.pt').exists():
            raise RuntimeError('RL policy appeared during startup; rerun to resume it')
        if (run_dir/'metrics.json').exists():
            raise ValueError('This directory contains historical metrics; resume its checkpoint or use the canonical RL directory')
        # Imitation did not train this head. Zero initializes its predictions
        # without changing the pretrained encoder or actor logits.
        critic_initialization=initialize_value(model,source)
        log=dict(schema_version=3,algorithm='ppo',status='training',settings=settings(args),
                 variant=variant_name(source),
                 parameter_count=sum(p.numel() for p in model.parameters()),
                 source_sha256=hashlib.sha256(args.init_checkpoint.read_bytes()).hexdigest(),
                 critic_initialization=critic_initialization,
                 reward='terminal win +1, loss -1, draw +0.1; per learner decision discount',
                 reward_scheme={'win': 1., 'loss': -1., 'draw': .1},
                 game='canonical referee.run_match with agent adapter',
                 inference_modes='eval: dropout off, BN running statistics fixed, weights trainable',
                 updates=[],evaluations=[],activation_diagnostics=[])
        completed=0;history=[];best_score=-1.;best_policy=None
    opponents=args.opponent or []
    commands=args.opponent_command or []
    if not opponents and not commands and not args.self_play:opponents=BASELINES
    pool=OpponentPool(opponents,commands,device) if partner is None else None
    eval_checkpoints=args.eval_opponent if args.eval_opponent is not None else ([] if args.eval_command else BASELINES)
    eval_pool=OpponentPool(eval_checkpoints,
                           args.eval_command or [],device) if args.eval_every else None
    if args.resume:
        # All architecture/opponent initialization is finished before restoring RNG.
        torch.set_rng_state(source['torch_rng_state'].cpu())
        if device=='cuda' and source.get('cuda_rng_state_all'):
            torch.cuda.set_rng_state_all(source['cuda_rng_state_all'])
        random.setstate(source['python_rng_state'])
    if args.resume and source['settings']['gae_lambda'] != args.gae_lambda:
        log.setdefault('recipe_changes', []).append(dict(
            after_update=completed, parameter='gae_lambda',
            previous=source['settings']['gae_lambda'], value=args.gae_lambda))
    log.update(status='training',settings=settings(args),actual_device=device,torch_version=str(torch.__version__),
               training_mode='paired' if partner else 'single',partner=partner)
    started=time.perf_counter();previous_seconds=log.get('wall_seconds',0.)
    print(f'PPO run: {run_dir}; device: {device}; starting after update {completed}',flush=True)

    def checkpoint(iteration):
        return dict(kind='feedforward',activation=source.get('activation','relu'),
                    negative_slope=source.get('negative_slope',.01),model_config=source.get('model_config',{}),
                    state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    inference_temperature=args.temperature,algorithm='ppo',update=iteration,
                    storage_version=1,variant=variant_name(source),
                    optimizer_state_dict=optimizer.state_dict(),torch_rng_state=torch.get_rng_state(),
                    cuda_rng_state_all=torch.cuda.get_rng_state_all() if device=='cuda' else None,
                    python_rng_state=random.getstate(),opponent_history=history,best_score=best_score,best_policy=best_policy,
                    settings=settings(args),training_log=log,source_checkpoint=str(args.init_checkpoint))

    def evaluation(iteration):
        nonlocal best_score,best_policy
        if not eval_pool:return
        def replay(name,index,data):
            if args.save_replays:
                folder=run_dir/'evaluation-replays'/f'update-{iteration:06d}'
                folder.mkdir(parents=True,exist_ok=True)
                # Names may contain paths; use stable opponent index instead.
                opponent_index=[n for n,f in eval_pool.static].index(name)
                write_json(folder/f'opponent-{opponent_index}-game-{index:03d}.json',data)
        scores=evaluate(model,eval_pool.static,args.eval_games,args.temperature,args.max_plies,
                        args.base_time,args.increment,args.eval_seed,replay,gamma=args.gamma)
        score=sum(v['score'] for v in scores.values())/len(scores)
        log['evaluations'].append(dict(update=iteration,mean_score=score,
                                      mean_reward=sum(v['mean_reward'] for v in scores.values())/len(scores),opponents=scores))
        if score>best_score:
            best_score=score
            best_policy=policy_weights(checkpoint(iteration))
            save(checkpoint(iteration),run_dir/'policy.pt')
            print(f'New best checkpoint: update {iteration}, score {score:.1%}',flush=True)
        print(f'Evaluation update {iteration}: frozen-opponent mean score {score:.1%}',flush=True)

    try:
        if not args.resume:evaluation(0)
        for iteration in range(completed+1,args.updates+1):
            training_pool=pool.training(model,args.self_play,history,args.temperature) if pool else []
            replay_index=0
            def replay(data):
                nonlocal replay_index
                if args.save_replays:
                    folder=run_dir/'training-replays'/f'update-{iteration:06d}'
                    folder.mkdir(parents=True,exist_ok=True)
                    write_json(folder/f'game-{replay_index:03d}.json',data)
                replay_index+=1
            batch,games,rollout_seconds=yield dict(model=model,args=args,iteration=iteration,folder=run_dir,replay=replay,
                                  collect=lambda:collect(model,training_pool,args.games_per_update,args.temperature,args.gamma,
                                                         args.gae_lambda,args.max_plies,args.base_time,args.increment,replay))
            if not log.setdefault('activation_diagnostics',[]):
                log['activation_diagnostics'].append(capture(model,batch.tokens,run_dir,iteration-1,args.seed))
            update_started=time.perf_counter();metrics=update(model,optimizer,batch,config)
            metrics.update(gamma=args.gamma, gae_lambda=args.gae_lambda)
            summary=game_summary(games)
            summary.update(games=games,
                           by_opponent={name:game_summary([g for g in games if g['opponent']==name])
                                        for name in sorted({g['opponent'] for g in games})},
                           by_color={color:game_summary([g for g in games if g['color']==color])
                                     for color in ['white','black']},
                           cumulative_games=sum(len(r['rollout']['games']) for r in log['updates'])+len(games),
                           cumulative_decisions=sum(r['ppo']['transitions'] for r in log['updates'])+len(batch),
                           games_per_second=len(games)/rollout_seconds,
                           decisions_per_second=len(batch)/rollout_seconds)
            row=dict(update=iteration,rollout=summary,ppo=metrics,rollout_seconds=rollout_seconds,
                     optimization_seconds=time.perf_counter()-update_started)
            log['updates'].append(row)
            if iteration%args.checkpoint_every==0 or iteration==args.updates:
                log['activation_diagnostics'].append(capture(model,batch.tokens,run_dir,iteration,args.seed))
            if args.self_play and args.self_play_history and iteration%args.snapshot_every==0:
                snapshot=policy_weights(checkpoint(iteration))
                history.append(snapshot);history=history[-args.self_play_history:]
            if args.eval_every and (iteration%args.eval_every==0 or iteration==args.updates):evaluation(iteration)
            log['wall_seconds']=previous_seconds+time.perf_counter()-started
            saved=checkpoint(iteration)
            save(saved,run_dir/'policy.pt')
            write_json(run_dir/'metrics.json',log)
            if iteration%args.checkpoint_every==0 or iteration==args.updates:plot_metrics(log,run_dir)
            print(f"Update {iteration}/{args.updates} | W/L/D {row['rollout']['wins']}/{row['rollout']['losses']}/{row['rollout']['draws']} | "
                  f"reward {summary['mean_reward']:.3f} | decisions {len(batch)} | KL {metrics['endpoint_kl']:.5f} | entropy {metrics['endpoint_entropy']:.3f}",flush=True)
        log.update(status='completed',wall_seconds=previous_seconds+time.perf_counter()-started)
        saved=checkpoint(args.updates)
        save(saved,run_dir/'policy.pt')
        write_json(run_dir/'metrics.json',log)
    except GeneratorExit:
        raise
    except BaseException as error:
        write_json(run_dir/'failure.json',dict(error=str(error),last_completed_update=log['updates'][-1]['update'] if log['updates'] else completed))
        raise
    return run_dir


if __name__=='__main__':
    print(main())
