#!/usr/bin/env python3
"""Train two feedforward policies from opposite sides of the same games."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rl.rollout import collect_pair
from rl.storage import RL_ROOT, variant_name, training_lock
from rl.train import parse_args, training_steps
from training.train import save_checkpoint


def commit(pending, folders):
    """A short-lived journal lets a restart finish an interrupted two-file commit."""
    if not pending:return
    journal=sorted(folders)[0]/'.paired-commit.tmp'
    payload=[(str(folder/'policy.pt'),pending[folder/'policy.pt']) for folder in folders]
    save_checkpoint(payload,journal)
    for path,saved in payload:save_checkpoint(saved,Path(path))
    journal.unlink()
    pending.clear()


def recover(folders):
    journal=sorted(folders)[0]/'.paired-commit.tmp'
    if not journal.exists():return
    payload=torch.load(journal,map_location='cpu',weights_only=True)
    if {Path(path) for path,saved in payload}!={folder/'policy.pt' for folder in folders}:
        raise ValueError('Interrupted commit belongs to another model pair; resume that pair first')
    for path,saved in payload:save_checkpoint(saved,Path(path))
    journal.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,epilog='Other options are shared with rl/train.py (e.g. --updates, --games-per-update, --device).')
    parser.add_argument('--first',type=Path,required=True,help='First supervised or PPO source checkpoint')
    parser.add_argument('--second',type=Path,required=True,help='Second supervised or PPO source checkpoint')
    parser.add_argument('--run-root',type=Path,default=RL_ROOT,help='Parent of persistent variant directories')
    pair,shared=parser.parse_known_args(argv)
    forbidden=['--resume','--init-checkpoint','--run-dir','--self-play','--opponent','--opponent-command']
    if any(word.split('=')[0] in forbidden for word in shared):
        parser.error('Paired training takes --first/--second and uses only those two training opponents')
    sources=[pair.first,pair.second]
    names=[variant_name(torch.load(p,map_location='cpu',weights_only=True)) for p in sources]
    if names[0]==names[1]:parser.error('Choose two different model variants')
    # Stable order also makes shared RNG continuation independent of CLI ordering.
    names,sources=map(list,zip(*sorted(zip(names,sources))))
    folders=[(pair.run_root/name).resolve() for name in names]
    pending={}
    def stage(saved,path):pending[Path(path)]=saved
    with ExitStack() as locks:
        for folder in sorted(folders):locks.enter_context(training_lock(folder))
        recover(folders)
        args=[parse_args(['--init-checkpoint',str(source),'--run-dir',str(folder)]+shared)
              for source,folder in zip(sources,folders)]
        if any(a.self_play for a in args):
            raise ValueError('This checkpoint used self-play history; paired mode requires self-play disabled')
        shared_fields=['updates','games_per_update','max_plies','base_time','increment','gamma','gae_lambda','device']
        for key in shared_fields:
            if getattr(args[0],key)!=getattr(args[1],key):
                raise ValueError(f'Paired collection requires matching {key}')
        steppers=[training_steps(a,partner=names[1-i],save=stage) for i,a in enumerate(args)]
        try:
            requests=[next(stepper) for stepper in steppers]
            if requests[0]['iteration']!=requests[1]['iteration']:
                raise ValueError('Paired policies must have the same completed update count')
            commit(pending,folders)  # Initial evaluations / new bests.
            while True:
                first,second=requests
                a=first['args']
                def replay(data):
                    for request in requests:request['replay'](data)
                started=time.perf_counter()
                batches=collect_pair([r['model'] for r in requests],names,a.games_per_update,
                                     [r['args'].temperature for r in requests],a.gamma,a.gae_lambda,
                                     a.max_plies,a.base_time,a.increment,replay)
                rollout_seconds=time.perf_counter()-started
                following=[]
                finished=[]
                # Both batches were collected before either policy receives an update.
                for stepper,batch in zip(steppers,batches):
                    try:
                        following.append(stepper.send((*batch,rollout_seconds)));finished.append(False)
                    except StopIteration:
                        finished.append(True)
                if any(finished) and not all(finished):
                    raise RuntimeError('Paired policies finished at different update counts')
                commit(pending,folders)
                if all(finished):return folders
                requests=following
        except BaseException as error:
            for stepper in steppers:
                try:stepper.throw(error)
                except BaseException:pass
            raise
        finally:
            for stepper in steppers:stepper.close()


if __name__=='__main__':
    for folder in main():print(folder)
