"""Mix arbitrary protocol bots, frozen neural checkpoints and past selves."""
import copy
from pathlib import Path
import shlex

from rl.game import Command
from rl.policy import PolicyAgent, load_policy


class OpponentPool:
    def __init__(self, checkpoints=(), commands=(), device='cpu'):
        self.device = device
        self.static = []
        self.cache = {}
        self.static_paths = {Path(p).resolve() for p in checkpoints}
        for i,path in enumerate(checkpoints):
            self.static.append(self.checkpoint(path))
        for i,command in enumerate(commands):
            self.static.append((f'command-{i+1}', Command(tuple(shlex.split(command)))))

    def checkpoint(self, path):
        path = Path(path).resolve()
        if path not in self.cache:
            model,saved = load_policy(path,self.device,best=True)
            self.cache[path]=(model,saved.get('inference_temperature',.9))
        model,temperature=self.cache[path]
        return (str(path),lambda:PolicyAgent(model,temperature))

    def training(self, learner, self_play=False, history=(), temperature=.9):
        active=self.static_paths
        self.cache={p:v for p,v in self.cache.items() if p in active}
        pool=list(self.static)
        if self_play:
            frozen=copy.deepcopy(learner).eval()
            for p in frozen.parameters():
                p.requires_grad_(False)
            pool.append(('self-current-rollout-snapshot',lambda:PolicyAgent(frozen,temperature)))
            for snapshot in history:
                past=copy.deepcopy(learner).eval()
                past.load_state_dict(snapshot['state_dict'])
                for p in past.parameters():p.requires_grad_(False)
                pool.append((f"self-update-{snapshot['update']}",
                             lambda past=past,t=snapshot.get('inference_temperature',temperature):PolicyAgent(past,t)))
        if not pool:
            raise ValueError('Supply checkpoint/command opponents or enable self-play')
        return pool
