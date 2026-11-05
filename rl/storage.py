"""One persistent PPO checkpoint per architecture, separate from imitation weights."""
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
from pathlib import Path

from starters.shared.model import Policy
from training.artifacts import ROOT

RL_ROOT = ROOT / 'out/rl'


def variant_name(saved):
    if saved['kind'] != 'feedforward':
        raise ValueError('Recurrent policies are excluded from PPO training')
    defaults={name:p.default for name,p in inspect.signature(Policy).parameters.items()}
    defaults.update(saved.get('model_config',{}))
    defaults.update(activation=saved.get('activation','relu'),negative_slope=saved.get('negative_slope',.01))
    digest=hashlib.sha256(json.dumps(defaults,sort_keys=True).encode()).hexdigest()[:10]
    family='residual' if defaults['residual_blocks'] else 'cnn'
    return f"{family}-c{defaults['channels']}-h{defaults['hidden_size']}-{digest}"


@contextmanager
def training_lock(folder):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'.training.lock').open('a') as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another trainer is already writing {folder}') from None
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def require_completed_pair(folder):
    """Do not let single-model training bypass an interrupted paired commit."""
    import torch
    folder=Path(folder).resolve()
    for journal in folder.parent.glob('*/.paired-commit.tmp'):
        try:
            payload=torch.load(journal,map_location='cpu',weights_only=True)
        except FileNotFoundError:
            continue  # Another, disjoint pair completed its commit after discovery.
        if folder/'policy.pt' in {Path(path).resolve() for path,saved in payload}:
            raise RuntimeError('An interrupted paired commit exists; restart rl/train_pair.py to recover both models first')
