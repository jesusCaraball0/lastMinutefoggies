"""Outcome labels for the existing boosted inputs; labels never enter observations."""
import fcntl
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from training.boosted_dataset import BoostedReplayDataset

GAMEPLAY_REASONS={'King capturable','No legal moves','Threefold repetition','50-move rule','Move limit reached'}


def outcome_label(replay,white):
    if replay.get('rulesVersion')!='fow-v1' or replay.get('failedSide') or replay.get('reason') not in GAMEPLAY_REASONS:
        return 0.,False
    result=replay.get('result')
    if result=='1/2-1/2':return .1,True
    if result not in ('1-0','0-1'):return 0.,False
    return (1. if (result=='1-0')==white else -1.),True


class BoostedOutcomeDataset(BoostedReplayDataset):
    """Reuse the exact policy-only cache and split; mask unreliable outcome labels."""
    def __init__(self,paths,cache_dir=None):
        paths=list(paths);super().__init__(paths,cache_dir)
        directory=self.directory/'outcome-v1';directory.mkdir(exist_ok=True)
        manifest=directory/'manifest.json'
        with (directory/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not manifest.exists():self._build_outcomes(paths,directory,manifest)
        self.outcome_metadata=json.loads(manifest.read_text())
        self.outcome_positions=self.outcome_metadata['valid_positions']
        self.outcomes=np.memmap(directory/'values.bin',dtype=np.float32,mode='r',shape=(self.positions,))
        self.outcome_valid=np.memmap(directory/'valid.bin',dtype=np.uint8,mode='r',shape=(self.positions,))

    def _build_outcomes(self,paths,directory,manifest):
        offset=valid_count=0;total=squares=0.;excluded_games=0
        with (directory/'values.tmp').open('wb') as values,(directory/'valid.tmp').open('wb') as validity:
            for path in tqdm(paths, desc="Read outcome labels", unit="game"):
                replay=json.loads(Path(path).read_text())
                if replay.get('result') not in ('1-0','0-1','1/2-1/2'):continue
                frames=replay['frames'][:-1];n=len(frames)
                if not n:continue
                if offset>=self.positions or int(self.starts[offset])!=offset:
                    raise ValueError('Outcome and boosted move cache game boundaries disagree')
                labels=[outcome_label(replay,f['ply']%2==0) for f in frames]
                y=np.asarray([a for a,b in labels],dtype=np.float32)
                mask=np.asarray([b for a,b in labels],dtype=np.uint8)
                y.tofile(values);mask.tofile(validity)
                valid_count+=int(mask.sum());total+=float(y[mask.astype(bool)].astype(np.float64).sum())
                squares+=float(np.square(y[mask.astype(bool)].astype(np.float64)).sum())
                excluded_games+=int(not mask.any());offset+=n
        if offset!=self.positions:raise ValueError('Outcome count does not match boosted observations')
        (directory/'values.tmp').replace(directory/'values.bin');(directory/'valid.tmp').replace(directory/'valid.bin')
        report=dict(schema='boosted_outcome_v1',positions=offset,valid_positions=valid_count,masked_positions=offset-valid_count,masked_games=excluded_games,target_mean=total/valid_count if valid_count else None,target_second_moment=squares/valid_count if valid_count else None,targets='own-side terminal outcome: win+1,loss-1,draw+0.1; undiscounted',eligibility='fow-v1, recognized gameplay termination, no bot failure')
        temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2));temp.replace(manifest)

    def __getitem__(self,index):
        tokens,moves=super().__getitem__(index)
        return tokens,dict(moves=moves,outcome=np.asarray([self.outcomes[index]],dtype=np.float32),outcome_valid=np.asarray([self.outcome_valid[index]],dtype=bool))


class OutcomeMetrics:
    def __init__(self):
        self.n=0;self.y=self.yy=self.p=self.pp=self.e=self.ee=self.a=0.;self.decisive=self.correct=0
    def add(self,predictions,targets):
        p=predictions.detach().double();y=targets.detach().double();e=p-y
        if not torch.isfinite(p).all():raise FloatingPointError('Nonfinite outcome prediction')
        self.n+=y.numel();self.y+=float(y.sum());self.yy+=float(y.square().sum());self.p+=float(p.sum());self.pp+=float(p.square().sum());self.e+=float(e.sum());self.ee+=float(e.square().sum());self.a+=float(e.abs().sum())
        mask=y.abs()==1;self.decisive+=int(mask.sum());self.correct+=int(((p[mask]>=0)==(y[mask]>0)).sum())
    def report(self):
        if not self.n:return None
        n=self.n;vy=max(0.,self.yy/n-(self.y/n)**2);ve=max(0.,self.ee/n-(self.e/n)**2)
        return dict(positions=n,mse=self.ee/n,rmse=(self.ee/n)**.5,mae=self.a/n,explained_variance=1-ve/vy if vy>1e-12 else None,prediction_mean=self.p/n,prediction_std=max(0.,self.pp/n-(self.p/n)**2)**.5,target_mean=self.y/n,target_std=vy**.5,win_loss_accuracy=self.correct/self.decisive if self.decisive else None,win_loss_positions=self.decisive)
