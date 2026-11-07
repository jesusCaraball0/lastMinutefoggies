"""Disk-backed compact turn records; construct20-turn stacks only per sample."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import chess
from starters.shared.boosted_features import BoostedHistory,ROW_SIZE,RECORD_SIZE,TURNS,MAGIC
from starters.shared.moves import ObservationTracker


def boosted_rows(replay):
    from training.datasets import replay_examples
    examples=list(replay_examples(replay))
    if not examples:return
    trackers=[ObservationTracker(True),ObservationTracker(False)]
    histories=[BoostedHistory(True),BoostedHistory(False)]
    for frame,after,(_,target) in zip(replay['frames'],replay['frames'][1:],examples):
        for side,color in enumerate(['white','black']):
            trackers[side].observe(frame[color]);histories[side].observe(frame[color],frame['ply'])
        side=frame['ply']%2
        yield histories[side].compact_row(trackers[side].rights),target
        # Happens after yield: the action target is never an input to itself.
        move=chess.Move.from_uci(after['move'])
        histories[side].record_move(move);trackers[side].record_move(move)


class BoostedReplayDataset(Dataset):
    def __init__(self,paths,cache_dir=None):
        paths=[Path(p).resolve() for p in paths]
        cache_dir=Path(cache_dir or os.environ.get('FOW_BOOSTED_CACHE_DIR',Path(__file__).resolve().parents[1]/'out/cache/boosted-v1'))
        cache_dir.mkdir(parents=True,exist_ok=True)
        source=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
        key=hashlib.sha256(json.dumps(['boosted-v1-turn20-row203',source]).encode()).hexdigest()[:24]
        self.directory=cache_dir/key;self.directory.mkdir(exist_ok=True)
        manifest=self.directory/'manifest.json'
        with (self.directory/'.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not manifest.exists():self._build(paths,manifest)
        m=json.loads(manifest.read_text());self.games=m['games'];self.positions=m['positions']
        self.rows=np.memmap(self.directory/'rows.bin',dtype=np.uint8,mode='r',shape=(self.positions,ROW_SIZE))
        self.targets=np.memmap(self.directory/'targets.bin',dtype=np.uint16,mode='r',shape=(self.positions,))
        self.starts=np.memmap(self.directory/'starts.bin',dtype=np.uint32,mode='r',shape=(self.positions,))
        self.offsets=2*np.arange(TURNS)

    def _build(self,paths,manifest):
        count=games=0
        with (self.directory/'rows.bin').open('wb') as rows,(self.directory/'targets.bin').open('wb') as targets,(self.directory/'starts.bin').open('wb') as starts:
            for path in tqdm(paths,desc='Encode boosted replays',unit='game'):
                try:
                    replay=json.loads(path.read_text());examples=list(boosted_rows(replay))
                except (ValueError,KeyError,TypeError) as error:
                    raise ValueError(f'Bad replay {path}: {error}') from error
                if not examples:continue
                records,labels=zip(*examples)
                np.asarray(records,dtype=np.uint8).tofile(rows)
                np.asarray(labels,dtype=np.uint16).tofile(targets)
                np.full(len(labels),count,dtype=np.uint32).tofile(starts)
                count+=len(labels);games+=1
        if not count:raise ValueError('No training examples in boosted split')
        temporary=manifest.with_suffix('.tmp');temporary.write_text(json.dumps(dict(schema='boosted_v1',positions=count,games=games,selected_files=len(paths),compact_row_bytes=ROW_SIZE),indent=2))
        temporary.replace(manifest)

    def __len__(self):return self.positions

    def __getitem__(self,index):
        index=int(index)
        if index<0:index+=self.positions
        if not 0<=index<self.positions:raise IndexError(index)
        ids=index-self.offsets;valid=ids>=int(self.starts[index])
        records=np.zeros((TURNS,RECORD_SIZE),dtype=np.uint8)
        records[valid]=self.rows[ids[valid],:RECORD_SIZE]
        packed=np.concatenate((np.asarray(MAGIC,dtype=np.uint8),records.reshape(-1),self.rows[index,RECORD_SIZE:]))
        return packed[None],np.asarray([self.targets[index]],dtype=np.int64)
