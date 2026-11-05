import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fow-matplotlib')
import json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'out/experiments/current-checkpoints-round-robin-update1200'
d=json.loads((out/'results.json').read_text());assert d['status']=='completed'
names=sorted(d['standings'],key=lambda k:d['standings'][k]['score'],reverse=True)
mat=np.full((len(names),len(names)),np.nan)
for r in d['matches']:
 i,j=names.index(r['a']),names.index(r['b']);mat[i,j]=r['score'];mat[j,i]=1-r['score']
fig,ax=plt.subplots(figsize=(10,8));im=ax.imshow(mat,vmin=0,vmax=1,cmap='RdYlGn')
ax.set_xticks(range(len(names)));ax.set_xticklabels(names,rotation=35,ha='right');ax.set_yticks(range(len(names)));ax.set_yticklabels(names)
for i in range(len(names)):
 for j in range(len(names)):
  ax.text(j,i,'—' if i==j else f'{mat[i,j]:.1%}',ha='center',va='center')
ax.set_title('Row policy score against column policy\n100 games per matchup; colors balanced; five seeds')
fig.colorbar(im,ax=ax,label='Score (win=1, draw=0.5)');fig.tight_layout();fig.savefig(out/'score-matrix.png',dpi=150);plt.close(fig)
lines=['\n## Stage 44 — Current checkpoint round robin at update1200 (2026-11-05)\n',
'Scope: current fullCNN and medium-residual supervised baselines plus latest and embedded best RL actors. Inventory found269 .pt files including historical ablations, intermediate checkpoints and nonpolicy tensors; user was asked whether the archive should also be included. This completed tournament covers current checkpoints only. Residual latest1200 and best1200 have identical actor hashes and are represented once, with the alias retained. Five distinct actors,10 matchups,1000 unique games. No training or best-checkpoint promotion performed.\n',
'Existing canonical referee and PolicyAgent contract; temperature0.9, max256 plies. Each pair plays100 games (50 per color),20 per seed743291–743295. These seeds differ from the earlier572940 benchmark. Match results include color breakdowns, termination reasons, lengths and seed-level scores.\n',
'| Policy | W/L/D | Games | Aggregate score |\n| --- | --- | --- | --- |']
for name in names:
 s=d['standings'][name];lines.append(f"| {name} | {s['wins']}/{s['losses']}/{s['draws']} | {s['games']} | {s['score']:.2%} |")
lines+=['\n| A | B | A W/L/D | A score | Approximate score SE |\n| --- | --- | --- | --- | --- |']
for r in d['matches']:
 p=r['score'];se=math.sqrt((r['wins']*(1-p)**2+r['losses']*p*p+r['draws']*(.5-p)**2)/(100*99))
 lines.append(f"| {r['a']} | {r['b']} | {r['wins']}/{r['losses']}/{r['draws']} | {p:.1%} | {se:.4f} |")
lines+=['\nAggregate ranking weights each of the other four actors equally; it is specific to this opponent set, not an absolute skill estimate. Pairwise samples of100 games leave uncertainty, so small differences do not establish superiority. No checkpoint was promoted using these post-hoc results. All games completed without bot failure. Residual latest/best is one actor, not two independent competitors.\n',
f"Wall time{d['wall_seconds']:.1f}s. Full results and roster hashes: out/experiments/current-checkpoints-round-robin-update1200/results.json; chart: score-matrix.png. Reproduction scripts: work/checkpoint_round_robin.py, work/current-checkpoint-roster.json, work/report_checkpoint_round_robin.py. Logs: logs/current-checkpoints-round-robin.out/.err.\n"]
p=root/'EXPERIMENT_HISTORY.md';assert '## Stage 44 —' not in p.read_text();p.write_text(p.read_text()+'\n'.join(lines))
print('\n'.join(lines[:20]))
