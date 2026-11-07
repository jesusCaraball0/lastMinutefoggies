"""Append completed2200 findings to the dedicated RL SSOT."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/fow-matplotlib')
from pathlib import Path
import json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
a=json.loads((ROOT/'out/rl/paired-lambda1-2200-analysis.json').read_text())
e=json.loads((ROOT/'out/rl/paired-lambda1-2200-evaluation.json').read_text());assert len(e['policies'])==4
out=ROOT/'out/experiments/rl2200-latest-vs-saved-best-round-robin';t=json.loads((out/'results.json').read_text());assert t['status']=='completed'
fig,axes=plt.subplots(2,2,figsize=(12,8))
for name,x in a.items():
 label='Full CNN' if name.startswith('cnn') else 'Residual';log=json.loads((ROOT/'out/rl'/name/'metrics.json').read_text());rows=log['updates']
 for ax,metric,title in [(axes[0,0],'rmse','Fresh-rollout MC RMSE'),(axes[0,1],'explained_variance','Fresh-rollout MC explained variance')]:
  xs=list(range(1199,2200,25));ys=[sum(r['ppo']['value_fit']['monte_carlo_before'][metric] for r in rows[max(0,i-49):i+1])/len(rows[max(0,i-49):i+1]) for i in xs]
  ax.plot([i+1 for i in xs],ys,label=label);ax.set_title(title+' (50-update mean)');ax.set_xlabel('Update');ax.legend()
 es=[r for r in log['evaluations'] if r['update']>=1200];axes[1,0].plot([r['update'] for r in es],[r['mean_score'] for r in es],label=label)
axes[1,0].set_title('Training checkpoint selector: 40 games/evaluation');axes[1,0].legend();axes[1,0].set_xlabel('Update');axes[1,0].set_ylim(0,1)
labels=list(t['standings']);axes[1,1].bar(range(4),[t['standings'][k]['score'] for k in labels]);axes[1,1].set_xticks(range(4));axes[1,1].set_xticklabels(labels,rotation=20,ha='right');axes[1,1].set_ylim(0,1);axes[1,1].set_title('Four-actor round-robin score (300 games/actor)')
fig.tight_layout();fig.savefig(ROOT/'out/rl/paired-lambda1-2200-comparison.png',dpi=160);plt.close(fig)
lines=['\n<a id="stage-45"></a>\n\n## 45. Paired PPO updates1201–2200: longer training and checkpoint reassessment (2026-11-05)\n',
'Both policies completed1000 more updates at unchanged lambda1,16 games/update, all other saved hyperparameters unchanged. There were16000 additional unique shared training games,35200 cumulative shared games over2200 updates. This is one ongoing paired training trajectory. No new independent training seed, architecture intervention or unchanged-opponent control was introduced. Both checkpoint files contain finite latest/best tensors, one canonical model file per variant, and unchanged original supervised source hashes. No evaluation result promoted or changed a checkpoint.\n',
'### Playing-strength benchmarks\n',
'Repeated benchmark: same seed572940, temperature0.9, max256 plies,50 games per original frozen baseline,100 games per candidate. Prior1200 numbers are the archived benchmark, not recreated1200 latest weights. Only embedded best snapshots survive alongside latest; missing intermediate weights were not reconstructed.\n',
'| Policy | Selected update | W/L/D | Score | Prior1200 score | Score standard error |\n| --- | --- | --- | --- | --- | --- |']
for name,r in e['policies'].items():lines.append(f"| {name} | {r['selected_update']} | {r['wins']}/{r['losses']}/{r['draws']} | {r['score']:.1%} | {r['previous_1200_score']:.1%} | {r['score_standard_error']:.4f} |")
lines+=['\nFullCNN latest improved only2.5 percentage points on this sample (80→82.5%), which is not a resolved gain with100 games. Residual latest dropped19.5 points (78→58.5%), large enough to warrant direct checking. Both selected bests remain earlier policies (full1170, residual1200); their repeated results exactly reproduce the prior benchmark.\n',
'### Five-seed direct round robin\n',
'Four distinct actors, six matchups,100 games per matchup, balanced50/50 colors,20 games per seed743291–743295,600 unique games,300 appearances per actor. Same game contract/temperature/max plies as prior tournament. Seeds are reused from stage44, not a previously untouched seed set.\n',
'| A | B | A W/L/D | A score | Approximate95% score interval |\n| --- | --- | --- | --- | --- |']
for r in t['matches']:
 p=r['score'];se=math.sqrt((r['wins']*(1-p)**2+r['losses']*p*p+r['draws']*(.5-p)**2)/(100*99))
 lines.append(f"| {r['a']} | {r['b']} | {r['wins']}/{r['losses']}/{r['draws']} | {p:.1%} | {max(0,p-1.96*se):.1%}–{min(1,p+1.96*se):.1%} |")
lines+=['\nIntervals use a normal approximation of the bounded per-game score. They are descriptive, unadjusted for multiple comparisons, and do not include training-seed variation. Full seed/color/outcome breakdown is in tournament JSON.\n',
'### Training outcomes and value quality\n',
'Value RMSE pools squared errors by decisions; EV is the mean of update-level EV. Metrics below are pre-update fresh-rollout MC returns. The current opponent and state distribution evolve; lower MC error is not a fixed-distribution critic generalization result.\n',
'| Model | Window | W/L/D | Score | Mean reward | MC RMSE | MC EV |\n| --- | --- | --- | --- | --- | --- | --- |']
for name,x in a.items():
 for key in ['previous50','first50','last50','last200','phase']:
  r=x[key];v=r['value_fit']['monte_carlo_before'];lines.append(f"| {name} | {key} | {r['wins']}/{r['losses']}/{r['draws']} | {r['score']:.3%} | {r['reward']:.6f} | {v['rmse']:.6f} | {v['mean_update_explained_variance']:.6f} |")
lines+=['\nprevious50=1151–1200; first50=1201–1250; last50=2151–2200; last200=2001–2200; phase=1201–2200. FullCNN phase55.575% versus residual44.425%; rewards differ slightly from zero-sum because both receive+0.1 on draws. Terminations15590 king-capturable,409 threefold and1 move-limit draw. Draw rate410/16000=2.5625%; one capped game alone does not establish cap exploitation.\n',
'### Optimization, features and complete phase summaries\n']
for name,x in a.items():
 lines+=['\n#### '+name+'\n','```json\n'+json.dumps({k:x[k] for k in ['settings','best_update','phase','last50','last200','features1200','features2200','checkpoint_sha256','original_source_sha256','finite_latest_and_best','canonical_model_files','recipe_changes']},indent=2)+'\n```\n']
lines+=['\n### Takeaways\n',
'1. Longer paired training did not improve both models uniformly. FullCNN remains strong against the frozen baselines; residual latest regressed on the repeated benchmark. Relative training scores increasingly favor fullCNN: its last two100-update chunks scored60.53% and60.16%, compared with50.75% in the first100 updates after1200. These are changing-peer matches, not an absolute rating.\n',
'2. Critics improved only modestly in the final50 window: full RMSE0.8376→0.8085, EV0.0253→0.0387; residual RMSE0.8305→0.8017, EV0.0174→0.0222. Last200 EV remains0.0289/full and0.0153/residual, illustrating noisy estimates. This is still weak outcome prediction. Actor gains and value accuracy are not moving together reliably.\n',
'3. Frequent clipping persists at100% of optimizer steps. Late KL~0.00399/full and0.00970/residual, ratio clipping~1.74%/4.75%, entropy1.402/1.373 nats. This phase had6/full and34/residual KL-stop updates, raw global maxima961.3/5890.8, and no nonfinite tensors. These are optimization constraints to monitor, not proof of numerical divergence. Do not raise the clipping threshold solely to avoid reported clipping.\n',
'4. Feature concentration persists: full second-conv top4 activity80.25%, dense top4 91.37%; residual final-trunk top1 89.23%, dormant46.875%. Some residual channels became less dormant, but this did not guarantee improved external performance.\n',
'5. Best-model selection remains noisy. Neither model established a new training-selector high after1200 despite100 additional40-game evaluations. The existing latest and incumbent weights remain independently preserved. Finite training and stable entropy/KL do not guarantee monotonic playing-strength gains.\n',
'### Next steps\n',
'Prioritize a reliable direct challenger-versus-incumbent checkpoint gate and multi-opponent evaluation before another long unchanged run. Preserve residual best1200 for deployment/comparison while retaining latest2200 for any intentional continuation. Use the direct-match results above to qualify whether regression is supported; do not automatically roll back optimizer state or replace latest with a selected actor.\n',
'If subsequent evaluation confirms residual degradation, a small controlled continuation against a frozen opponent (or a simple mix of current and frozen incumbent opponents) would test whether chasing a moving peer is hurting generality. This is a hypothesis, not a demonstrated mechanism, and does not require PSRO machinery. Keep a matched game budget and change one factor. A larger network or more critic epochs is not supported by these results; fresh-game critic data scaling remains a separate possible experiment.\n',
'No further training, promotion, checkpoint reset or opponent-pool change was performed. Historical recommendations to simply train longer are now qualified by this nonuniform outcome.\n',
'### Artifacts\n',
'- [Complete2200 analysis](out/rl/paired-lambda1-2200-analysis.json), including ten100-update windows and all in-loop evaluations.\n- [Repeated evaluation](out/rl/paired-lambda1-2200-evaluation.json).\n- [Four-actor tournament](out/experiments/rl2200-latest-vs-saved-best-round-robin/results.json).\n- [Comparison plot](out/rl/paired-lambda1-2200-comparison.png).\n- Canonical per-model metrics retain all2200 updates and feature probes. Training logs logs/ppo-pair-1201-2200.out/.err; evaluation logs logs/ppo-pair-2200-evaluation.out/.err and logs/rl2200-round-robin.out/.err; errors empty. Scripts work/analyze_pair_2200.py, work/evaluate_pair_2200.py, work/report_pair_2200.py, work/checkpoint_round_robin.py. Older400/800/1200 evidence remains unchanged.\n']
p=ROOT/'RL_EXPERIMENT_HISTORY.md';s=p.read_text();assert '## 45. Paired PPO' not in s;s=s.replace('stage numbers30–44','stage numbers30–45');s=s.replace('9. **Next priorities:**','9. **Latest2200 continuation:** further training was nonuniform: repeated benchmark82.5% full versus58.5% residual (previous80%/78%). Read stage45 for the five-seed direct check and updated recommendations.\n10. **Next priorities:**');s=s.replace('## Chronological findings','- [45. Updates1201–2200 and checkpoint reassessment](#stage-45)\n\n## Chronological findings',1);p.write_text(s+'\n'.join(lines)+'\n')
print('Appended stage45 and comparison plot')
