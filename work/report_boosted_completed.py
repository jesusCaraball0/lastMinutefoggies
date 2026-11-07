import json
from pathlib import Path
import torch
ROOT=Path.cwd(); folder=ROOT/'out/experiments/boosted-cnn-28018186'; out=folder/'completed-run-analysis'
x=json.load(open(out/'snapshot-analysis.json')); b=x['boosted_best']; r=x['residual_best']
m=json.load(open(folder/'metrics.json'))
# Confirm the downloaded archive is stable and its three completed policies agree.
states={}
for name in ['best_policy.pt','last_policy.pt','final_policy.pt']:
 saved=torch.load(folder/name,map_location='cpu',weights_only=True)
 states[name]=saved['state_dict']
 x['checkpoint_checks'].setdefault(name,{})
 x['checkpoint_checks'][name].update(epoch=saved.get('epoch'),finite=all(torch.isfinite(v).all().item() for v in saved['state_dict'].values() if v.is_floating_point()))
x['selected_latest_final_tensors_identical']=all(torch.equal(v,states[name][key]) for key,v in states['best_policy.pt'].items() for name in ['last_policy.pt','final_policy.pt'])
x['wall_seconds']=m['wall_seconds']
x['analysis_script']='work/analyze_boosted_snapshot.py'
(out/'snapshot-analysis.json').write_text(json.dumps(x,indent=2))
text='''
<a id="stage-50"></a>

## 50. Boosted CNN full historical pretraining: 50 epochs and readiness for PPO (2026-11-05; completed)

### Evidence, completion, and comparability

Run: `out/experiments/boosted-cnn-28018186/`. The initial local reading caught a download in progress (48 epochs); the completed download supersedes it. Final metrics explicitly report **completed**,50/50epochs, best_epoch50. Final plots, weight/activation histograms and best/last/final checkpoints are present. All three checkpoint state dictionaries are finite and tensor-identical, at epoch50. No training or PPO was started during this analysis. This is the first completed intended boosted pretraining run; the cancelled stage47 pilot is not its matched current-only control.

Ordered train and validation split filenames were checked against residual-medium-27890681 and match exactly. Both use5350942 training moves/55250 usable games and595179 validation moves/6141 games,seed7980,50epochs,batch128,AdamW(.9,.99),eps1e-8,decay.01,clip5,warmup5820updates,LR1e-4→1e-3→1e-5. Each completed2090250optimizer updates. Boosted retains32channels/128hidden/3residual blocks/convBN/ReLU/dropout.1 but expands inputs to1819planes and parameters from850785 to1370625 (+61.1%). Thus data and training budget match; this is a combined representation/parameter-count/initialization realization comparison, not a feature-isolating ablation or replicated-seed estimate. No feature-group ablation identifies whether20-turn boards, past actions, captures, or last-seen/age supplied the gains.

Reported training wall time16687.9seconds (4.64hours), versus13678.5seconds (3.80hours) for the old residual. Hardware/data-loading differences can affect this comparison. It is an observed run time, not a throughput guarantee.

### Selected-checkpoint imitation metrics

The selection criterion remains highest legal top1 on8192 fixed held-out positions, not minimum full-validation loss. Boosted selects epoch50; old residual selects49. Full-validation raw metrics below use all595179 held-out moves. Legal metrics use the fixed8192-position probe, without conflating legality with the observation-based king-safety filter. These are imitation metrics, not win rates.

| Metric | Residual best49 | Boosted best50 |
| --- | ---: | ---: |
| Training cross entropy |2.66135|2.61881|
| Full-validation cross entropy |2.61368|2.47430|
| Full-validation top1 |31.153%|34.884%|
| Full-validation top5 |62.190%|65.741%|
| Legal probe top1 |31.519%|35.046%|
| Legal probe NLL |2.51455|2.42001|
| Probability on legal candidates |86.983%|89.440%|
| Legal distribution entropy (nats) |2.24923|2.09562|

Legal accuracy improves3.528percentage points; raw held-out accuracy improves3.731points; full-validation NLL falls.13938 (5.33%). At epoch48, the matched epoch comparison is34.853% versus31.145% raw top1 and2.47667 versus2.61503 validation NLL, so the advantage does not depend solely on best-checkpoint selection. The legal probe is a small correlated subset of moves and not8192 independent games; do not infer a game-win confidence interval from it.

| Boosted epoch | Train CE | Validation CE | Full top1 | Legal top1 |
| --- | ---: | ---: | ---: | ---: |
|1|3.17478|2.71872|31.182%|30.823%|
|10|2.77683|2.65885|32.322%|33.044%|
|20|2.75233|2.71986|32.706%|32.617%|
|30|2.70133|2.57725|33.349%|33.350%|
|40|2.53058|2.50160|34.414%|34.473%|
|45|2.54952|2.48263|34.701%|34.778%|
|48|2.49293|2.47667|34.853%|34.583%|
|49|2.50241|2.47525|34.862%|34.741%|
|50|2.61881|2.47430|34.884%|35.046%|

Learning continues late, but at diminishing gains: full-validation top1 adds.470points from40→50 and.032points from49→50. Validation loss remains improving;50epochs does not establish convergence. This is not the classic training-loss-down/validation-loss-up pattern of sustained overfitting. Training loss includes dropout and train-mode BN, while validation uses eval mode, so train>validation is not paradoxical. The last-epoch training CE rises despite improving validation; raw gradient spikes and feature scale drift justify caution rather than treating the training-loss jump alone as generalization collapse.

### Optimization: the old concern persists, with a heavier extreme tail

At boosted epoch50, pre-clipping global gradient L2 mean62.915,median34.769,p95=63.941,max278925.19. Old residual best49:mean53.251,median44.743,p95=111.486,max739.91. Boosted has lower typical gradients (median/p95), but much larger rare spikes. The largest boosted epoch-level maximum is289807.125 at46; the residual's run-wide maximum is69942.656 at50. Comparing selected-epoch maxima alone would obscure that the residual also has outliers.

Clipping affects10.20% of updates at epoch1, exceeds99% by epoch4, and reaches100% in later epochs. At50 all41805 updates are clipped; post-clipping global norm averages5.00000. Every recorded global gradient nonfinite count is0 across50epochs, and final weights/recorded activation statistics are finite. Clipping controls the update input norm; it does not eliminate the cause of scale drift, and clipping frequency alone is not evidence that training diverged.

| Boosted epoch50 layer | Mean pre-clip L2 | p95 pre-clip | Maximum pre-clip | Mean post-clip L2 |
| --- | ---: | ---: | ---: | ---: |
|Input convolution encoder.0|4.983|2.251|38485.38|.255|
|Input BN encoder.norm0|27.560|11.301|253943.22|1.100|
|Last residual conv2|5.501|8.542|2531.48|.664|
|Last residual BN2|18.499|27.457|37997.47|1.727|
|Dense encoder.5|10.443|12.018|287.40|1.428|
|Policy linear head|19.107|20.605|43.47|2.639|

The mean exceeding p95 in early layers reflects rare severe outliers. The worst input-BN spike shows that the issue is not confined to the final dense/head layers. Sampled actual optimizer update/parameter L2 medians at50 are1.083e-5/input convolution,1.549e-6/dense and2.796e-7/policy; respective p95 values1.181e-5,1.692e-6,3.061e-7. These are small late actual relative updates, despite alarming raw gradients. Sampling every100steps can miss rare updates. Policy gradient magnitude is much larger than in the old residual (mean19.107 vs.17325), but global clipping, weight scale and Adam preconditioning prevent interpreting that ratio as an equally large parameter-update ratio.

### Activation and weight histograms: mixed health, not wholesale collapse

Statistics use512 fixed validation observations, not all held-out games. First ReLU encoder.1 has56.48%zeros,mean.0128,std.0210; zeros alone are normal for ReLU. The residual trunk becomes very sparse: block0 output81.06%zeros,block1 output91.85%,final block95.53%. Final-block dormant-channel fraction is68.75% using the recorded relative-activity threshold; these are activity diagnostics, not proof those channels are permanently dead on every state.

Final-trunk largest-channel activity share improves from89.67%/old residual best to77.48%/boosted; top4 still account for99.17%. Dense output encoder.6 goes the other way: its largest unit contributes96.37% of total mean absolute activation, versus7.78% in the old residual. Its overall zero fraction is54.76%,similar to residual54.49%,which demonstrates why aggregate zero fraction alone misses scale concentration. Dense std grows.368/epoch1→1.478/20→6.908/40→21.739/50. Dominant channel72 has mean238.199,std65.004 and is positive on every sampled observation. This is substantial feature-scale drift, not a clean bill of architectural health.

Final policy logits have mean−555.13,std141.03 on these observations; the policy weight histogram is strongly negative/skewed rather than a requirement-satisfying Gaussian. Negative logits themselves are harmless: softmax depends on differences, and a state-dependent common offset cancels exactly. The outgoing weights for dense72 have mean−2.16131 and across-action std.03724,versus median across-action std2.96459 for policy columns. Much of that huge unit therefore contributes a common logit offset. A normally distributed weight histogram is not itself the learning objective, and activation mass is not predictive importance.

To test function rather than infer from histograms, the selected boosted50 checkpoint was evaluated locally in eval/inference mode on the same8192 held-out probe and then with dense channel72 set to zero by a removable forward hook. No weights, BN buffers or checkpoint files were updated:

| CPU inference probe | Intact | Dense72 zeroed |
| --- | ---: | ---: |
|Legal top1|35.022%|34.277%|
|Legal NLL|2.42006|2.48208|
|Raw top1|34.875%|34.058%|
|Legal probability mass|89.439%|88.972%|

Removing the unit costs.745percentage points of legal accuracy and.06202NLL. It is useful, but96%activation share emphatically does not mean96%of the policy's decisions come from that unit. This inference intervention is an out-of-distribution ablation, not a retrained smaller model. CPU intact legal accuracy differs from the saved CUDA probe by2/8192 positions (.0244points); NLL differs.000054. Small backend rounding differences with large logits are a plausible explanation, not a tested attribution. Report saved CUDA results for the training comparison and local CPU results for the paired intervention.

The value output histogram (mean5.33,std1.43) is **not a learned outcome predictor**: imitation only supervised policy loss. Existing PPO initialization zeroes the untrained value head for a fresh supervised checkpoint. Do not carry this arbitrary value scale into an interpretation of critic quality.

### Takeaways and next experiment

1. The combined boosted representation/model is a better historical move predictor than the prior medium residual under matched split and optimization budget. It supplies a promising starting actor; this does not establish superior live play, history-specific causality, or robustness to new opponents.
2. History did not cure gradient clipping, trunk concentration, or dense feature-scale drift. Nevertheless, loss/accuracy improve and the unit ablation argues against describing the entire actor as collapsed. More architecture changes or another long imitation run are not justified solely by histogram appearance.
3. **Proceed next to a short monitored PPO experiment, preceded by a frozen-game benchmark**, rather than immediately committing a long blind RL run. Use boosted best_policy.pt (epoch50; identical to latest/final). Benchmark against the supervised residual and established RL opponents with balanced colors and fixed seeds, then train boosted against a balanced fixed mixture of full latest2200 and residual saved-best1200. Preserve those opponents; update boosted alone initially, so opponent drift does not obscure its own progress. Do not use residual latest2200 as the only strong opponent: stage45 found it had regressed against its incumbent.
4. A400-update initial PPO budget is a reasonable next experiment, using the established lambda1 recipe as the starting reference, with evaluations at0 and regular intervals. Track fixed-opponent score, rewards/draws/length, KL/entropy/ratio clip, global gradient clipping, MC outcome RMSE/EV and feature scales. Keep the original supervised file separate and the new boosted canonical RL file plus embedded best. Stop/reassess if score or critic transfer deteriorates;400 is a diagnostic budget, not a promised convergence horizon. Critic initialization and the high-scale shared representation deserve monitoring; zero initialization does not guarantee a well-conditioned critic after its first updates.
5. Hold PSRO until there is evidence that this third actor learns useful complementary behavior. No boosted PPO, arena benchmark or PSRO run was launched by this analysis, and no new game-win metric is claimed here.

Reproducible derived analysis: `work/analyze_boosted_snapshot.py`; completed checkpoint/report verification: `work/report_boosted_completed.py`. New numerical evidence: `out/experiments/boosted-cnn-28018186/completed-run-analysis/snapshot-analysis.json`, including full per-epoch layer gradients/update ratios/activity summaries and SHA256s for best/last; derived comparison plot `training-comparison.png`. Original metrics and checkpoints are untouched. Final server-generated per-layer histograms remain in `histograms.json`, `activations/` and `weights/`.
'''
report=ROOT/'RL_EXPERIMENT_HISTORY.md';s=report.read_text()
if '<a id="stage-50"></a>' in s: raise RuntimeError('stage50 already exists; review before replacing')
s=s.replace('stage numbers30–49','stage numbers30–50')
s=s.replace('- [49. Boosted CNN server pretraining setup](#stage-49)','- [49. Boosted CNN server pretraining setup](#stage-49)\n- [50. Completed boosted CNN pretraining and PPO readiness](#stage-50)')
s+=text
report.write_text(s)
p=ROOT/'EXPERIMENT_HISTORY.md';s=p.read_text();s=s.replace('No pretraining results yet. Cluster operations:', 'Completed50-epoch results and diagnostics are recorded in [RL report stage50](RL_EXPERIMENT_HISTORY.md#stage-50): boosted legal top1=35.046% versus31.519% for the previous residual, with persistent clipping and dense feature-scale drift. This is imitation accuracy, not a game-win result. Cluster operations:');p.write_text(s)
# Discard only this analysis session's superseded, incomplete generated artifact.
stale=folder/'analysis-through-epoch48'
if stale.exists():
 for q in stale.iterdir():
  if q.name=='snapshot-analysis.json':q.unlink()
 if not list(stale.iterdir()):stale.rmdir()
print('Appended stage50; checkpoint agreement:',x['selected_latest_final_tensors_identical'])
