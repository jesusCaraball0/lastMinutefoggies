"""Append completed supervised findings after inspecting retained run artifacts."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'out/experiments/boosted-outcome-cnn-28094904/analysis'
x=json.loads((OUT/'comparison.json').read_text());a=x['best'];f=x['final'];b=x['baseline_best'];o=a['validation']['outcome'];fo=f['validation']['outcome']
section='''
<a id="stage-58"></a>

## 58. Completed medium boosted outcome pretraining: historical value learning and matched move-only comparison (2026-11-06)

### Completion, data and comparison contract

Run `out/experiments/boosted-outcome-cnn-28094904`:status completed,50/50epochs. Best policy selected at43 by the unchanged8192-position held-out legal-move accuracy probe. Final/last checkpoint is50;final histograms/plots are present. Best43,last50,final50 checkpoints all load with finite tensors andvalue_pretrained=true/outcome_supervision=true. The currently running64/256wide job28107503 is not part of these findings;its downloaded31-epoch snapshot is neither a completed run nor evidence of termination.

Matched reference:boosted-cnn-28018186,ordinary move-only boosted32/128,best50. Both have1370625parameters,1819input planes,32channels/128hidden,3residual blocks,convBN/ReLU/dropout.1. Ordered train/validation filenames match exactly;parameter count,seed,optimizer/schedule,global clip,batch size,total updates anddata/probe counts were compared and match. Same full5350942train moves/55250usable games and595179validation moves/6141games;50epochs,2090250updates,batch128,seed7980,AdamWbetas(.9,.99)/eps1e-8/decay.01,globalclip5,start1e-4/peak1e-3/floor1e-5,5820warmupupdates then cosine. The previous setup tests verified identical same-seed initialized weights with the outcome flag toggled. This is one training seed;there is no training-seed uncertainty estimate.

Treatment:moveCE+.5*outcomeMSE jointly updates encoder,policy andvalue heads. Outcome targets are own-side **undiscounted** terminalwin+1/loss−1/draw+.1,not hidden state inputs orcurrent-policy discounted returns. Eligible outcomes:5332962train and592666validation positions. Outcome-only masks exclude17980train positions/190games and2513validation positions/27games with unreliable/incompatible/failure outcomes;their existing valid move examples remain. Thus move data stay matched. Historical outcome statistics below are not reinforcement-learning results.

Training walltime18191.1seconds(5.05hours),versus16687.9seconds(4.64hours)for ordinary boosted,about9.0%longer in recorded walltime. This is a descriptive comparison,not a controlled hardware-throughput benchmark. No new training orarena games were launched during analysis.

### Policy prediction: close to move-only,small selected-checkpoint cost

Full-validation metrics cover595179moves;legalmetrics cover the fixed8192position probe. Classification accuracy is matching the recorded historical move,not game-winning accuracy.

|Metric|Move-only best50|Joint selected43|Joint final50|
|---|---:|---:|---:|
|Train move CE|2.61881|2.51719|2.53456|
|Validation move CE|2.47430|2.49157|2.47568|
|Validation raw top1|34.884%|34.552%|34.767%|
|Validation raw top5|65.741%|65.469%|FINAL_TOP5|
|Legal probe top1|35.046%|34.851%|34.607%|
|Legal probe NLL|2.42001|2.43204|FINAL_LEGAL_NLL|
|Probability on legal candidates|89.440%|88.780%|FINAL_LEGAL_MASS|

At selected checkpoints,joint legal accuracy is.195percentage points lower(16/8192positions)andfull raw accuracy.332points lower. At matched final50,the full raw difference shrinks to.118points andCE difference to.00138. These small move differences do not establish stronger/weaker game play;the probe is correlated within games andhas been repeatedly used for selection. At matched43,joint full raw accuracy34.552%versusmove-only34.484%,andlegal34.851%versus34.521%;the selected-best comparison is not the only view of the trajectory.

Final50 has minimum validationCE andmaximumfull-validation rawaccuracy,but43 wins the legal selector. Minimum outcomeMSE occurs49(.712748). These different rankings are expected for different metrics and noisy subsets;43 being selected does not establish late training collapse or that50 is universally worse. Keep the selection protocol unchanged for comparisons rather than silently choosing a new criterion after observing results.

### Outcome prediction: a meaningful historical signal

|Outcome metric|Train43|Validation43|Train50|Validation50|
|---|---:|---:|---:|---:|
|MSE|.717284|.715806|.710501|.713818|
|RMSE|.846926|.846053|.842912|.844878|
|MAE|.743538|.745874|.737497|.741160|
|Explained variance|.216716|.221705|.224123|.224011|
|Win/loss sign accuracy,draws excluded|69.051%|69.096%|69.465%|69.181%|

Validation training-mean constant predictor:MSE.919615,RMSE.958966,EV0. Selected43 reduces MSE by22.16%relative tothatconstant;final50 reduces it22.38%. Final43 predictionmean−.004807 versus targetmean+.003826,predictionstd.431604 versus targetstd.958966. The mean bias is small,so this improvement is not merely learning the average outcome. Win/loss accuracy is evaluated on544552decisive held-out positions;majority-label accuracy isabout50.23%. Remaining48114eligible positions are fromdraws. None of these percentages is a game win rate,three-class probability calibration,or a causal estimate ofposition quality.

Train values are online measurements spanning changing weights,dropout andtrain-modeBN;validation isfixed-model eval-mode. Close train/validation errors are reassuring butnot an exact fixed-model train-versus-validation generalization comparison. Full held-out games are unseen by the optimizer,but repeated experimentation/selection has adapted to thisvalidation split;there is no new untouched test set orgame-cluster confidence interval here. Near-terminal positions may be much easier than early/midgame positions;aggregateEV does not establish useful long-horizon credit prediction.

|Joint epoch|Validation moveCE|Validation raw top1|Legal top1|Outcome validation RMSE|Outcome validation EV|
|---|---:|---:|---:|---:|---:|
|1|2.73313|30.989%|31.372%|.878609|.169386|
|10|2.87062|31.361%|31.860%|.884482|.155795|
|20|2.62081|32.525%|32.019%|.868014|.182142|
|30|2.57537|33.347%|33.301%|.871075|.180900|
|40|2.50377|34.345%|34.595%|.849944|.215198|
|43|2.49157|34.552%|34.851%|.846053|.221705|
|50|2.47568|34.767%|34.607%|.844878|.224011|

No sustained late outcome-validation divergence appears:validation andtraining improve at broadly similar rates. This is stronger evidence for learning useful historical outcome features than the prior frozen-encoder head-only attempts,butdifferent labels,policy distributions andtraining budgets prevent direct causal comparisons. Move-only boosted did not superviseitsvaluehead atall;itsarbitrary value output is not a fair trained-critic baseline.

### Activation health: concentration improved substantially,not eliminated

Statistics below use the same512validation-observation probe,not theentireheld-out distribution. Largest-unit share means share oftotalmeanabsoluteactivity,not predictive importance. Final-epoch histograms arefrom50,whereasselected43statistics areinmetrics.json.

|Metric|Move-only best/final50|Joint selected43|Joint final50|
|---|---:|---:|---:|
|Final-trunk largest-channel activity share|77.48%|55.68%|56.99%|
|Final-trunk zero fraction|95.53%|96.61%|96.51%|
|Final-trunk dormant-channel fraction|68.75%|62.50%|65.625%|
|Dense largest-unit activity share|96.37%|46.70%|52.99%|
|Dense activation std|21.7387|.9671|1.2440|
|Dense zero fraction|54.76%|54.02%|53.33%|
|Dense dormant-unit fraction|11.72%|14.84%|14.84%|

This is a substantial reduction of the previously extreme dense scale/outlier concentration withnear-preservedmoveaccuracy. It does not prove the representation is healthy everywhere:oneunitstillcarriesabout half ofdenseactivity,andthefinaltrunkremainsvery sparse. Dense dormancy did not uniformly improve. ReLUzeros alone are not evidence of failure;large activity share is not equivalent todecision importance,as the earlier channel72ablation demonstrated.

Selected43policy raw-logit mean−75.93/std20.03 versusmove-only50−555.13/std141.03;finaljoint50mean−83.82/std25.88. Reduced raw scale is descriptive,not a direct quality metric;common logit offsets cancel under softmax. Histograms namedvaluecapturethelinear head **before tanh**,so theirstd(.817at43,.831at50)differsfromfinaloutcome-predictionstd(.432/.442). Thebounded predictions andfull-validation outcomesummary aretheappropriatecriticmetrics.

### Gradients:joint supervision did not cure clipping orrare spikes

Selected43pre-clipglobalL2 mean53.222,median41.952,p95=110.857,max74320.03;final50mean61.533,median40.180,p95=99.047,max199584.75. Move-only50:mean62.915,median34.769,p95=63.941,max278925.19. Joint run-wide maximum322113.47(epoch32),versusmove-only289807.13(epoch46). Theaddedobjectivechanges gradients,so this isnot an equal-loss gradient comparison. Joint typicalmedian/p95gradientsarenot uniformlylower,andtheextreme tailremains.

Clippingfraction11.91%at1,59.35%at2,97.93%at3,99.993%at4,and100%atselected43/final50. Post-clipping globalnorm averages5. All recordedglobalgradient nonfinitecountsare0;loadedselected/finalweightsarefinite. Do not infer numericaldivergence from clippingfrequency alone,nor claim thejointobjective solved optimizationstability because denseactivationstd improved.

|Layer,joint43|Mean pre-clipL2|p95pre-clip|Maximum pre-clip|
|---|---:|---:|---:|
|Input convolution|2.179|2.340|11409.41|
|Input BatchNorm|9.763|12.739|52231.93|
|Final residual BN2|31.469|74.924|7927.25|
|Dense encoder.5|11.662|14.241|196.63|
|Policy head|.900|.998|2.61|
|Value head|.715|1.714|4.75|

At50,sampledmedianactualupdate/weightL2:inputconv9.719e-6,dense1.546e-6,policy2.889e-7,value2.855e-6. Smalllateactualupdatescoexistwithrarelargegradientspikes;100-stepsamplingcanmissrareupdates. Full layergradient/update tracesremain innewanalysisJSONandoriginalmetrics.

### Findings,takeaways,andnextdirections

1. **Joint supervised outcome learningworked on historical held-out games.** EV~.22,~69%decisive-position accuracy anda~22%MSEimprovementoveraconstant demonstrate an outcome signal beyondmean calibration. This makes the new model a promising PPO initialization;itdoesnotshowthatcurrent-policycriticEVwillbe.22.
2. **Themove-predictiontradeoff is small in this run.** Joint selectedcheckpoint trailsmove-onlybestby.195legalpoints;finalfullrawaccuracy isonly.118points lower. There isnohistorical-moveaccuracyreasonheretodiscardjointpretraining,butnogame-strengthclaimwithoutplay.
3. **Dense concentration/scale improved markedly whileclipping/spikespersisted.** This separatesrepresentationimprovementsfroma blanketclaimofstability. Onecommonbottleneckreadoutstilldominatesmuchactivity;widercapacitycouldhelp,butthependingwidejobmustfinishbeforedrawingconclusions.
4. Nextcomparethecompleted64/256outcomerunwiththis32/128jointmodelonsamecriteria:policyCE/raw/legalaccuracy,outcomeMSE/EVversusconstant,train/validationcurves,andactivation/gradienthealth. Do notcallanunfinisheddownloadakilledjoborcompareits31epochsnapshotasafull50epochresult.
5. Thecriticalnexttestisfresh-game transfer:benchmarkthejointactoragainsttheestablishedfrozenfull2200/residualbest1200mixture,measure**current-policy discounted-return**criticRMSE/EV,andthencompareequal400-updatePPOlearningbudgetswiththeoldboostedprotocol. Preservepretrainedvalueweights(theimplementedflagdoes)ratherthanzeroingthem. Historicalundiscountedlabelsandcurrentdiscountedreturnsdiffer;headadaptationandcalibrationareexpected. No PPO orarena was launchedbythisanalysis.
6. Keepcheckpoint43asthestandardpolicy-selectedcandidateand50asaseparatelyidentifiedfinalcandidateiflaterbenchmarked;do notsilentlyrelabel50asbestbecauseitsoutcome/rawmetricslookbetter. Moreseedsandgame-leveluncertaintywouldstrengthenclaims. Breakdownbygamephase/remainingdecisionswouldtestwhetheroutcomeaccuracyismostlynear-terminalsignal.

Reproducibility:`work/analyze_boosted_outcome_pretrain.py`and`work/report_boosted_outcome_pretrain.py`. Newderivedartifacts:`out/experiments/boosted-outcome-cnn-28094904/analysis/comparison.json`and`joint-vs-move-only.png`,includingmatched-split/recipechecks,threecheckpointSHA256/finitechecks,selected/finalmetrics,allper-epochgradient/update/activitysummariesandbothcurves. Originalcheckpoints/metrics/histogramsareuntouched. Measuredsupervisedresultsaremaintainedhere;subsequentPPOresultsbelonginRL_EXPERIMENT_HISTORY.md.
'''
section=section.replace('FINAL_TOP5',f"{100*f['validation']['top5']:.3f}%").replace('FINAL_LEGAL_NLL',f"{f['legal']['legal_nll']:.5f}").replace('FINAL_LEGAL_MASS',f"{100*f['legal']['legal_probability_mass']:.3f}%")
p=ROOT/'EXPERIMENT_HISTORY.md';s=p.read_text()
if '<a id="stage-58"></a>' in s:raise RuntimeError('Stage58 already written; review before modifying')
s=s.replace('- [57. Wide outcome baseline reduced to64/256 (2026-11-06; setup revision, not launched)](#stage-57)','- [57. Wide outcome baseline reduced to64/256 (2026-11-06; setup revision, not launched)](#stage-57)\n- [58. Completed medium boosted outcome pretraining](#stage-58)')
s+=section;p.write_text(s)
print('Measured supervised stage58 appended')
