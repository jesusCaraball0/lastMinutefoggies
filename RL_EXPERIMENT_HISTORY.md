# Fog-of-War Chess: RL Experiment History

Updated **2026-11-06 (America/New_York)**. This is the **single source of truth for RL experiments, evaluation results, and takeaways**. The companion [supervised experiment report](EXPERIMENT_HISTORY.md) covers imitation training and the baseline models. This report preserves the original chronological stage numbers30–60 so existing references remain meaningful. Supervised stages49–50 and55–57 link to their complete records in the companion model experiment report. Historical proposals and launch statuses are retained; later completed results supersede them.

Raw JSON, checkpoints, and replays remain the numerical evidence. Tables summarize measured results; links identify complete metric traces and artifacts. No historical weights or measurements have been reconstructed. The three400-update phases are successive portions of the same paired training trajectory, not independent training replicates.

## Current takeaways

1. **PPO improved both actors against their supervised starting baselines.** In the five-seed current-checkpoint tournament, the three distinct RL actors scored69–79% against the supervised opponents. This establishes useful improvement within the tested opponent pool, not general strength or convergence.
2. **No RL checkpoint is a clear winner.** Direct100-game matches: full latest1200 versus full best1170 was45W52L3D; full latest versus residual1200 was44W52L4D; full best versus residual was49W49L2D. Overall scores63.0%,61.625%,60.25% are close. Residual latest and saved-best are identical weights and count once.
3. **“Best” is the winner of the training-time checkpoint selector, not a guaranteed strongest model.** That selector uses40 games against frozen baselines. The later repeated100-game benchmark and five-seed round robin can rank candidates differently. Direct incumbent challenges were discussed but have not replaced the implemented selector. No tournament result promoted or overwrote a checkpoint.
4. **Critics remain weak while actors improve.** At1200, late fresh-rollout outcome EV was0.0253/full and0.0174/residual. Low value EV is a concern, but neither proof of numerical failure nor a reason by itself to enlarge the actor.
5. **Historical head-only warmup mainly calibrated mean predictions.** About81%/77% of the historical held-out MSE improvements came from reduced squared bias. Strong transfer to current-policy returns was not established.
6. **Independent critic fitting on a small fixed dataset overfit.** With320 training games, validation selected epochs2/full and1/residual. Further epochs reduced training error while worsening validation. Neither fitted critic clearly beat the existing critic on96 untouched test games. This experiment did not test removing actor-gradient interference during online PPO.
7. **Lambda1 remains provisional.** Latest benchmark scores improved71.5→80%/full and63.5→78%/residual during800→1200, but there was no matched lambda0.95 continuation. Extra training and lambda effects are confounded. Value loss changes across this boundary are not comparable because lambda1 targets full discounted returns.
8. **Frequent gradient clipping is not sufficient evidence of divergence.** Every PPO optimizer step was clipped, but KL stayed modest and tensors remained finite. Inherited feature concentration persists; its causal effect was not isolated.
9. **Next priorities:** reliable multi-seed checkpoint comparisons; more independent fresh games and early stopping for critic experiments; controlled changes with matched budgets. Continue training from latest while preserving the independently selected incumbent. No additional training is authorized or launched by this documentation update.

## Score definitions and tournament accounting

For any evaluated policy:

**Score = (wins + 0.5 × draws) / games.** Losses contribute zero. Win rate is wins/games and is different when draws occur.

Each of the five distinct policies in the current round robin played the other four100 times: **400 games per policy**. Overall score pools those400 outcomes. Since all matchups have equal sizes, this is also the arithmetic mean of the four matchup scores. Self-matches are excluded, and identical residual latest/best weights are represented once.

Example: residual RL recorded249 wins,145 losses,6 draws. Its score is **(249 + 0.5×6)/400 = 63.0%**. Its win rate is249/400=62.25%. Its four matchup scores are79%,69%,54%,50%, whose mean is63%.

There are10 distinct pairs and100 games per pair, so **1000 unique tournament games**. The five policies' individual400-game totals sum to2000 appearances because each game appears once for each participant. Overall score is specific to this opponent pool; it is neither an Elo rating nor a universal win probability. Small ranking differences remain uncertain.

Training reward uses a different convention: **win+1, loss−1, draw+0.1**. Mean reward is(W−L+0.1D)/N. Neither positive draw reward nor more draws proves unexploitable play.

## Experiment progression at a glance

| Stage | Work | Main result |
| --- | --- | --- |
|31–35|PPO implementation, diagnostics, persistence, paired updates|Canonical game reused; both agents update from their own decisions; one RL file per variant|
|36|Updates1–400,6400 shared games|Latest frozen-baseline benchmark66% full,60.5% residual; weak critics|
|38–39|One historical value-head epoch, then401–800|Historical RMSE improved; latest benchmark71.5%/63.5%; limited critic transfer|
|40|Literature review|Separate critic fitting and lambda comparison proposed; hypotheses, not results|
|41|Frozen-policy fresh-game critic diagnostic|512 games; independent critic overfit; no clear improvement over existing critic|
|42–43|Lambda1,801–1200,6400 shared games|Latest repeated benchmark80%/78%; critics still weak; no lambda control|
|44|Five-policy round robin|1000 games, five seeds; RL beats supervised; RL actors closely matched|

Benchmark percentages across400/800/1200 above use the same572940 evaluation seed and frozen opponents; the stage44 tournament uses five different seeds. They are distinct evaluations. In-loop best selection uses yet another protocol and must not be conflated with either.

## Latest continuation through2200

The1000-update continuation completed: latest repeated benchmark82.5% fullCNN and58.5% residual, versus80%/78% at1200. FullCNN's2.5-point change is unresolved on100 games; residual's19.5-point benchmark decline is reinforced by a five-seed direct loss to its saved-best (36W61L3D). Full latest beat its saved-best58W37L5D. Critics improved modestly but remain weak. See [stage45](#stage-45) for complete metrics and revised next steps. Earlier current summaries and stage44 rankings describe their original checkpoints.

- [49. Boosted CNN server pretraining setup](EXPERIMENT_HISTORY.md#stage-49)
- [50. Completed boosted CNN pretraining and PPO readiness](EXPERIMENT_HISTORY.md#stage-50)
- [51. Balanced boosted PPO launch](#stage-51)
- [52. Completed boosted PPO400 results](#stage-52)
- [53. Fresh-game boosted value-head-only launch](#stage-53)
- [54. Completed value-head-only400 results](#stage-54)
- [55. Boosted outcome CNN joint-pretraining setup](EXPERIMENT_HISTORY.md#stage-55)
- [56. Wide boosted outcome CNN setup](EXPERIMENT_HISTORY.md#stage-56)
- [57. Revised wide baseline:64/256](EXPERIMENT_HISTORY.md#stage-57)

- [59. Outcome-pretrained boosted PPO400 launch](#stage-59)

## Chronological findings

- [30. Readiness to move from imitation baselines to PPO (2026-11-04)](#stage-30)
- [31. PPO infrastructure implementation and CPU pipeline validation (2026-11-04)](#stage-31)
- [32. PPO draw bonus and expanded learning diagnostics](#stage-32)
- [33. One persistent RL checkpoint per model variant](#stage-33)
- [34. Training both CNN variants: current capability and literature](#stage-34)
- [35. Minimal paired PPO: both CNNs learn from the same games](#stage-35)
- [36. Full CNN and medium residual: 400 paired PPO updates](#stage-36)
- [37. Assessing value pretraining on historical games](#stage-37)
- [38. Historical value-head warmup, then PPO updates401–800](#stage-38)
- [39. Completed historical value warmup and paired PPO through update800](#stage-39)
- [40. Literature review and proposed critic experiments (2026-11-05; not run)](#stage-40)
- [41. Frozen update800 fresh-game critic diagnostic (2026-11-05; completed)](#stage-41)
- [42. Paired PPO continuation with lambda1, updates801–1200 (2026-11-05; launched)](#stage-42)
- [43. Paired PPO lambda1 continuation through1200 (2026-11-05; completed)](#stage-43)
- [44. Current checkpoint round robin at update1200 (2026-11-05)](#stage-44)
- [45. Paired PPO through2200 and incumbent comparison](#stage-45)
- [46. Literature: robust fog play and PSRO proposal](#stage-46)
- [47. History-enriched imitation and frozen-opponent PPO pilot](#stage-47)

<a id="stage-30"></a>

## 30. Readiness to move from imitation baselines to PPO (2026-11-04)

The completed medium-residual versus full-CNN matchup supports using both frozen checkpoints as practical PPO opponents/evaluation baselines. It does not establish that PPO will improve either model or that the imitation optimization issues are fixed. Further imitation sweeps are not a prerequisite for starting the reinforcement-learning phase.

Repository inspection found no implemented PPO trainer, rollout buffer or environment adapter. Existing reusable pieces are the referee rules/observations/outcomes, shared encoders and policy logits, observation-based move filtering, checkpoint loading, and arena evaluator. The feedforward Policy has a value head, but imitation cross-entropy never supervised it; this is an untrained critic, not a pretrained value estimator.

Proposed initial design (not yet implemented or experimentally validated): initialize the actor from the medium residual; retain both baseline checkpoints frozen; train initially against a fixed mix of the two, alternate player colors, and judge improvement through separate arena evaluations. PPO needs newly collected on-policy games, rewards, correctly signed returns, a trained critic, advantages, consistent action masks/log-probabilities, and terminal-versus-truncation handling. Historical imitation replay data are not a drop-in PPO rollout buffer. For the current BN/dropout network, disable hidden dropout and keep BN running statistics fixed consistently during rollout and PPO loss evaluation, while allowing intended trainable weights to receive gradients; this is an implementation choice to avoid introducing mode-dependent action-probability differences. It is not a demonstrated fix for the earlier feature concentration. Monitor policy KL, PPO ratio-clipping fraction (distinct from gradient clipping), entropy, value prediction quality, actual gradient/update statistics, and frozen-opponent win rates. No PPO code or training was added in this readiness review.

Reference: [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). PPO alternates fresh environment interaction with minibatch optimization of the collected policy data.


<a id="stage-31"></a>

## 31. PPO infrastructure implementation and CPU pipeline validation (2026-11-04)

The user authorized a separate `rl/` implementation that reuses the provided game and interoperates with existing bot contracts. This is an infrastructure milestone, not a demonstrated reinforcement-learning strength improvement. Operating commands live in `/Users/jesuscaraball0/code/school_code/67980/fow/rl/README.md`; findings remain in this document.

### Game and agent composition

`referee.run_match` gained one optional `bot_factory` argument; its default remains the original executable Bot. No board, movement, visibility, terminal, clock, repetition, or draw logic was reimplemented. `rl.game.play_game` calls this match loop. In-process Agent factories implement reset/observe/act/close, and `rl.game.Command` delegates to the original referee.Bot for existing stdin/stdout executables. Actors receive only the provided fog observations and their own tracking history. Referee replay hidden-board fields are never used as policy inputs, masks or reward shaping.

Frozen feedforward/recurrent checkpoint opponents, arbitrary wire-protocol bot opponents, and latest/past-self neural opponents can be mixed. Self-play takes a frozen current-actor copy per rollout; recent snapshots are saved periodically. Selected opponents are paired across colors. External executable opponents must themselves remain unchanged if a fixed environment/opponent benchmark is intended. Protocol/agent failures abort training rather than becoming free win rewards.

The PPO learner initially supports **feedforward** shared Policy architectures, including both plain and residual CNNs. Recurrent opponents work, but recurrent PPO actors are explicitly rejected pending sequence/hidden-state rollout support. A generic game-contract bot need not be differentiable to play; the PPO learner additionally requires the differentiable actor/value/log-probability interface, supplied by the shared neural Policy.

### Learning implementation

PPO initialization loads the checkpoint architecture and encoder/policy weights. For imitation weights, the unused value head is zero initialized and then trained; encoder and actor logits are preserved at initialization. New runs initialized from PPO weights retain the learned critic; resume additionally restores optimizer state. The objective changes from matching recorded expert moves to improving terminal game outcome.

Fresh complete games supply terminal rewards +1/-1/0, from the learner's color. GAE is computed separately per learner trajectory, discounting learner decisions rather than opponent plies. Terminal bootstrap is zero. The referee's configurable ply cap is an actual game draw, not an incorrectly bootstrapped rollout cutoff. Default cap256 is applied to training and internal evaluation; ordinary arena evaluation can use its own explicitly recorded cap. There is no partial-game rollout chunking or distributed/vectorized collector in this first version.

Rollouts retain tokens, exact observation-based masks, sampled action, old masked log probability and old value. Optimization uses those same masks/temperature, normalized advantages, clipped ratios, critic MSE, entropy regularization and a KL early-stop guardrail. Dropout is off and BN running statistics are fixed consistently in both collection and optimization; encoder/affine/policy/value parameters remain trainable. PPO's ratio-clipping metric is distinct from gradient clipping.

Initial unvalidated default recipe: actor/encoder Adam LR1e-5, value-head LR1e-3, no weight decay, constant LRs, eps1e-5, four PPO optimization epochs, minibatches128, policy clip.2, value coefficient.5, entropy coefficient.01, global gradclip.5, targetKL.02 (early stop at1.5x), gamma.99/lambda.95,16games/update. Source-checkpoint inference temperature is preserved, falling back to.9 for historical imitation weights. These are starting settings, not a tuned or proven optimum.

Outputs include score/critic/KL/entropy/clipping plots, per-update/per-opponent results, explained variance, gradient/layer update summaries, and fixed-probe activation/weight histogram JSON using shared diagnostics. The diagnostic probe is sampled from the first captured training rollout, not held-out validation. Evaluations use an isolated fixed RNG and frozen baselines, selecting best_policy.pt by mean draw-adjusted evaluation score. Known baselines are a benchmark, not an independent test distribution.

Resume is supported at complete PPO update boundaries, including model, Adam, CPU/CUDA/Python RNG, logs, best score and past-self pool. `--updates` may increase, enabling extension of completed runs at the same constant LRs without restarting. Other recipe changes require a new initialized run. Interrupted incomplete updates are recollected, not reused off-policy. Same-platform deterministic CPU extension was tested exactly; GPU training/resume has not been executed in this session.

PPO exports use the existing standalone neural bot and bundle machinery. A small shared-loader change honors stored inference_temperature while preserving historical .9 defaults. A six-hour one-GPU Slurm script is provided but no cluster training was launched.

### Validation evidence

The repository test suite now has **53 tests**. New cases cover canonical-referee parity between Python and executable bots, genuine executable-opponent rollouts, referee illegal-move forfeits, color/reward signs and closed-form terminal GAE, mask exclusion, initial actor preservation/value initialization, frozen BN/dropout mode, old-log-probability consistency, evaluation and opponent-loading RNG isolation, recurrent-opponent/actor restrictions, exact completed-run extension, and standalone export/temperature compatibility. Tests and the local pipeline use CPU; sbatch syntax and Python compilation were also checked.

### Actual medium-checkpoint smoke run

Source: `/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/ppo-medium-residual-infrastructure-smoke/metrics.json`. The medium residual best epoch49 was used, with the two existing fixed imitation baselines as training/evaluation opponents. The source architecture is unchanged. This run used seed7980, **four PPO updates**, four complete training games/update, **two PPO epochs/update**, cap128, minibatch128 and temperature.9; other PPO settings match the defaults above. It first completed two updates and then successfully resumed with total updates increased to four. There were16 training games and five fixed-seed internal evaluations of four games each. The tiny repeated evaluation sample cannot estimate improvement reliably.

| PPO update | Training W/L/D | Learner decisions | Endpoint KL | Value loss | Explained variance after |
| --- | --- | --- | --- | --- | --- |
| 1 | 2/2/0 | 164 | 0.002175 | 0.103692 | 0.020037 |
| 2 | 1/3/0 | 103 | 0.000811 | 0.155594 | 0.005416 |
| 3 | 3/1/0 | 145 | 0.000786 | 0.118359 | 0.007260 |
| 4 | 2/2/0 | 150 | 0.002771 | 0.111377 | 0.016146 |

Fixed-seed internal mean evaluation score was75% at update0 and remained75% through update4. This is only3/4 score points on the same tiny benchmark sample, **not evidence of a75% expected win rate or PPO improvement**. Critic explained variance remains low; this is not a converged RL run. All logged losses/KL/gradients were finite. Every smoke-run optimizer step triggered gradient clipping at .5. Pre-clip global maxima by update were282.95,239.72,437.44,303.94; small endpointKL therefore does not imply small raw gradients. On the fixed145-observation continuation probe, the final-trunk top-channel activity share was89.616% after update2 and89.625% after update4, with62.5% relative dormancy at both. These near-unchanged statistics are not evidence that PPO fixed the previous concentration issue. Diagnostics added during continuation used145 fixed observations from the third rollout, comparing model states after updates2and4; they are not measurements on an initial held-out probe.

The final-update checkpoint was exported and played two games through the ordinary evaluate.py against the fullCNN: **0wins,2losses,0draws**, one game per color, both Kingcapturable, no bot failures. This confirms exported contract execution, not a statistically meaningful strength change. Both the unchanged75% internal sample and0/2 external sample must be reported, rather than selecting the favorable number. External result source:

`/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/ppo-medium-residual-infrastructure-smoke/export-check/arena-ppo-infrastructure-smoke-vs-full-cnn-2games/summary.json`.

### Limits and next step

The infrastructure is ready for a controlled initial PPO experiment, with frozen baselines retained as evaluation anchors. No substantial PPO learning curve, GPU run, learning superiority, or cure for supervised feature concentration has been established. Start with a bounded run, monitor frozen-opponent scores/entropy/KL/critic fit and layer diagnostics, then use a larger independent arena series. Hyperparameters and opponent mixtures remain experimental. PPO does not guarantee improvement merely because initialization is strong.

References: [Schulman et al., PPO](https://arxiv.org/abs/1707.06347), [Huang et al., The37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

Current primary training-metrics coverage is43records:42 historical imitation/stress records plus this PPO pipeline run. Repeated temporary test fixtures are not independent experiments and are not retained as findings.


<a id="stage-32"></a>

## 32. PPO draw bonus and expanded learning diagnostics

**Request:** persist/plot average reward, losses, value quality and other RL training diagnostics; make draw reward +0.1. This section supersedes the initial implementation's draw-zero recipe for new runs. Historical stage31 artifacts retain their original objective and observations.

### Reward and outcome accounting

New PPO games use terminal **win +1, loss -1, draw +0.1**, discounted per learner decision as before. Outcome classification uses referee result/color independently of numeric reward. Standard game score and best-policy selection remain win1/draw0.5/loss0; a positive draw reward must not be counted as a win or scored0.55. Both players receiving +0.1 makes this objective non-zero-sum. This rewards the draw outcome; a draw alone does not establish that each move was blunder-free or unexploitable. Termination reasons remain logged so repeated/capped survival can be distinguished from decisive performance. No game rules were changed.

The exact reward scheme is recorded in metrics/checkpoints. A draw-zero PPO checkpoint cannot silently resume into draw+0.1. Starting a separate run with --init-checkpoint OLD/last_policy.pt preserves the PPO actor/critic weights, starts a fresh optimizer/log, and uses the new reward. New-objective runs still support exact update-boundary resume.

### Measurements and graphs

Metrics JSON is saved every completed update; graphs refresh every checkpoint interval (default10 updates) and on the final update. Eight graph files are now produced:

| File | Measurements |
| --- | --- |
| training.png | Reward/score/critic loss/KL/entropy/ratio-clipping overview |
| rewards.png | Mean terminal reward per game and separate standard score; training and frozen evaluation |
| losses.png | Actor loss, half-MSE GAE critic loss and total entropy-regularized PPO objective |
| value_quality.png | Value MAE, RMSE and explained variance versus discounted actual terminal returns, before update, after same-batch fitting, and on frozen-opponent evaluation games |
| outcomes.png | Win/draw/loss rates and mean plies |
| policy_health.png | Endpoint approximate KL, masked entropy, probability-ratio clipping, gradient clipping and pre-clip gradient L2 mean/max |
| throughput.png | Games/decisions per collection second, cumulative training games/decisions |
| evaluation_opponents.png | Separate fixed-opponent score/reward curves |

Additional JSON includes reward standard deviation/standard error, per-color/per-opponent results, terminal reasons, GAE and independent Monte Carlo target fit, predicted/target means/std, raw advantage mean/std, optimizer steps/KL stopping, config/LRs, layer gradients and actual weight-update ratios. Existing fixed-probe activation/weight histograms remain available. The standard error is descriptive, not a guarantee of independent games.

Value prediction is regression, so classification accuracy is not the default metric. Independent discounted terminal targets were added alongside GAE targets: GAE targets themselves depend on the rollout critic. Fresh-rollout predictions precede fitting those games; post-update errors are same-batch fit, not generalization. Evaluation regressions weight learner decisions equally (longer games contribute more); aggregate plotted evaluation statistics weight opponents equally. Frozen opponents are a benchmark, not an untouched independent test population. Explained variance is null when target variance is effectively zero; target spread is retained for interpretation. Historical unmeasured quantities remain absent/gaps rather than invented measurements.

### Validation and saved pipeline check

All **55 repository tests pass**, including closed-form Monte Carlo returns and regression errors, the draw bonus through actual referee collection/evaluation, separate draw/score accounting, reward-scheme resume rejection, all graph files and exact new-objective resumed-versus-continuous training. Existing game-contract, inference-mode and RNG checks continue to pass.

Retained CPU pipeline run: `out/experiments/ppo-draw-bonus-metrics-smoke/`. This initializes the actual medium residual imitation checkpoint, trains against its frozen copy, uses two updates with two games each, two PPO optimization epochs, minibatch8, temperature0.9, one CPU thread and **max8 plies**. Evaluation is two color-balanced games at updates0,1,2. This deliberately short cap exercises draw handling, not playing-strength learning. All four training games and all six evaluation games reached draws; each update reported W0/L0/D2, mean reward0.1 and standard score0.5. There were16 total learner training decisions. All eight graphs were saved; reward and value panels were visually inspected.

| Update | Fresh-rollout MC RMSE | After-update same-rollout MC RMSE | MC explained variance before / after | Endpoint approximate KL |
| --- | --- | --- | --- | --- |
| 1 | 0.09851620 | 0.08391526 | 0.000000 / -5.710667 | 0.00146732 |
| 2 | 0.08539069 | 0.07355254 | -2.837661 / -16.260843 | 0.00537028 |

The terminal-target std was only0.00110690 in these fixed-length all-draw training batches. Consequently RMSE improved while explained variance worsened: almost constant targets make the variance ratio sensitive to small position-dependent prediction differences. This is a useful instrumentation check, not evidence of broader critic quality, collapse, improved strength, or an optimal draw bonus. No large PPO/GPU experiment was run for this change.

Current retained primary training-metrics coverage is **44 records**:42 historical imitation/stress records, the original draw-zero PPO pipeline run, and this separate draw-bonus diagnostic check. Temporary unit-test runs are not counted as experiments.


<a id="stage-33"></a>

## 33. One persistent RL checkpoint per model variant

**User requirement:** keep RL separate from original supervised checkpoints, use one RL file per feedforward model variant, continue future training from that file, and checkpoint new best policies. Recurrent PPO remains excluded.

The default storage is now `out/rl/<variant>/policy.pt`, separate from supervised and historical experiment artifacts. Variant identity normalizes the complete feedforward architecture configuration (widths, residual depth, optional head reduction, normalization/affine settings, dropout and activation) and includes a stable configuration digest. Current actual baseline identities are:

| Supervised source variant | Persistent RL file, created on first training launch |
| --- | --- |
| Medium residual32/128,3 blocks | out/rl/residual-c32-h128-518d814dce/policy.pt |
| Full plainCNN64/256 | out/rl/cnn-c64-h256-8ebd1cf653/policy.pt |

Other feedforward variants receive their own architecture-derived location on first use. This change does not fabricate trained RL policies for variants that have not been trained. Historical smoke results remain in their original experiment directories, with their recorded objectives; none were promoted into canonical production policies or deleted.

Repeating --init-checkpoint for an architecture automatically resumes its existing canonical RL file instead of reloading supervised weights. --updates remains the requested total completed PPO-update budget and must exceed the completed count. Explicit --resume also works. Optimizer, current actor/critic/BN state, RNG, update count, metrics, best score/weights and recent self-play pool are restored. Changing the initialization path to another checkpoint of the same architecture does not reset an existing RL lineage. Recipe-change validation remains active; the original source/checkpoint provenance is retained.

One file contains two intentional states: the top-level latest training state and an embedded, independently cloned best_policy inference state. Latest is saved after every completed update; each strict improvement in frozen-opponent mean standard score immediately replaces the embedded best and atomically checkpoints the file. Ties preserve the earlier best. Future training continues the latest weights/optimizer, rather than rolling back training to the best. Evaluation frequency determines when a new best can be observed (default every10 updates and final). Evaluation-disabled training has no evaluated best.

No new last/final/numbered/best checkpoint model files accumulate. Recent past-self opponents are bounded and embedded as inference snapshots in the same file, avoiding separate per-update snapshot files. Metrics/plots/replays and the diagnostic token probe are still separate data artifacts. Concurrent writers to the same variant directory are rejected with an OS file lock; writes use the shared atomic temporary-file replacement. Explicit --run-dir remains available for isolated experiments/tests, while the default and sbatch workflows use the canonical location. Existing historical metrics directories are protected from being reused as fresh runs.

rl/export.py selects the embedded best weights by default, with --latest for current training weights, and exports an ordinary inference-only game-contract checkpoint. Frozen RL checkpoint opponents also select the embedded best at load time. The Slurm script no longer creates job-ID-specific RL directories; repeated jobs extend the same variant, while log filenames retain the job IDs. Original supervised checkpoint files are never a save destination.

Validation: **57 repository tests passed** after the lifecycle changes, plus Python compilation and sbatch shell syntax checks. Tests include automatic continuation into one file/directory, exact explicit-resume equivalence (including embedded self-play history), source checkpoint byte preservation, architecture identity separation/default normalization, concurrent writer rejection, a new best at update1 surviving worse evaluation at update2, best-weight export without optimizer payload, tie retention and immutable stored best weights. Unit tests use temporary tiny policies and are not playing-strength experiments. No new full-scale PPO or GPU training was launched; retained primary metrics coverage remains44records.


<a id="stage-34"></a>

## 34. Training both CNN variants: current capability and literature

**Question:** can the full CNN and residual CNN both learn against each other, and is that advisable? This is a capability/research review, not an implementation or experimental result.

Code inspection: either feedforward architecture can be the PPO learner and either can be a frozen checkpoint opponent. Each has its own persistent RL checkpoint/optimizer. However, one train process updates only one learner; OpponentPool loads/caches checkpoint opponents once. Running two independent jobs does not dynamically refresh peer weights, even if the peer's file changes. Current self-play refreshes the learner's own frozen rollout copy and its own saved historical snapshots, not another live learner. A coordinated two-model trainer, cross-model historical pool and automatic peer-refresh barrier are not implemented. Separate short processes can reload newer peers between training blocks, but this is not an existing automated league facility.

### Primary literature and what it establishes

- [Berner et al., Dota 2 with Large Scale Deep Reinforcement Learning (2019)](https://cdn.openai.com/dota-2.pdf), AppendixN: OpenAI Five used latest-policy self-play for80% of games and past opponents for20%, explaining historical opponents as protection against narrow strategies and strategy collapse. The paper also evaluates against a fixed reference pool. This supports opponent history and stable benchmarks; those percentages are not established optimal for fog-of-war chess or two heterogeneous CNNs.
- [Vinyals et al., Grandmaster level in StarCraft II using multi-agent reinforcement learning (2019)](https://www.nature.com/articles/s41586-019-1724-z), and [DeepMind's accompanying technical account](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/): AlphaStar used a diverse league of main/exploiter strategies. The lab explains cyclic forgetting in naive self-play and motivates historical strategies/exploiters. This supports a broader opponent distribution, not a requirement to implement a large AlphaStar-style league here. Its learning algorithm and computational scale differ from this PPO prototype.
- [Heinrich and Silver, Deep Reinforcement Learning from Self-Play in Imperfect-Information Games (2016)](https://arxiv.org/abs/1603.01121): NFSP approached a Nash equilibrium in Leduc poker while common RL baselines diverged. This is especially relevant motivation for caution in imperfect-information games. Our PPO with a small recent-snapshot pool is not NFSP and inherits no NFSP equilibrium result.
- [Lanctot et al., A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning (2017)](https://arxiv.org/abs/1711.00832): independent learners can overfit their training partners; policy-space response oracles train approximate responses to policy mixtures and compute selection strategies. This supports testing generalization to other opponents rather than treating one evolving head-to-head score as absolute progress.

### Proposed design and recommendation (not yet tested)

Training both existing architectures is a reasonable next experiment after pipeline validation. Architectural diversity may expose different weaknesses, but we have not demonstrated that it yields strategic diversity or outperforms one learner with a diverse opponent pool. The earlier54–45–1 medium-versus-full result supplies a reasonably competitive observed starting matchup, not established superiority or a learning guarantee.

A minimal coordinated design would freeze both current policies for each collection round, use the provided referee to collect complete color-balanced games, retain each model's own actions/masks/old log-probabilities/critic values in separate buffers, and update both models with their own PPO optimizer only after collection. Games between the two current policies can teach both sides without pooling one architecture's actions into the other's learner batch. Reward is computed from each side's outcome; draw+0.1 is given to both sides and cannot be obtained by simply negating the other side's reward. Peer snapshots refresh at round boundaries. This limits mid-rollout policy drift; it does not eliminate learning nonstationarity across rounds or guarantee convergence.

The opponent distribution should include the current peer, selected historical versions of both models, and the frozen original imitation baselines. We do not propose a literature-proven mixture ratio for this task. Each variant retains its one persistent RL checkpoint with embedded snapshots; training peer versions and frozen evaluation opponents must be distinguished explicitly. Both policies should be evaluated separately against the same fixed benchmark pool and older policies, tracking standard score, reward and draw reasons. Their mutual win rate can stay near50% while both improve, both regress, or cycle; it is insufficient as a progress measure. Best-checkpoint selection should continue to use the fixed benchmark score, not the moving peer alone.

The +0.1 draw bonus makes the objective non-zero-sum; zero-sum equilibrium claims from other algorithms do not transfer directly. A practical small population remains sensible, but any claims about stability, exploitability, convergence or superiority require actual experiments. No coordinated trainer, new training job or new experimental result was added in this review. Primary metrics coverage remains44records.


<a id="stage-35"></a>

## 35. Minimal paired PPO: both CNNs learn from the same games

**Scope correction:** the user requested only that the full CNN and residual CNN play each other and both update. The broader historical-opponent mixture proposed in stage34 is not implemented in this mode. That proposal was population-based training in spirit, but not a full PSRO algorithm: PSRO additionally constructs/selects strategies through a population-level meta-strategy computation. This implementation has no payoff matrix, population matchmaking or meta-strategy solver.

### Implemented behavior

rl/train_pair.py accepts --first and --second source checkpoints. They must be distinct feedforward variants; recurrent actors remain rejected. It uses the existing referee through rl.game.play_game, collecting alternating-color complete games with both PolicyAgents recording their own transitions. Both policies remain unchanged during collection. Each then receives a separate PPO update using its own observations, masks, actions, old action probabilities, critic predictions, GAE/Monte Carlo targets and optimizer. No opponent moves are misattributed as the learner's own data. The updated policies face each other in the next round. Both receive +0.1 for a draw; decisive rewards are assigned independently by color, not by blindly negating the opponent's draw bonus.

The original single-learner lifecycle was factored into a shared collection/update interface so paired training retains the same diagnostics, best-policy selection, configuration validation, inference modes and exports. Paired training supplies exactly the other current model as the training opponent; it does not sample historical snapshots or the original baselines for training. Frozen baseline evaluation remains separate and selects each model's embedded best independently. Evaluation never updates the policies.

Each model continues its existing canonical out/rl/<variant>/policy.pt. The two models must have the same completed update count to resume together; --updates is the new total budget. Stable variant ordering preserves shared RNG continuation even if --first/--second are reversed on a later launch. Each model has its own metrics, plots and optimizer, while the games are shared: games-per-update16 means16 games and16 episodes for each learner, not32 distinct games. Learner decision counts remain separate. Both updates currently execute sequentially on one selected device.

After a successful shared round, both checkpoint files are committed. A short-lived transaction journal allows a restart to finish a process interruption between the two atomic replacements, then disappears. Single-model training refuses to bypass an interrupted pair commit. If optimization fails before that commit, previously saved model checkpoints remain the completed pair. Diagnostics/metrics may have been written for an attempted round; checkpoint-embedded logs remain authoritative on resume. Each model still has only one persistent RL model file; the journal is temporary recovery data, not another training lineage. Best policies found during a round are included in that round's pair commit. Slurm supports the same workflow through sbatch rl/ppo.sbatch --pair.

Usage from the repository root:

```bash
.venv/bin/python rl/train_pair.py \
  --first out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt \
  --second out/experiments/residual-medium-27890681/best_policy.pt \
  --updates 20 --games-per-update 16
```

Repeat with a higher update total to continue. Current normal defaults include max256plies; the verification run below deliberately uses a shorter cap. Operational local/cluster commands are in rl/README.md.

### Verification and saved actual-model check

All **60 repository tests pass** after the paired-loop implementation. New checks cover separate probability-consistent batches for both players through the real referee, unchanged weights during collection, per-color win/loss targets and +0.1 to both draw participants, updates to both policies, byte-identical preservation of original source checkpoints, exact resumed-versus-continuous training including a reversed CLI model order, one RL file per variant and recovery after a simulated interruption between the two model saves. Python compilation and sbatch shell syntax also pass.

Retained CPU smoke evidence: out/experiments/ppo-two-model-update-smoke/, containing two descriptive variant directories. The actual full CNN64/256 and medium residual32/128 initialized from their existing supervised best checkpoints, each with its imitation-unsupervised critic zeroed. Settings: two rounds, two shared games per round, max8plies, two PPO epochs, minibatch8, one CPU thread, temperature0.9, otherwise current PPO defaults. Each policy separately evaluated two games against each of the two frozen original baselines at updates0,1,2. These24 evaluation games are distinct from the4 shared training games.

All4 training games reached the cap: each model logged W0/L0/D2 per round, reward0.1, standard score0.5, and8 own decisions per round (16 per model total). Both optimizers executed two steps each round. Each variant saved one policy.pt and eight learning-curve PNGs, plus metrics and diagnostics. There is no large training/GPU run or strength conclusion here.

| Variant | Round | Actor loss | Critic loss | Endpoint approximate KL | Frozen benchmark mean score |
| --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | 1 | -0.00513769 | 0.00616541 | 0.00070755 | 37.5% |
| cnn-c64-h256-8ebd1cf653 | 2 | -0.00509593 | 0.00140494 | 0.00058394 | 37.5% |
| residual-c32-h128-518d814dce | 1 | -0.02058411 | 0.00401953 | 0.00956590 | 50.0% |
| residual-c32-h128-518d814dce | 2 | -0.01669215 | 0.00337528 | 0.00631805 | 50.0% |

The full CNN's policy-head weight L2 change from its original source was0.00124104; the residual's was0.000418629. Thus both actors actually updated, rather than merely executing games or changing only the critics. Frozen benchmark scores were37.5% for fullCNN and50% for residual at update0 and remained unchanged through update2; both retained their initial policy as the evaluated best. These are four-game-per-model fixed-seed evaluations with a deliberately eight-ply cap, not estimates of normal-length playing strength. The smoke verifies data ownership, learning updates and persistence, not that paired PPO improves performance or prevents cycling.

Retained primary training-metrics coverage is now **46 records**:44 previous records plus the two model logs from this one paired smoke experiment. They share the same4 training games and must not be counted as8 independent games. No historical artifacts were overwritten or promoted to canonical production RL policies.


<a id="stage-36"></a>

## 36. Full CNN and medium residual: 400 paired PPO updates

**Request:** run400 steps for each model, monitor, briefly summarize results/metrics and next recommendations. The stated interpretation was400 completed PPO updates per model, not400 individual optimizer minibatches. Both completed without training errors. This is the first sustained paired PPO run, not the earlier short-cap smoke checks.

### Recipe, scope and verification

Sources are the fullCNN64/256 supervised best from job27800942 and medium residual32/128 supervised best from job27890681. Canonical output locations are out/rl/cnn-c64-h256-8ebd1cf653/ and out/rl/residual-c32-h128-518d814dce/. Each keeps one policy.pt containing latest training state and the independently selected best; metrics, eight learning curves and fixed-probe histogram/statistic JSON are separate evidence. No original supervised file was overwritten. Source hashes match the hashes recorded at initialization; all latest checkpoint tensors are finite, all inherited BN running buffers are unchanged, both checkpoints/logs confirm completed update400, and no unfinished pair transaction journal remains.

Paired collection used16 shared color-balanced complete games/update, max256plies, terminal rewards+1/-1/+0.1, gamma0.99/lambda0.95 over learner decisions, temperature0.9, seed7980. Actor/encoder Adam LR1e-5; critic LR1e-3; Adam eps1e-5, default betas(.9,.999), no weight decay;4 optimization epochs, minibatch128, probability ratio clip0.2, value coefficient0.5 applied to half-MSE, entropy coefficient0.01, global gradient clip0.5, approximate-KL stop threshold1.5*0.02. No schedule, historical opponent sampling, population solver or architecture modification was added. Dropout remained off and BN running statistics stayed fixed consistently during collection and optimization. Actual device was local CPU with one Torch thread.

Each model evaluated20 color-balanced games against each of the two frozen supervised baselines at update0 and every10 updates:40 games per evaluation,41 evaluation events per model,3280 evaluation games total. Evaluation seed9137 was fixed and isolated from training RNG. A thread monitor was registered and paused after training completed; no retraining or hyperparameter changes occurred during monitoring. Logs: logs/ppo-pair-400.out/.err; stderr was empty. Per-model wall timers were1677.55s (fullCNN) and1663.23s (residual), approximately28minutes plus initial startup; these overlap and must not be summed. Shared collection used465.33s total; model optimization timers totaled241.42s and426.51s, excluding most evaluation/diagnostic overhead.

Analysis artifacts: out/rl/paired-400-analysis.json, out/rl/paired-400-fresh-evaluation.json, out/rl/paired-400-comparison.png. Analysis/evaluation scripts are retained in work/. The comparison chart summarizes the fixed-seed benchmark, fresh-seed benchmark, value RMSE and KL; full per-update/layer evidence remains in each variant's metrics/diagnostics.

### Shared training outcomes and actual update counts

There were **6400 distinct training games**, yielding6400 own-color episodes per learner. Summing both model logs as12800 games would double-count the same games. Terminations:6339 King capturable and61 threefold repetitions; draw fraction0.953125%. No training game reached the ply cap, so this run gives no evidence of exploiting cap draws. There were no bot/infrastructure failures. Mean reward is terminal reward per game, not discounted return; critic targets are discounted separately.

| Model | Wins / losses / draws | Standard score | Mean terminal reward | Learner decisions | Optimizer minibatch steps | KL early-stop updates |
| --- | --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | 3217 / 3122 / 61 | 50.7422% | 0.01579687 | 226,002 | 7,868 | 0 |
| residual-c32-h128-518d814dce | 3122 / 3217 / 61 | 49.2578% | -0.01389063 | 226,049 | 7,766 | 10 |

Head-to-head score over the first50 updates was48.0625% fullCNN /51.9375% residual; the last50 updates was54.9375% /45.0625%. Mean training reward/game changed from−0.038625 to+0.100625 for fullCNN and+0.038875 to−0.096875 for residual across those windows. Mean length rose from67.3225 to73.48375plies (whole-run mean70.63297). These rewards are against the changing peer, distinct from the fresh frozen-baseline rewards below. That is a moving matchup, not absolute playing strength. Both models can improve against the original baselines while their mutual score remains close to50%.

### Checkpoint-selection benchmark (same fixed seed used during training)

| Model | Initial score | Selected best update | Selected best score | Final update400 score |
| --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | 52.50% | 350 | 75.00% | 60.00% |
| residual-c32-h128-518d814dce | 40.00% | 210 | 80.00% | 43.75% |

These peaks were selected across41 repeated40-game benchmarks and are optimistic performance estimates. The residual's initial fixed-seed benchmark was40%, not55%; an intermediate commentary misstated it and was corrected from the stored metrics. Do not conflate these initial fixed-seed measurements with the separate fresh-seed baseline values below.

### Fresh-seed benchmark after completion

To check whether apparent gains survive new game trajectories, evaluated original policies, selected-best policies and latest update400 policies on **100 games each**,50 against each frozen original baseline, alternating colors. Seed572940, temperature0.9 and max256plies. Total600 additional evaluation games. Initial actor weights were unchanged; initial critics were zeroed as in PPO initialization. Neither training nor checkpoint selection consumed these games. This is a fresh trajectory sample against the same known opponent pool, not an independent opponent distribution.

| Policy | Selected update | Wins / losses / draws | Score | Terminal reward/game | Approximate descriptive score SE |
| --- | --- | --- | --- | --- | --- |
| full_cnn_initial | supervised source | 54 / 46 / 0 | 54.00% | 0.0800 | 0.0501 |
| residual_initial | supervised source | 45 / 55 / 0 | 45.00% | -0.1000 | 0.0500 |
| full_cnn_ppo_best | 350 | 65 / 35 / 0 | 65.00% | 0.3000 | 0.0479 |
| residual_ppo_best | 210 | 59 / 41 / 0 | 59.00% | 0.1800 | 0.0494 |
| full_cnn_ppo_latest | 400 | 66 / 34 / 0 | 66.00% | 0.3200 | 0.0476 |
| residual_ppo_latest | 400 | 60 / 39 / 1 | 60.50% | 0.2110 | 0.0489 |

Update400 gains were+12percentage points for fullCNN (54→66%) and+15.5points for residual (45→60.5%) on this new seed. Selected-best scores were65% and59%. Their difference from the latest scores (1 and1.5points) is much smaller than the sampling uncertainty; this does not establish which checkpoint is superior. Each100-game score has roughly±9–10percentage points of nominal95% sampling uncertainty, and initial-versus-trained differences also have uncertainty. Common sampling seeds and opponent strata complicate independence assumptions; no formal paired significance or exploitability claim is made. Nevertheless, both trained models improving on new trajectories is more encouraging than the selected75/80% fixed-seed peaks alone. Original selected-best checkpoints remain selected by the original protocol; fresh evaluation did not silently replace them.

Per-opponent fresh scores:

| Policy | Frozen opponent | Score | W / L / D |
| --- | --- | --- | --- |
| full_cnn_initial | full-data-plain-cnn-adamw-bn-8epochs-job27800942 | 54.00% | 27 / 23 / 0 |
| full_cnn_initial | residual-medium-27890681 | 54.00% | 27 / 23 / 0 |
| residual_initial | full-data-plain-cnn-adamw-bn-8epochs-job27800942 | 42.00% | 21 / 29 / 0 |
| residual_initial | residual-medium-27890681 | 48.00% | 24 / 26 / 0 |
| full_cnn_ppo_best | full-data-plain-cnn-adamw-bn-8epochs-job27800942 | 68.00% | 34 / 16 / 0 |
| full_cnn_ppo_best | residual-medium-27890681 | 62.00% | 31 / 19 / 0 |
| residual_ppo_best | full-data-plain-cnn-adamw-bn-8epochs-job27800942 | 62.00% | 31 / 19 / 0 |
| residual_ppo_best | residual-medium-27890681 | 56.00% | 28 / 22 / 0 |
| full_cnn_ppo_latest | full-data-plain-cnn-adamw-bn-8epochs-job27800942 | 66.00% | 33 / 17 / 0 |
| full_cnn_ppo_latest | residual-medium-27890681 | 66.00% | 33 / 17 / 0 |
| residual_ppo_latest | full-data-plain-cnn-adamw-bn-8epochs-job27800942 | 54.00% | 27 / 23 / 0 |
| residual_ppo_latest | residual-medium-27890681 | 67.00% | 33 / 16 / 1 |

### Losses, critic quality and policy stability

Window comparisons below use first50 versus last50 updates (800 shared games/window). Losses are means over updates of the per-update minibatch-sample-weighted loss. Monte Carlo RMSE pools squared error by learner decisions; explained variance is an update-average, not a pooled variance ratio. Opponents and state distributions change during training, so these are descriptive trends rather than fixed-dataset learning curves. GAE critic loss is distinct from error against independently discounted terminal outcomes.

| Model | Window | Actor loss | Critic half-MSE loss | MC RMSE before update | Mean MC explained variance before update | Entropy (nats) | Endpoint KL | PPO ratio clipping |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | first50 | -0.013296 | 0.118857 | 0.846425 | 0.005520 | 1.838986 | 0.003676 | 1.277% |
| cnn-c64-h256-8ebd1cf653 | last50 | -0.012639 | 0.110308 | 0.821758 | 0.014473 | 1.530814 | 0.003649 | 1.337% |
| residual-c32-h128-518d814dce | first50 | -0.008375 | 0.121641 | 0.845745 | -0.001595 | 1.838923 | 0.004276 | 2.954% |
| residual-c32-h128-518d814dce | last50 | -0.011043 | 0.109874 | 0.817802 | 0.017172 | 1.322847 | 0.007446 | 4.134% |

Both critics improved RMSE only modestly (about0.846→0.822 fullCNN and0.846→0.818 residual); fresh-rollout explained variance remained near0.01–0.02 late in training. They are weak outcome predictors despite actor performance gains. This is not a converged critic, but does not prove PPO cannot improve with it or that a larger architecture is needed. Fixed-baseline evaluation has an additional opponent/state-distribution shift relative to current-peer training.

| Model | Optimizer-step-weighted gradient clipping fraction (whole run) | Mean pre-clip gradient L2 (last50 update means) | Largest pre-clip gradient L2 (whole run) | Largest endpoint KL (whole run) |
| --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | 100.00000% | 20.423789 | 285.728302 | 0.013465 |
| residual-c32-h128-518d814dce | 100.00000% | 153.529180 | 3693.254639 | 0.029346 |

Both clipped every optimizer step in the last50 updates. Residual raw gradients were substantially larger, but endpoint policy changes stayed modest, no losses/checkpoint tensors became nonfinite, and the KL guard stopped further residual minibatches on10/400 updates. Frequent clipping is a monitored scale concern, not sufficient evidence of numerical divergence; low KL also does not mean the inherited feature concentration is fixed. The global gradient threshold was kept at0.5 throughout; it was not raised during the run.

### Fixed-probe feature health

Each model's512-observation probe was sampled from its own first training rollout, then held fixed. Probes are not held-out validation and differ between models, so within-model initial/final changes are more interpretable than direct architecture comparisons. BN running statistics were verified unchanged. Values below are activation mass shares and relative dormancy, not feature-importance proofs.

| Model | Layer | Update | Zero fraction | Relative dormant fraction | Top1 activity share | Top4 activity share |
| --- | --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | encoder.1 | 0 | 51.3777% | 0.0000% | 5.7192% | 17.4654% |
| cnn-c64-h256-8ebd1cf653 | encoder.3 | 0 | 96.3753% | 17.1875% | 24.0929% | 79.1239% |
| cnn-c64-h256-8ebd1cf653 | encoder.6 | 0 | 64.3387% | 29.6875% | 30.4388% | 91.3591% |
| cnn-c64-h256-8ebd1cf653 | encoder.1 | 400 | 51.0955% | 0.0000% | 5.7393% | 17.4469% |
| cnn-c64-h256-8ebd1cf653 | encoder.3 | 400 | 96.0919% | 10.9375% | 24.3028% | 78.7399% |
| cnn-c64-h256-8ebd1cf653 | encoder.6 | 400 | 57.5562% | 25.0000% | 29.6256% | 88.9439% |
| residual-c32-h128-518d814dce | encoder.1 | 0 | 45.0327% | 0.0000% | 7.5749% | 26.4392% |
| residual-c32-h128-518d814dce | encoder.2.2.relu2 | 0 | 96.7492% | 62.5000% | 89.6036% | 98.0270% |
| residual-c32-h128-518d814dce | encoder.6 | 0 | 52.6779% | 10.1562% | 7.1508% | 16.1101% |
| residual-c32-h128-518d814dce | encoder.1 | 400 | 44.7476% | 0.0000% | 7.5602% | 26.0643% |
| residual-c32-h128-518d814dce | encoder.2.2.relu2 | 400 | 95.6946% | 59.3750% | 89.1573% | 97.7737% |
| residual-c32-h128-518d814dce | encoder.6 | 400 | 39.6988% | 10.1562% | 4.7712% | 13.0775% |

Residual final-trunk top1 share remained about89% (89.60→89.16%); this run did not fix the inherited concentration. Dense activity became less sparse and less concentrated in both models, particularly the residual. The fullCNN second-trunk top4 share remained about79%. Those modest internal changes coexist with benchmark improvement and weak critics; no causal architecture/norm conclusion follows from this single paired run.

### Takeaways and next recommendations

1. The minimal two-current-policy PPO pipeline completed a meaningful run and both actors improved against the original baselines on a fresh trajectory sample. This is encouraging first evidence, not equilibrium convergence or general strength against an unseen opponent population. Keep the persistent latest states for continuation and preserve the independently selected bests already embedded in each file.
2. Improve evaluation reliability before judging small gains or selecting another long run: use more games and additional sampling seeds for checkpoint comparison, and eventually an additional opponent outside the two training lineages. The75/80% selection peaks substantially overstated this fresh sample's65/59% selected-best scores. Latest versus selected-best is unresolved here.
3. Prioritize a small rollout-size experiment (e.g.16 versus64 games/update at matched total game budget), holding architecture and optimizer recipe fixed. Its purpose is to test whether larger batches reduce noisy policy/value estimates; improvement is a hypothesis, not established by this run. More updates alone have not demonstrated a well-fitted critic. If value fit remains weak, isolate a critic-learning-rate/fit-budget ablation rather than changing several components together.
4. Do not enlarge either CNN or loosen clipping solely because gradients were clipped. There was no numerical failure, and policy KL stayed controlled. Inherited concentration and low critic explained variance remain open issues to measure; raw norm magnitude by itself does not identify a remedy.

No PSRO or historical-opponent scheme was introduced. Total primary training-metrics coverage is now **48 records**:46 previous records plus these two canonical sustained-PPO model logs. This is one paired experiment with6400 training games, not two independent6400-game datasets. The600 fresh evaluation games and3280 repeated fixed-seed evaluation games are evaluations, not additional training experience.


<a id="stage-37"></a>

## 37. Assessing value pretraining on historical games

**Question:** do the weak critic metrics justify pretraining the value head on historical games? This is a proposed controlled experiment, not a completed training change. Low Monte Carlo explained variance and modest error reduction identify weak outcome prediction, but do not alone establish that critic quality is the dominant cause of limited policy improvement. Both actors improved against the original benchmark field despite those metrics.

Local inspection found61445 game_data/*.json replay files. Two inspected examples contain final result/reason, rulesVersion/player-version metadata, and frame-level white/black fog observations alongside the hidden board. They ended King capturable with recorded decisive results. This confirms the format supplies outcome labels and appropriate visible inputs; it is not an audit that all61445 files are valid, compatible or complete. The existing supervised replay_examples reads result to reject unfinished games but only yields observation/move labels, and ReplayDataset stores no value targets.

A value-only dataset could pair each player's pre-decision fog observation with the discounted actual terminal outcome, using the PPO convention: win+1/loss−1/draw+0.1, gamma0.99 over that player's decisions, final decision discount exponent0. Do not supply the hidden board to the deployed critic or use a full-information engine evaluation as if it were the same target. Filter unfinished/protocol-failure/incompatible-rules games; split by complete games before sampling positions, preferably using the established game-level split. Use the same input encoding as PPO. Correlated positions from one game must not cross train/validation partitions.

The safest initial experiment starts from the current PPO weights, freezes both the shared encoder (including BN state) and policy head, and optimizes only the existing value head, selecting by held-out-game regression error. Its purpose is to improve the critic without changing the already-improved actor. The existing head is just nn.Linear(hidden_size,1) followed by tanh: a frozen representation may limit what it can predict. If head-only fitting yields little useful signal, investigate label correctness, optimization and feature adequacy rather than assuming additional historical examples alone solve it. Updating the shared encoder for value loss would also change actor logits even if the policy output layer were frozen; that is a different ablation, not included in this proposal.

Historical returns estimate outcome expectations under the historical players/opponent mix. They are not unbiased evaluations of the current400-update policies. Treat this as a warm start and measure transfer on fresh current-policy games, then continue on-policy value learning. In particular, a lower historical validation MSE alone does not prove PPO improves. Compare warmed versus unchanged critics from identical actor checkpoints with matched subsequent PPO game budgets, checking fresh-rollout Monte Carlo RMSE/explained variance and frozen-opponent policy scores. Reward/discount conventions must remain consistent. PPO continuation already preserves critics in PPO checkpoints; fresh initialization from an ordinary imitation checkpoint currently zeroes the head, so any future non-PPO value-pretrained checkpoint path would need an explicit preservation mechanism.

Literature context: [AlphaGo's authors describe learning value networks from self-play games](https://research.google/blog/alphago-mastering-the-ancient-game-of-go-with-machine-learning/), supporting outcome-based value fitting while highlighting the relevance of the policies generating those games. [Cobbe et al., Phasic Policy Gradient (ICML2021)](https://proceedings.mlr.press/v139/cobbe21a.html) separated policy/value training phases, enabled greater value-training sample reuse, and improved Procgen sample efficiency relative to PPO. PPG supports the motivation for more dedicated critic optimization and attention to shared-network interference; it is not evidence that our historical-game head-only warm start will work, nor a recommendation to implement all of PPG.

Recommendation: a small value-head-only warm-start versus unchanged-critic experiment is worth prioritizing before enlarging models or another long PPO run. No value-training code or pretraining run was launched in this assessment; retained primary training-metrics coverage remains48records.


<a id="stage-38"></a>

## 38. Historical value-head warmup, then PPO updates401–800

**Authorized experiment in progress:** one full epoch of historical value-head-only training for both current update400 policies, followed by400 more paired PPO updates to total800. No un-warmed400→800 control was requested; a before/after comparison will not isolate the causal contribution of warmup from additional PPO training.

Implementation: rl/pretrain_value.py uses the existing full-data split manifest, remapping old cluster paths by filename to local game_data. Pre-decision fog observations are labeled with actual terminal win+1/loss−1/draw+0.1, gamma0.99 discounted per own decision, exponent0 on that player's final decision. Hidden boards are not inputs. Only fow-v1 gameplay terminations are included. No game rules are reimplemented. Split identity and excluded counts are recorded.

Verified usable dataset: train55060 games/5332962 positions from55301 listed files; validation6114 games/592666 positions from6144 listed files. Excluded241 train files and30 validation files had non-gameplay termination reasons; no replay parse errors or missing split files were encountered. Training results:28791 white wins,22822 black wins,3447 draws. Validation:3215 white wins,2544 black wins,355 draws. Training reasons:51613 Kingcapturable,3253 threefold,194 fifty-move. Validation:5759 Kingcapturable,340 threefold,15 fifty-move. This is the same file-level partition used previously, with stricter value-target outcome filtering; its usable-position counts differ from move imitation.

Both CNN encoders, policy heads and BN buffers are frozen; only the existing linear+tanh value heads are optimized. Warmup: exactly one shuffled epoch, batch1024, local CPU4threads, each head's saved LR0.001, Adam default betas(.9,.999),eps1e-5,no decay, half-MSE loss, head-gradient clip0.5. Full-validation error is measured before and after. Actor/BN state identity is checked bitwise. The one persistent RL file per variant is updated through the existing pair commit mechanism; actor Adam and saved RNG states remain intact, critic moments are cleared for on-policy adaptation, and warmup metadata is recorded after PPO update400. Existing best snapshots are preserved. An explicit stage guard makes retrying this warmup idempotent.

Validation at implementation:62 repository tests passed, including closed-form signed/discounted labels for both sides, positive rewards for both draw participants, no hidden-board dependency, frozen actor/BN weights, preservation of actor optimizer moments/RNG, critic-moment reset, duplicate-warmup prevention and subsequent paired PPO continuation. Python compilation/diff checks pass. No preliminary truncated-data training substituted for the authorized one-epoch run.

The pipeline work/run_value_warmup_then_ppo.sh was launched locally (exec3157). Warmup stdout/stderr: logs/value-pretrain-after-400.out/.err. Progress/results: out/rl/value-pretrain-after-400.json/.png. On successful warmup it automatically executes the existing paired trainer with the unchanged PPO recipe to total800, CPU1thread: logs/ppo-pair-401-800.out/.err. Canonical checkpoints stay in their original out/rl variant directories. A recurring monitor was updated for both phases. Results are pending; no performance improvement is asserted at this point.


<a id="stage-39"></a>

## 39. Completed historical value warmup and paired PPO through update800

This completes the experiment authorized in stage38: exactly one historical training epoch fitting only both value heads, followed by400 additional paired PPO updates (401–800). Existing actor architectures, rewards, opponent pairing, PPO recipe and evaluation protocol were retained. No un-warmed400→800 control was run, so subsequent policy gains cannot be attributed solely to pretraining. All preceding400-update evidence is preserved in its original analysis/evaluation files.

### Historical head-only fitting

The existing full-data split supplied55060 usable training games/5332962 positions and6114 validation games/592666 positions, after excluding241/30 non-gameplay-termination files. Historical inputs were the player's own pre-decision fog views. Terminal return targets used +1/−1/+0.1 and gamma0.99 per own decision. See stage38 and the raw phase report for result/reason distributions, filtering and split identity.

One shuffled epoch had5208 optimizer batches, batch1024, CPU4threads, Adam LR0.001/eps1e-5/default betas/no decay, half-MSE and head-only gradient clip0.5. Encoders, policy heads and BN buffers remained frozen, with a bitwise assertion before commit. Actor Adam moments/RNG and earlier evaluated best snapshots were preserved; critic moments were cleared before on-policy continuation. Both heads committed to the same persistent policy.pt files. Warmup is recorded at PPO update400 and does not increment its PPO counter.

| Model | Historical validation RMSE before / after | Validation explained variance before / after | Validation MAE before / after | Online train RMSE | Head gradient clip fraction |
| --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | 0.779684 / 0.721843 | 0.059410 / 0.088583 | 0.691147 / 0.669160 | 0.722599 | 94.316% |
| residual-c32-h128-518d814dce | 0.773018 / 0.733963 | 0.033984 / 0.057378 | 0.715260 / 0.687074 | 0.731724 | 0.019% |

Warmup wall time: 888.18s. Historical error is on a different state/policy/episode-length distribution than current-peer PPO error; compare before versus after on the same historical validation set, not historical RMSE directly against PPO RMSE. One epoch does not prove convergence or head-capacity sufficiency.

### Continued paired PPO

The additional6400 shared training games bring total shared PPO experience to12800 games, not25600. Each model learns from its own decisions in the same games. Fixed-baseline evaluations remain40 games/model every10 updates, using the existing isolated seed9137. Best selection spans the entire run; it was not reset by warmup. No historical opponents were added to PPO training.

| Model | Phase401–800 W / L / D | Score | Mean terminal reward/game | Phase decisions | Selected best update | Selected fixed-seed score | Final fixed-seed score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | 3514 / 2738 / 148 | 56.063% | 0.123562 | 227,521 | 680 | 80.00% | 67.50% |
| residual-c32-h128-518d814dce | 2738 / 3514 / 148 | 43.938% | -0.118938 | 227,908 | 210 | 80.00% | 57.50% |

cnn-c64-h256-8ebd1cf653 phase2 reasons: {'King capturable': 6252, 'Threefold repetition': 148}.

residual-c32-h128-518d814dce phase2 reasons: {'King capturable': 6252, 'Threefold repetition': 148}.


### Critic transfer, losses and update stability

Window measurements use pre-update predictions on fresh rollouts. RMSE pools squared errors by decision; explained variance and PPO loss/health values are update means, not pooled variance ratios. First10 post-warmup updates contain160 shared games, each50-update window800 games. Opponents/state distributions evolve, so comparisons are descriptive.

| Model | Window | MC RMSE | Mean MC explained variance | Critic half-MSE loss | Actor loss | Endpoint KL | PPO ratio clip | Gradient clip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | pre-warmup last50 | 0.821758 | 0.014473 | 0.110308 | -0.012639 | 0.003649 | 1.337% | 100.000% |
| cnn-c64-h256-8ebd1cf653 | post-warmup first10 | 0.827248 | 0.006016 | 0.113002 | -0.013105 | 0.003648 | 1.057% | 100.000% |
| cnn-c64-h256-8ebd1cf653 | post-warmup first50 | 0.814632 | 0.010334 | 0.108274 | -0.012801 | 0.003622 | 1.460% | 100.000% |
| cnn-c64-h256-8ebd1cf653 | post-warmup last50 | 0.810397 | 0.042042 | 0.109876 | -0.012582 | 0.003749 | 1.440% | 100.000% |
| residual-c32-h128-518d814dce | pre-warmup last50 | 0.817802 | 0.017172 | 0.109874 | -0.011043 | 0.007446 | 4.134% | 100.000% |
| residual-c32-h128-518d814dce | post-warmup first10 | 0.818640 | 0.016882 | 0.111115 | -0.010806 | 0.007243 | 3.229% | 100.000% |
| residual-c32-h128-518d814dce | post-warmup first50 | 0.815386 | 0.004209 | 0.109072 | -0.011131 | 0.008230 | 4.539% | 100.000% |
| residual-c32-h128-518d814dce | post-warmup last50 | 0.821264 | 0.010650 | 0.114067 | -0.011430 | 0.009119 | 3.975% | 100.000% |


Large raw gradients and frequent global clipping must be interpreted together with actual policy KL, finite weights/losses and update ratios. No causal diagnosis of gradient explosion follows solely from the clipping frequency. Full norms/layer updates, entropy, throughput, reward/outcome curves and fixed-probe activation histograms remain in the canonical per-model metrics/plots. PPO figures mark the value-warmup boundary at400.

### Repeated trajectory benchmark

Evaluated latest and selected-best policies against the same two frozen original baselines,50 games/opponent,100 games/policy, temperature0.9,max256plies,seed572940. This repeats the earlier400-update benchmark protocol/seed; it is not a new unseen opponent distribution or untouched sampling-seed test. Reused update400 scores below are references, not new games. Differences of a few percentage points are unresolved by100 games per policy.

| Policy | Update400 reference score | Current selected update | Current W / L / D | Current score | Current reward/game |
| --- | --- | --- | --- | --- | --- |
| full_cnn_ppo_latest | 66.00% | 800 | 71 / 28 / 1 | 71.50% | 0.4310 |
| full_cnn_ppo_best | 65.00% | 680 | 70 / 29 / 1 | 70.50% | 0.4110 |
| residual_ppo_latest | 60.50% | 800 | 63 / 36 / 1 | 63.50% | 0.2710 |
| residual_ppo_best | 59.00% | 210 | 59 / 41 / 0 | 59.00% | 0.1800 |


Selected-best peaks are optimistic because the same small frozen benchmark chooses among many checkpoints. Fresh/repeated benchmark scores should not silently replace the stored best state. More PPO updates and warmup were combined here; a matched no-warmup continuation is needed to isolate whether pretraining caused policy improvement or degradation. Historical held-out head fitting and current-policy value transfer are distinct questions.

### Artifact and lifecycle verification

Source supervised checkpoint hashes remain unchanged; latest checkpoint tensors are finite; BN running buffers still match their inherited supervised values. Both checkpoints/logs complete update800, record the warmup after400, and have no unfinished transaction journal. Each variant still contains one top-level RL model file, with latest optimizer state plus independently preserved best weights. Existing first400 histories were extended rather than replaced. The full test suite passed62 tests during implementation; source compilation/diff checks passed. No additional large architecture or PSRO machinery was introduced.

Artifacts: out/rl/value-pretrain-after-400.json/.png; out/rl/paired-after-value-warmup-800-analysis.json; out/rl/paired-after-value-warmup-800-evaluation.json; out/rl/paired-after-value-warmup-800-comparison.png; per-model canonical metrics/plots/diagnostics. Logs are logs/value-pretrain-after-400.out/.err, logs/ppo-pair-401-800.out/.err and logs/ppo-pair-800-analysis.out/.err.

The two canonical PPO histories are extensions of existing training records, not two independent new experiments. The historical one-epoch warmup has its own combined summary artifact. Further interpretation and next recommendations are stated below after reviewing the measured values and charts.

### Interpretation and next recommendations

The value-only epoch improved historical held-out RMSE by7.42% fullCNN and5.05% residual, but its MSE improvement was largely calibration of the mean prediction. Using MSE = error variance + squared mean error, the measured decomposition is:

| Model | Mean prediction minus target, before / after | Fraction of MSE improvement accounted for by reduced squared bias |
| --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | 0.265804 / 0.021586 | 80.813% |
| residual-c32-h128-518d814dce | -0.213959 / -0.016926 | 77.296% |

This is useful calibration on historical games, with modest increases in explained variance (0.0594→0.0886 fullCNN,0.0340→0.0574 residual). It is not evidence of a strong value predictor for the current self-play policies. Historical train-online and final validation errors are close, which gives no sign of a large train/validation gap in this head-only pass; the online training measurements span changing head weights and are not a final fixed-model training-set evaluation.

Immediate transfer was limited: first10 post-warmup fresh-rollout mean explained variance was0.0060 for fullCNN and0.0169 for residual, versus0.0145/0.0172 over the last50 pre-warmup updates. Different rollout samples and evolving opponents prevent interpreting this as a controlled regression test. By the last50 updates, fullCNN improved to0.0420 with RMSE0.8104 (previously0.0145/0.8218); residual ended0.0107/RMSE0.8213 (previously0.0172/0.8178). Thus the experiment did not resolve the on-policy value issue for both architectures. The frozen linear+tanh head has limited capacity, the shared representation was trained primarily for move prediction, and outcome uncertainty in fog-of-war remains relevant; none of those explanations was causally isolated.

On the repeated100-game benchmark, latest fullCNN improved66→71.5% and residual60.5→63.5%. Selected best fullCNN moved to update680 and scored70.5%; residual retained update210 and scored59%. Residual's stored best retaining an earlier policy despite the latest policy's higher repeated-seed score illustrates how noisy the40-game checkpoint selector can be. Differences of a few points on100 games are unresolved; latest-versus-best rankings should not be treated as established. In particular, no comparison here proves that value pretraining caused the gains:400 additional PPO updates were also performed, without an otherwise matched no-warmup continuation.

Head-to-head during phase2 favored fullCNN56.0625% versus residual43.9375%, with mean rewards+0.1235625 and−0.1189375. The148/6400 draws (2.3125%) all ended by threefold repetition, up from61/6400 (0.9531%) in phase1; there was no ply-cap exploitation in either phase. A rising draw count does not by itself establish stronger, safer or unexploitable moves. Both models still clipped gradients on every phase2 optimizer step while policy KL stayed modest (phase means0.00379/0.00827); no logged execution errors or nonfinite latest checkpoint tensors were found.

Fixed-probe concentration was not repaired by head-only pretraining plus PPO: residual final-trunk top1 activation share was90.09% at800 versus89.16% at400, with56.25% relative dormancy versus59.375%. FullCNN second-trunk top4 share was79.34% at800 versus78.74% at400; dense top4 share91.13% versus88.94%. Probe inputs remain the same within each model, but probes differ between models and are not held-out validation. These are descriptive feature-health measurements, not causal evidence for an architecture change.

Recommended next steps:

1. Prioritize a small critic-only diagnostic using fresh games from the frozen current pair, with game-level held-out data. Compare the learned value against constant-return predictions and inspect error by game stage. This removes historical-policy shift from the fitting task and tests whether the present frozen features/readout can predict current outcomes. Do not assume more historical epochs will solve the on-policy issue.
2. Increase evaluation game counts and use multiple seeds before relying on small differences or selecting another large run. Preserve the current best snapshots and latest continuation states; do not silently overwrite the selected best using this post-hoc benchmark.
3. If current-policy held-out critic fit remains weak, isolate a small nonlinear value readout or dedicated critic optimization phase as the next capacity/optimization ablation, keeping the actor fixed for the diagnostic. A larger shared CNN is not the first conclusion supported by this experiment.

The run and post-completion evaluation are complete. Reports/checkpoints retain the previous experiment's evidence and one persistent RL model file per variant. The monitor is disabled after this final review; no further training or hyperparameter changes were scheduled.

<a id="stage-40"></a>

## Stage 40 — Literature review and proposed critic experiments (2026-11-05; not run)

The user requested literature-supported remedies after stage39. No training settings or checkpoints were changed for this review.

### Evidence and diagnosis

Historical head-only pretraining mainly corrected mean prediction bias (80.8%/77.3% of historical held-out MSE improvement for full/residual). It did not establish strong transfer to current-policy outcomes. The shared encoder is trained at actor LR1e-5; only the final linear+tanh value head receives LR1e-3. Both objectives share four optimization epochs and one global gradient clip0.5. Policy KL stopping also stops value optimization. These are plausible optimization constraints, not measured proof of gradient interference. KL stopping was rare, so it is not a sufficient explanation by itself.

Additional inspection of the last50 updates (751–800), taking unweighted means of each update's logged statistic:

| Model | GAE target EV before / after update | MC outcome EV before / after update | GAE RMSE before / after | MC RMSE before / after |
| --- | --- | --- | --- | --- |
| Full CNN | .103139 / .157312 | .042042 / .077596 | .479001 / .455509 | .809511 / .776243 |
| Residual CNN | .026152 / .054927 | .010650 / .026746 | .483629 / .467178 | .821569 / .797375 |

These are means of per-update metrics, not pooled metrics; slight differences from earlier aggregated RMSE summaries reflect aggregation. After-update measurements reuse the optimization batch and are not held-out evidence. The critic is weak even against its GAE targets, particularly residual. MC prediction is harder and includes outcome noise. Low EV alone does not prove representational collapse or that actor learning has failed. Last50 KL stops: full0, residual3.

### Primary literature

1. Cobbe et al., *Phasic Policy Gradient*, ICML2021, OpenAI: https://proceedings.mlr.press/v139/cobbe21a.html (paper https://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf). Shared policy/value objectives can interfere; value fitting can tolerate more sample reuse than policy fitting. Their Procgen experiments motivate separating optimization budgets. Full PPG also includes auxiliary feature distillation; the proposal below is ordinary PPO with an independent critic, not an implementation of full PPG.
2. Andrychowicz et al., *What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study*: https://arxiv.org/abs/2006.05990. Section3.2 finds separate policy/value networks better in four of five continuous-control environments. This supports a controlled separation experiment but is not a guarantee for fog-of-war chess.
3. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*: https://arxiv.org/abs/1506.02438. GAE lambda trades bootstrap bias against variance. With complete terminal episodes, lambda1 telescopes to discounted observed return minus the pre-action value baseline. This removes dependence on intermediate value predictions, at the cost of higher variance; it does not remove PPO clipping/optimization effects or discounting. Their empirical results often favored intermediate lambda, so lambda1 is a diagnostic, not a universal recommendation.

### Recommended controlled next steps

First collect a modest fresh dataset from the frozen update800 pair and split by entire games. Fit current value readouts against actual discounted terminal returns and compare with a training-set mean baseline, including metrics by game stage. In the same diagnostic, compare an independently trainable critic initialized by copying the current encoder/value weights. Use only the existing fog observations. This distinguishes frozen-feature limitations from historical/current-policy mismatch and tests value representation adaptation without moving the actor or opponent.

The primary proposed PPO intervention is an independent critic encoder and value head with its own optimizer, clipping and update budget. Initialize from the current model, keep the actor architecture/recipe, start the critic at LR1e-4 and eight value epochs versus four actor epochs, and select the fitting budget using the fresh-game diagnostic. Those numbers are experiment starting points, not literature-established optima. Do not raise the entire copied encoder to the current head LR1e-3 without testing. Critic weights and optimizer can remain inside the single canonical RL checkpoint per variant. Extra critic computation is a training cost; deployed actors need no larger network.

A smaller independent ablation is current PPO with lambda1 instead of .95, retaining gamma.99. Our complete-game collection makes this also change critic targets to observed discounted returns. Compare against a matched unchanged continuation from800, rather than combining separation and lambda changes immediately. Evaluate actor strength on fixed opponents with more games/multiple evaluation seeds; compare fresh pre-update and held-out value metrics, not only training loss. Any proposed improvement requires testing: prior actor gains and modest KL do not support calling the whole PPO run unstable solely because clipping is frequent.

<a id="stage-41"></a>

## Stage 41 — Frozen update800 fresh-game critic diagnostic (2026-11-05; completed)

Executed the first experiment proposed in stage40. This is supervised critic regression with frozen game-generating policies, not a PPO continuation or an actor-strength experiment. Both canonical RL checkpoint file hashes were unchanged before/after. No fitted experimental weights were installed or persisted as additional RL checkpoints.

Protocol: 512 unique games from the latest update800 fullCNN/residual pair, alternating colors, temperature0.9, referee max256 plies, seed831705. Both player trajectories come from the same games and use their own existing fog observations. Targets are actual discounted terminal outcomes (+1/-1/draw0.1, gamma0.99 per own decision), not bootstrapped GAE. FullCNN outcomes273W225L14D. All 512 games and both players' return arrays were independently checked against the recorded results and remaining-decision counts.

Game-level split:320 train/96 validation/96 untouched test, balanced by model color, shared game IDs across models; disjoint game IDs verified. FullCNN positions11014/3598/3392; residual11030/3607/3392. Total18004/18029 positions respectively. Related positions within a game are correlated, so game counts are the more relevant independent sample size.

For each model compare: unchanged update800 critic; constant prediction equal to training-set mean return; head-only training with frozen encoder at Adam LR1e-3; independently trainable copy of encoder+value head at Adam LR1e-4. Both fitting arms start from identical update800 weights, use the same minibatch shuffle seed, batch256, Adam eps1e-5/default betas, no weight decay, half-MSE, global clip0.5, 12 epochs. BN running buffers remain frozen and dropout stays off; independent critic BN affine parameters are trainable. Buffer equality and head-only encoder equality verified. Validation MSE selects among epochs1–12; test data are excluded from fitting/selection. This compares two practical fitting recipes, not a pure architecture ablation: their LRs differ and there is one collection/split/training seed.

| Model | Arm | Selected epoch | Train RMSE | Validation RMSE | Test RMSE | Test explained variance |
| --- | --- | --- | --- | --- | --- | --- |
| full_cnn | unchanged | 0 | 0.836246 | 0.826050 | 0.773841 | 0.077551 |
| full_cnn | constant | 0 | 0.825555 | 0.812751 | 0.809554 | 0.000000 |
| full_cnn | head_only | 2 | 0.811732 | 0.795715 | 0.792320 | 0.070027 |
| full_cnn | independent_critic | 2 | 0.801046 | 0.797337 | 0.778022 | 0.072011 |
| residual | unchanged | 0 | 0.823359 | 0.812330 | 0.789714 | 0.044535 |
| residual | constant | 0 | 0.825093 | 0.811746 | 0.810480 | 0.000000 |
| residual | head_only | 1 | 0.819133 | 0.809075 | 0.794726 | 0.044641 |
| residual | independent_critic | 1 | 0.809768 | 0.803515 | 0.789614 | 0.053113 |

### Training trajectories and generalization

- full_cnn, head_only: epoch1 train/validation RMSE 0.816471/0.804473; epoch12 0.816863/0.815350. Epoch12 train/validation EV 0.084529/0.020371; clipping fraction 97.7273%, mean/max raw gradient norm 6.775060/17.616697.
- full_cnn, independent_critic: epoch1 train/validation RMSE 0.809757/0.797818; epoch12 0.714859/0.828175. Epoch12 train/validation EV 0.252966/-0.044986; clipping fraction 100.0000%, mean/max raw gradient norm 7.229157/19.618362.
- residual, head_only: epoch1 train/validation RMSE 0.819133/0.809075; epoch12 0.807103/0.813606. Epoch12 train/validation EV 0.043325/-0.003196; clipping fraction 0.0000%, mean/max raw gradient norm 0.159811/0.474174.
- residual, independent_critic: epoch1 train/validation RMSE 0.809768/0.803515; epoch12 0.702919/0.838219. Epoch12 train/validation EV 0.275411/-0.064015; clipping fraction 100.0000%, mean/max raw gradient norm 15.805703/39.084179.

Independent critics show a clear train/validation divergence on this small fixed dataset. Best epochs are2 fullCNN and1 residual. Better training fit does not produce a demonstrated test improvement over the original critic. This weakens the claim that simply freeing the encoder and adding value epochs is sufficient. It does not disprove separate critics in online PPO with continually refreshed data.

### Paired uncertainty

Paired game-cluster bootstrap,3000 resamples of the96 test games, position-weighted MSE difference (independent minus reference; negative favors independent). These intervals condition on this training run, do not include training-seed variability, and are not adjusted for multiple comparisons.

| Model | Reference | MSE difference | 95% interval |
| --- | --- | --- | --- |
| full_cnn | head_only | -0.022453 | [-0.048184, 0.006806] |
| full_cnn | unchanged | 0.006489 | [-0.042346, 0.051920] |
| full_cnn | constant | -0.050059 | [-0.080487, -0.019742] |
| residual | head_only | -0.008099 | [-0.019374, 0.002778] |
| residual | unchanged | -0.000158 | [-0.013439, 0.012674] |
| residual | constant | -0.033388 | [-0.056904, -0.009642] |

Both independent critics beat the training-mean constant on this test split, but their intervals versus unchanged and head-only critics include zero. FullCNN test results numerically favor the unchanged critic; residual unchanged and independent are almost tied. Validation/test ranking reversals emphasize the uncertainty from96 games per held-out split.

### Distance-to-terminal breakdown

Remaining own decisions are used only for retrospective analysis, never as model input. "last6" includes remaining counts0–5; other bins6–20 and>20. Discounting reduces target magnitudes far from termination, so lower RMSE in distant positions does not imply better discrimination.

| Model | Arm | Remaining decisions | Test positions | RMSE | Explained variance |
| --- | --- | --- | --- | --- | --- |
| full_cnn | unchanged | last_6 | 574 | 0.872318 | 0.191850 |
| full_cnn | unchanged | 6_to_20 | 1334 | 0.826514 | 0.083543 |
| full_cnn | unchanged | over_20 | 1484 | 0.678492 | -0.000402 |
| full_cnn | constant | last_6 | 574 | 0.950383 | -0.000000 |
| full_cnn | constant | 6_to_20 | 1334 | 0.859154 | 0.000000 |
| full_cnn | constant | over_20 | 1484 | 0.696498 | 0.000000 |
| full_cnn | head_only | last_6 | 574 | 0.854498 | 0.191965 |
| full_cnn | head_only | 6_to_20 | 1334 | 0.831744 | 0.077277 |
| full_cnn | head_only | over_20 | 1484 | 0.728431 | -0.008342 |
| full_cnn | independent_critic | last_6 | 574 | 0.839374 | 0.224500 |
| full_cnn | independent_critic | 6_to_20 | 1334 | 0.820776 | 0.082630 |
| full_cnn | independent_critic | over_20 | 1484 | 0.710979 | -0.021726 |
| residual | unchanged | last_6 | 573 | 0.904428 | 0.095708 |
| residual | unchanged | 6_to_20 | 1333 | 0.836897 | 0.048209 |
| residual | unchanged | over_20 | 1486 | 0.692723 | -0.000793 |
| residual | constant | last_6 | 573 | 0.951027 | 0.000000 |
| residual | constant | 6_to_20 | 1333 | 0.859960 | 0.000000 |
| residual | constant | over_20 | 1486 | 0.698047 | 0.000000 |
| residual | head_only | last_6 | 573 | 0.911305 | 0.087864 |
| residual | head_only | 6_to_20 | 1333 | 0.840979 | 0.046699 |
| residual | head_only | over_20 | 1486 | 0.697878 | 0.006383 |
| residual | independent_critic | last_6 | 573 | 0.893299 | 0.120638 |
| residual | independent_critic | 6_to_20 | 1333 | 0.838440 | 0.048828 |
| residual | independent_critic | over_20 | 1486 | 0.696348 | 0.008577 |

Predictive signal is stronger near game termination: unchanged full/residual EV about0.192/0.096 in the last6 decisions, versus approximately zero >20 decisions away. Independent fits reach0.224/0.121 near termination but have no established overall benefit. This pattern is consistent with long-horizon outcome uncertainty/credit-assignment difficulty and limited data; it is not proof of an irreducible information ceiling or a diagnosis of hidden-state insufficiency.

### Takeaways and next recommendations

1. Keep the current actors/checkpoints. This experiment did not train actors and cannot establish a win-rate improvement. Neither more fixed-data critic epochs nor replacing the canonical critic is justified by the observed test results.

2. If pursuing critic fitting, prioritize more fresh independent games rather than replaying320 games for additional epochs. A useful follow-up is4–8x the game count with the same frozen actors, recipe and an untouched enlarged test set, training briefly/early-stopping. A data-scaling improvement would support a sample-size explanation; failure would motivate revisiting representation/targets. This is proposed, not launched.

3. The smaller algorithmic next test remains a matched PPO continuation with lambda1 versus0.95, starting from identical update800 states. Complete-game lambda1 uses observed discounted returns and reduces reliance on intermediate value predictions, at higher variance. Keep other knobs equal, judge fixed-opponent playing strength with sufficient evaluation games, and monitor fresh value prediction rather than only fitting loss. Do not combine it immediately with critic architecture changes. This is proposed, not launched.

4. Longer-term separate-critic optimization remains a hypothesis supported by literature but not validated here. The fixed-data experiment omits actor-gradient interference by freezing the actor in both arms, so it cannot measure the benefit of removing such interference during PPO.

Wall time 125.91s (~2.1min). Artifacts: out/experiments/fresh-update800-critic-head-vs-independent/results.json, fresh_games.npz, split.json, game_reports.json, per-model prediction arrays, validation-curves.png and train-validation-curves.png. Reproduction: work/fresh_critic_diagnostic.py; report generation: work/report_fresh_critic_diagnostic.py. Logs: logs/fresh-critic-diagnostic.out/.err (error log empty). Existing canonical checkpoints unchanged. No background task remains.


<a id="stage-42"></a>

## Stage 42 — Paired PPO continuation with lambda1, updates801–1200 (2026-11-05; launched)

User authorized changing lambda and continuing both existing policies for400 more paired updates. Resumed canonical update800 states with --updates1200 and explicit --gae-lambda1,16 shared games/update, CPU1thread, preserving every other saved setting, optimizer state, RNG, and embedded best actor. No second unchanged-lambda continuation control was requested. Improvements over800 therefore cannot establish a causal lambda effect independently of additional training.

Resume previously prohibited all recipe changes. It now permits an explicit GAE lambda override, retains saved lambda when omitted, records the boundary in training_log.recipe_changes, and records gamma/lambda in each new update's PPO metrics. All other recipe restrictions remain. Regression test exercises resume to lambda1, checks unchanged earlier history and the recorded boundary, and verifies complete-game lambda1 targets equal discounted terminal returns. All20 tests in test_rl passed in9.7seconds.

Full before-run training logs and checkpoint SHA256/settings are preserved in out/experiments/paired-ppo-lambda1-updates801-1200/. Canonical files remain one policy.pt per variant. Launch command: .venv/bin/python -u rl/train_pair.py --first out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt --second out/experiments/residual-medium-27890681/best_policy.pt --updates 1200 --games-per-update 16 --gae-lambda 1 --device cpu --threads 1. Logs: logs/ppo-pair-lambda1-801-1200.out/.err. Exec session12624. Existing monitor reactivated to check completion, evaluate latest/best against the established frozen baselines, append measured results and report recommendations. No duplicate historical value warmup is performed.

Expected6400 additional unique shared training games, with each model updating from its own decisions. Lambda1 changes the regression targets from bootstrapped GAE lambda returns to actual discounted returns, so value-loss scales across this boundary are not directly comparable. MC outcome metrics and fixed-opponent playing strength remain meaningful comparisons. Results pending.

<a id="stage-43"></a>

## Stage 43 — Paired PPO lambda1 continuation through1200 (2026-11-05; completed)

Both canonical policies completed400 additional updates (801–1200),6400 unique shared games,19200 cumulative unique training games. All other saved settings were preserved. The boundary at800 changes lambda0.95→1 and is recorded in both logs. No unchanged-lambda continuation control exists; actor gains cannot be attributed specifically to lambda.

### Fresh-rollout value quality and paired outcomes

RMSE below pools squared errors by decision count; EV is the mean of per-update EV. Predictions are measured before updating on the batch. Opponents evolve throughout, so these are changing data distributions, not fixed validation sets.

| Model | Window | W/L/D | Score | Mean reward | MC RMSE | MC EV |
| --- | --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | pre_change_last50 | 439/341/20 | 56.1250% | 0.125000 | 0.810397 | 0.042042 |
| cnn-c64-h256-8ebd1cf653 | phase2_first10 | 80/78/2 | 50.6250% | 0.013750 | 0.852134 | 0.061559 |
| cnn-c64-h256-8ebd1cf653 | phase2_first50 | 432/353/15 | 54.9375% | 0.100625 | 0.831920 | 0.039785 |
| cnn-c64-h256-8ebd1cf653 | last50 | 423/365/12 | 53.6250% | 0.074000 | 0.837583 | 0.025297 |
| cnn-c64-h256-8ebd1cf653 | phase2 | 3472/2808/120 | 55.1875% | 0.105625 | 0.840182 | 0.024309 |
| residual-c32-h128-518d814dce | pre_change_last50 | 341/439/20 | 43.8750% | -0.120000 | 0.821264 | 0.010650 |
| residual-c32-h128-518d814dce | phase2_first10 | 78/80/2 | 49.3750% | -0.011250 | 0.878532 | 0.000320 |
| residual-c32-h128-518d814dce | phase2_first50 | 353/432/15 | 45.0625% | -0.096875 | 0.840522 | 0.011216 |
| residual-c32-h128-518d814dce | last50 | 365/423/12 | 46.3750% | -0.071000 | 0.830451 | 0.017392 |
| residual-c32-h128-518d814dce | phase2 | 2808/3472/120 | 44.8125% | -0.101875 | 0.830403 | 0.017497 |

Here phase2 means801–1200; pre_change_last50 means751–800; first10/50 means801–810/850; last50 means1151–1200.

### Repeated benchmark

Same seed572940, temperature0.9, max256 plies,50 games against each original frozen baseline (100 per candidate). This is the established benchmark, not an untouched opponent population or seed. Scores count draws0.5, training rewards count draws0.1.

| Candidate | Selected update | W/L/D | Score | Previous800 score | Standard error |
| --- | --- | --- | --- | --- | --- |
| full_cnn_ppo_latest | 1200 | 80/20/0 | 80.0% | 71.5% | 0.0402 |
| full_cnn_ppo_best | 1170 | 67/32/1 | 67.5% | 70.5% | 0.0468 |
| residual_ppo_latest | 1200 | 78/22/0 | 78.0% | 63.5% | 0.0416 |
| residual_ppo_best | 1200 | 78/22/0 | 78.0% | 59.0% | 0.0416 |

### Detailed optimization and feature evidence


#### cnn-c64-h256-8ebd1cf653

```json
{
  "phase_mean_ppo": {
    "policy_loss": -0.011342994863210715,
    "value_loss": 0.32058234785053835,
    "total_loss": 0.1352828049176274,
    "endpoint_kl": 0.0038375417233328336,
    "endpoint_entropy": 1.365629090666771,
    "ratio_clip_fraction": 0.01625756545901708,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 23.330599040516905,
    "gradient_global_max": 55.85536033153534
  },
  "last50_ppo": {
    "policy_loss": -0.011481221285672931,
    "value_loss": 0.3253271966979387,
    "total_loss": 0.1386676628894737,
    "endpoint_kl": 0.003929675635881722,
    "endpoint_entropy": 1.2513597774505616,
    "ratio_clip_fraction": 0.015867705100888068,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 23.53324252048671,
    "gradient_global_max": 54.38846572875977
  },
  "early_stop_updates": 0,
  "termination_reasons": {
    "King capturable": 6280,
    "Threefold repetition": 120
  },
  "optimizer_steps": 7456,
  "decisions": 212932,
  "best_eval": {
    "update": 1170,
    "mean_score": 0.85,
    "mean_reward": 0.7,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 19,
        "losses": 1,
        "draws": 0,
        "failures": 0,
        "score": 0.95,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 9,
            "losses": 1,
            "draws": 0
          },
          "black": {
            "wins": 10,
            "losses": 0,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.9,
        "mean_plies": 58.85,
        "value_monte_carlo": {
          "mse": 0.8343842625617981,
          "rmse": 0.9134463667869568,
          "mae": 0.8914403915405273,
          "explained_variance": -0.025522470474243164,
          "prediction_mean": -0.07725625485181808,
          "prediction_std": 0.16252480447292328,
          "target_mean": 0.7153176665306091,
          "target_std": 0.4484180510044098,
          "samples": 584
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 15,
        "losses": 5,
        "draws": 0,
        "failures": 0,
        "score": 0.75,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 10,
            "losses": 0,
            "draws": 0
          },
          "black": {
            "wins": 5,
            "losses": 5,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.5,
        "mean_plies": 68.75,
        "value_monte_carlo": {
          "mse": 0.7834465503692627,
          "rmse": 0.8851251602172852,
          "mae": 0.8536926507949829,
          "explained_variance": 0.05083513259887695,
          "prediction_mean": -0.1198059543967247,
          "prediction_std": 0.16016212105751038,
          "target_mean": 0.37958431243896484,
          "target_std": 0.7501057982444763,
          "samples": 685
        }
      }
    }
  },
  "final_eval": {
    "update": 1200,
    "mean_score": 0.675,
    "mean_reward": 0.35,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 14,
        "losses": 6,
        "draws": 0,
        "failures": 0,
        "score": 0.7,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 9,
            "losses": 1,
            "draws": 0
          },
          "black": {
            "wins": 5,
            "losses": 5,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.4,
        "mean_plies": 43.5,
        "value_monte_carlo": {
          "mse": 0.6604777574539185,
          "rmse": 0.8126978278160095,
          "mae": 0.7472752928733826,
          "explained_variance": -0.015206098556518555,
          "prediction_mean": 0.261579692363739,
          "prediction_std": 0.18375331163406372,
          "target_mean": 0.3301475942134857,
          "target_std": 0.8037124872207642,
          "samples": 433
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 13,
        "losses": 7,
        "draws": 0,
        "failures": 0,
        "score": 0.65,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 9,
            "losses": 1,
            "draws": 0
          },
          "black": {
            "wins": 4,
            "losses": 6,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.3,
        "mean_plies": 53.65,
        "value_monte_carlo": {
          "mse": 0.6704758405685425,
          "rmse": 0.818825900554657,
          "mae": 0.7630284428596497,
          "explained_variance": 0.07880407571792603,
          "prediction_mean": 0.2635818421840668,
          "prediction_std": 0.18502549827098846,
          "target_mean": 0.12501345574855804,
          "target_std": 0.8408259153366089,
          "samples": 535
        }
      }
    }
  },
  "features800": {
    "encoder.1": {
      "zero_fraction": 0.5113201141357422,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.056928174506235406,
      "top4_activity_share": 0.17353582863053268
    },
    "encoder.3": {
      "zero_fraction": 0.9603657722473145,
      "dormant_fraction": 0.203125,
      "top1_activity_share": 0.24618788131899516,
      "top4_activity_share": 0.7933890833722435
    },
    "encoder.6": {
      "zero_fraction": 0.6371231079101562,
      "dormant_fraction": 0.29296875,
      "top1_activity_share": 0.3035025414857871,
      "top4_activity_share": 0.9113183431530452
    }
  },
  "features1200": {
    "encoder.1": {
      "zero_fraction": 0.5113863945007324,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.05668428258908208,
      "top4_activity_share": 0.1739516882333672
    },
    "encoder.3": {
      "zero_fraction": 0.9575161933898926,
      "dormant_fraction": 0.21875,
      "top1_activity_share": 0.23732393195244889,
      "top4_activity_share": 0.7972560787752949
    },
    "encoder.6": {
      "zero_fraction": 0.6273651123046875,
      "dormant_fraction": 0.17578125,
      "top1_activity_share": 0.3024195593863404,
      "top4_activity_share": 0.9076698504109654
    }
  },
  "verification": {
    "update": 1200,
    "checkpoint_sha256": "4a456b381292161c88e9c64dcd0a41c2e98d5475afa19dc117ba1982ad5ae7d3",
    "source_sha256": "7ba9870bb4151241c50c7d288e892725806f9fab6e7f3844aaa445b812d4656c",
    "finite": true,
    "history_preserved": true,
    "recipe_changes": [
      {
        "after_update": 800,
        "parameter": "gae_lambda",
        "previous": 0.95,
        "value": 1.0
      }
    ]
  }
}
```


#### residual-c32-h128-518d814dce

```json
{
  "phase_mean_ppo": {
    "policy_loss": -0.0113153573723204,
    "value_loss": 0.33643926235251564,
    "total_loss": 0.14267750492779607,
    "endpoint_kl": 0.008809618504019455,
    "endpoint_entropy": 1.4222358056902886,
    "ratio_clip_fraction": 0.04236597805604638,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 158.704356570459,
    "gradient_global_max": 498.1955959701538
  },
  "last50_ppo": {
    "policy_loss": -0.012607227678965716,
    "value_loss": 0.3367980196465745,
    "total_loss": 0.14237902494473806,
    "endpoint_kl": 0.009118794966489076,
    "endpoint_entropy": 1.3390421223640443,
    "ratio_clip_fraction": 0.04379435968876194,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 137.72895090989144,
    "gradient_global_max": 445.7666998291016
  },
  "early_stop_updates": 29,
  "termination_reasons": {
    "King capturable": 6280,
    "Threefold repetition": 120
  },
  "optimizer_steps": 7291,
  "decisions": 213270,
  "best_eval": {
    "update": 1200,
    "mean_score": 0.85,
    "mean_reward": 0.7,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 15,
        "losses": 5,
        "draws": 0,
        "failures": 0,
        "score": 0.75,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 8,
            "losses": 2,
            "draws": 0
          },
          "black": {
            "wins": 7,
            "losses": 3,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.5,
        "mean_plies": 75.75,
        "value_monte_carlo": {
          "mse": 0.7501545548439026,
          "rmse": 0.866114616394043,
          "mae": 0.8334201574325562,
          "explained_variance": 0.07632690668106079,
          "prediction_mean": -0.12789799273014069,
          "prediction_std": 0.16366198658943176,
          "target_mean": 0.3582569658756256,
          "target_std": 0.7458323836326599,
          "samples": 755
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 19,
        "losses": 1,
        "draws": 0,
        "failures": 0,
        "score": 0.95,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 9,
            "losses": 1,
            "draws": 0
          },
          "black": {
            "wins": 10,
            "losses": 0,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.9,
        "mean_plies": 70.15,
        "value_monte_carlo": {
          "mse": 0.8840565085411072,
          "rmse": 0.9402427673339844,
          "mae": 0.923287570476532,
          "explained_variance": -0.08369863033294678,
          "prediction_mean": -0.10998078435659409,
          "prediction_std": 0.15546000003814697,
          "target_mean": 0.748103678226471,
          "target_std": 0.36923739314079285,
          "samples": 697
        }
      }
    }
  },
  "final_eval": {
    "update": 1200,
    "mean_score": 0.85,
    "mean_reward": 0.7,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 15,
        "losses": 5,
        "draws": 0,
        "failures": 0,
        "score": 0.75,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 8,
            "losses": 2,
            "draws": 0
          },
          "black": {
            "wins": 7,
            "losses": 3,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.5,
        "mean_plies": 75.75,
        "value_monte_carlo": {
          "mse": 0.7501545548439026,
          "rmse": 0.866114616394043,
          "mae": 0.8334201574325562,
          "explained_variance": 0.07632690668106079,
          "prediction_mean": -0.12789799273014069,
          "prediction_std": 0.16366198658943176,
          "target_mean": 0.3582569658756256,
          "target_std": 0.7458323836326599,
          "samples": 755
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 19,
        "losses": 1,
        "draws": 0,
        "failures": 0,
        "score": 0.95,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 9,
            "losses": 1,
            "draws": 0
          },
          "black": {
            "wins": 10,
            "losses": 0,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.9,
        "mean_plies": 70.15,
        "value_monte_carlo": {
          "mse": 0.8840565085411072,
          "rmse": 0.9402427673339844,
          "mae": 0.923287570476532,
          "explained_variance": -0.08369863033294678,
          "prediction_mean": -0.10998078435659409,
          "prediction_std": 0.15546000003814697,
          "target_mean": 0.748103678226471,
          "target_std": 0.36923739314079285,
          "samples": 697
        }
      }
    }
  },
  "features800": {
    "encoder.1": {
      "zero_fraction": 0.4465808868408203,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.07583736145427732,
      "top4_activity_share": 0.26040639614203537
    },
    "encoder.2.0.relu1": {
      "zero_fraction": 0.5805130004882812,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.0893322140672368,
      "top4_activity_share": 0.2873770058951226
    },
    "encoder.2.0.relu2": {
      "zero_fraction": 0.7164478302001953,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.10440303351087173,
      "top4_activity_share": 0.38068159712698363
    },
    "encoder.2.1.relu1": {
      "zero_fraction": 0.48381710052490234,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.18134926850490574,
      "top4_activity_share": 0.4975933338669227
    },
    "encoder.2.1.relu2": {
      "zero_fraction": 0.9229230880737305,
      "dormant_fraction": 0.25,
      "top1_activity_share": 0.5226632884113731,
      "top4_activity_share": 0.9017137877637351
    },
    "encoder.2.2.relu1": {
      "zero_fraction": 0.6958322525024414,
      "dormant_fraction": 0.03125,
      "top1_activity_share": 0.1797842369291798,
      "top4_activity_share": 0.4630044462173537
    },
    "encoder.2.2.relu2": {
      "zero_fraction": 0.9569473266601562,
      "dormant_fraction": 0.5625,
      "top1_activity_share": 0.9008608251287143,
      "top4_activity_share": 0.9777536041913703
    },
    "encoder.6": {
      "zero_fraction": 0.4203948974609375,
      "dormant_fraction": 0.1015625,
      "top1_activity_share": 0.050023160822367956,
      "top4_activity_share": 0.13755170849494794
    }
  },
  "features1200": {
    "encoder.1": {
      "zero_fraction": 0.4449586868286133,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.07405003519423335,
      "top4_activity_share": 0.2543121089496898
    },
    "encoder.2.0.relu1": {
      "zero_fraction": 0.5848751068115234,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.0880653695532022,
      "top4_activity_share": 0.29365345811662336
    },
    "encoder.2.0.relu2": {
      "zero_fraction": 0.7070102691650391,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.1633749400028517,
      "top4_activity_share": 0.44672100498451167
    },
    "encoder.2.1.relu1": {
      "zero_fraction": 0.48252296447753906,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.22109280103641973,
      "top4_activity_share": 0.504125527058736
    },
    "encoder.2.1.relu2": {
      "zero_fraction": 0.9228420257568359,
      "dormant_fraction": 0.25,
      "top1_activity_share": 0.514514277469061,
      "top4_activity_share": 0.9016219663817893
    },
    "encoder.2.2.relu1": {
      "zero_fraction": 0.6735725402832031,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.1616185140642077,
      "top4_activity_share": 0.43140818103141443
    },
    "encoder.2.2.relu2": {
      "zero_fraction": 0.9601659774780273,
      "dormant_fraction": 0.5625,
      "top1_activity_share": 0.8989174527950764,
      "top4_activity_share": 0.9801234437013668
    },
    "encoder.6": {
      "zero_fraction": 0.476959228515625,
      "dormant_fraction": 0.1015625,
      "top1_activity_share": 0.06731348542623257,
      "top4_activity_share": 0.1737231813665922
    }
  },
  "verification": {
    "update": 1200,
    "checkpoint_sha256": "ef6a3913091cdfa4998cd0794aaf76b0f37231cb4dd2ec5f1a0864507f9778ba",
    "source_sha256": "203d399c41d13259d6b2d2e78fe34072aa555e7674114521d7ef111d92a8fedd",
    "finite": true,
    "history_preserved": true,
    "recipe_changes": [
      {
        "after_update": 800,
        "parameter": "gae_lambda",
        "previous": 0.95,
        "value": 1.0
      }
    ]
  }
}
```


### Interpretation and next recommendations

The value issue is not resolved. FullCNN late fresh-rollout EV decreased0.0420→0.0253 and RMSE increased0.8104→0.8376; residual EV increased slightly0.0107→0.0174 but RMSE increased0.8213→0.8305. Different return variance and evolving policies matter. The loss increase to roughly0.33 is not directly comparable with the previous roughly0.11: lambda1 trains on full discounted MC returns, a higher-variance target. It is not by itself evidence of divergence.

Both models still clip every optimizer step, yet late policy KL remains modest (~0.00393/full,0.00912/residual); tensors remain finite. Feature concentration persists: full second-conv top4 about79.7%, dense top4 about90.8%; residual final-trunk top1 about89.9%. Lambda1 did not repair concentrated representations or create a strong value predictor.

Actor strength should be judged using the benchmark table rather than paired reward alone. FullCNN won55.19% score within this training phase; residual44.81%. Both opponents change, so this is relative performance. The40-game in-loop best selector remains noisy; a selected-best checkpoint need not outperform the latest on the100-game repeated benchmark.

Keep lambda1 provisionally if continuing, but do not call it superior to0.95 from this uncontrolled continuation. Prioritize a larger multi-seed fixed-opponent evaluation before treating modest score differences or best/latest rankings as reliable. For further critic work, use more independent fresh games and short/early-stopped fitting; stage41 already showed overfitting with320 training games. Do not increase model size or value epochs solely to drive down training loss. A matched lambda comparison remains the clean way to attribute changes if needed. No further training is launched by this report.

Artifacts: out/rl/paired-lambda1-1200-analysis.json, paired-lambda1-1200-evaluation.json, paired-lambda1-1200-comparison.png; out/experiments/paired-ppo-lambda1-updates801-1200/verification.json and preserved800 logs. Checkpoints contain latest optimizer/RNG state and embedded best actors; original supervised source hashes verified unchanged. Older400/800 evidence is preserved.


<a id="stage-44"></a>

## Stage 44 — Current checkpoint round robin at update1200 (2026-11-05)

Scope: current fullCNN and medium-residual supervised baselines plus latest and embedded best RL actors. Inventory found269 .pt files including historical ablations, intermediate checkpoints and nonpolicy tensors; user was asked whether the archive should also be included. This completed tournament covers current checkpoints only. Residual latest1200 and best1200 have identical actor hashes and are represented once, with the alias retained. Five distinct actors,10 matchups,1000 unique games. No training or best-checkpoint promotion performed.

Existing canonical referee and PolicyAgent contract; temperature0.9, max256 plies. Each pair plays100 games (50 per color),20 per seed743291–743295. These seeds differ from the earlier572940 benchmark. Match results include color breakdowns, termination reasons, lengths and seed-level scores.

| Policy | W/L/D | Games | Aggregate score |
| --- | --- | --- | --- |
| Residual RL latest1200 | 249/145/6 | 400 | 63.00% |
| Full CNN RL best1170 | 243/150/7 | 400 | 61.62% |
| Full CNN RL latest1200 | 236/154/10 | 400 | 60.25% |
| Residual supervised | 139/260/1 | 400 | 34.88% |
| Full CNN supervised | 119/277/4 | 400 | 30.25% |

| A | B | A W/L/D | A score | Approximate score SE |
| --- | --- | --- | --- | --- |
| Full CNN supervised | Residual supervised | 45/55/0 | 45.0% | 0.0500 |
| Full CNN supervised | Full CNN RL latest1200 | 27/70/3 | 28.5% | 0.0445 |
| Full CNN supervised | Full CNN RL best1170 | 26/73/1 | 26.5% | 0.0441 |
| Full CNN supervised | Residual RL latest1200 | 21/79/0 | 21.0% | 0.0409 |
| Residual supervised | Full CNN RL latest1200 | 23/77/0 | 23.0% | 0.0423 |
| Residual supervised | Full CNN RL best1170 | 30/69/1 | 30.5% | 0.0460 |
| Residual supervised | Residual RL latest1200 | 31/69/0 | 31.0% | 0.0465 |
| Full CNN RL latest1200 | Full CNN RL best1170 | 45/52/3 | 46.5% | 0.0494 |
| Full CNN RL latest1200 | Residual RL latest1200 | 44/52/4 | 46.0% | 0.0491 |
| Full CNN RL best1170 | Residual RL latest1200 | 49/49/2 | 50.0% | 0.0497 |

Aggregate ranking weights each of the other four actors equally; it is specific to this opponent set, not an absolute skill estimate. Pairwise samples of100 games leave uncertainty, so small differences do not establish superiority. No checkpoint was promoted using these post-hoc results. All games completed without bot failure. Residual latest/best is one actor, not two independent competitors.

Wall time60.7s. Full results and roster hashes: out/experiments/current-checkpoints-round-robin-update1200/results.json; chart: score-matrix.png. Reproduction scripts: work/checkpoint_round_robin.py, work/current-checkpoint-roster.json, work/report_checkpoint_round_robin.py. Logs: logs/current-checkpoints-round-robin.out/.err.

Interpretation: all three distinct RL actors substantially outperformed both supervised baselines (69–79 wins per100, with a few draws). Their direct matches were close: full latest versus full best45W52L3D; full latest versus residual44W52L4D; full best versus residual49W49L2D. No clear best RL actor is established. The previous single-seed80% versus67.5% full latest/best baseline ranking is not a stable estimate: on these new seeds their average supervised-opponent scores were74.25% and71.5%, respectively. Retain the incumbent fullCNN best rather than promoting latest on inconclusive evidence; continue training from latest separately. A larger direct comparison would be needed for confident promotion. Overall residual63.0%, full best61.625%, full latest60.25% rankings are descriptive and do not establish significant differences.

## Complete current round-robin metrics

Includes roster paths and actor hashes, duplicate aliases, W/L/D, colors, termination reasons, lengths, all five seed scores, and standings.

```json
{
  "status": "completed",
  "temperature": 0.9,
  "max_plies": 256,
  "games_per_pair": 100,
  "seeds": [
    743291,
    743292,
    743293,
    743294,
    743295
  ],
  "roster": {
    "Full CNN supervised": {
      "path": "out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt",
      "best": false,
      "update": null,
      "actor_sha256": "108e0c0e1b6c8701278d55ac11a000ebac118540d299a8e80c7774fc354b5a9b",
      "aliases": []
    },
    "Residual supervised": {
      "path": "out/experiments/residual-medium-27890681/best_policy.pt",
      "best": false,
      "update": null,
      "actor_sha256": "511fb6752d73cd8f62cbf79126b8c064fd18b8ee4165c3ca8dcc1c1209f79bd4",
      "aliases": []
    },
    "Full CNN RL latest1200": {
      "path": "out/rl/cnn-c64-h256-8ebd1cf653/policy.pt",
      "best": false,
      "update": 1200,
      "actor_sha256": "b5120b8b4ce748ecf45e2760a024c50ee4c25563976790e69dc857b87ec7abe8",
      "aliases": []
    },
    "Full CNN RL best1170": {
      "path": "out/rl/cnn-c64-h256-8ebd1cf653/policy.pt",
      "best": true,
      "update": 1170,
      "actor_sha256": "e10f77cb77f295fcaf52c122451b4731cd5b48f516508eba4ddfc885a6a041fa",
      "aliases": []
    },
    "Residual RL latest1200": {
      "path": "out/rl/residual-c32-h128-518d814dce/policy.pt",
      "best": false,
      "update": 1200,
      "actor_sha256": "ce9bf1f57840f3cfeffdd80a9aa0124ee9fd588360774d7dc68fd93a761d7ce6",
      "aliases": [
        "Residual RL best1200"
      ]
    }
  },
  "matches": [
    {
      "a": "Full CNN supervised",
      "b": "Residual supervised",
      "wins": 45,
      "losses": 55,
      "draws": 0,
      "score": 0.45,
      "by_color": {
        "white": {
          "wins": 23,
          "losses": 27,
          "draws": 0
        },
        "black": {
          "wins": 22,
          "losses": 28,
          "draws": 0
        }
      },
      "reasons": {
        "King capturable": 100
      },
      "mean_plies": 66.57,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.65
        },
        {
          "seed": 743292,
          "score": 0.3
        },
        {
          "seed": 743293,
          "score": 0.45
        },
        {
          "seed": 743294,
          "score": 0.55
        },
        {
          "seed": 743295,
          "score": 0.3
        }
      ]
    },
    {
      "a": "Full CNN supervised",
      "b": "Full CNN RL latest1200",
      "wins": 27,
      "losses": 70,
      "draws": 3,
      "score": 0.285,
      "by_color": {
        "white": {
          "wins": 17,
          "losses": 31,
          "draws": 2
        },
        "black": {
          "wins": 10,
          "losses": 39,
          "draws": 1
        }
      },
      "reasons": {
        "King capturable": 97,
        "Threefold repetition": 3
      },
      "mean_plies": 62.83,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.275
        },
        {
          "seed": 743292,
          "score": 0.2
        },
        {
          "seed": 743293,
          "score": 0.3
        },
        {
          "seed": 743294,
          "score": 0.375
        },
        {
          "seed": 743295,
          "score": 0.275
        }
      ]
    },
    {
      "a": "Full CNN supervised",
      "b": "Full CNN RL best1170",
      "wins": 26,
      "losses": 73,
      "draws": 1,
      "score": 0.265,
      "by_color": {
        "white": {
          "wins": 15,
          "losses": 34,
          "draws": 1
        },
        "black": {
          "wins": 11,
          "losses": 39,
          "draws": 0
        }
      },
      "reasons": {
        "King capturable": 99,
        "Threefold repetition": 1
      },
      "mean_plies": 56.34,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.25
        },
        {
          "seed": 743292,
          "score": 0.3
        },
        {
          "seed": 743293,
          "score": 0.25
        },
        {
          "seed": 743294,
          "score": 0.375
        },
        {
          "seed": 743295,
          "score": 0.15
        }
      ]
    },
    {
      "a": "Full CNN supervised",
      "b": "Residual RL latest1200",
      "wins": 21,
      "losses": 79,
      "draws": 0,
      "score": 0.21,
      "by_color": {
        "white": {
          "wins": 9,
          "losses": 41,
          "draws": 0
        },
        "black": {
          "wins": 12,
          "losses": 38,
          "draws": 0
        }
      },
      "reasons": {
        "King capturable": 100
      },
      "mean_plies": 71.31,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.25
        },
        {
          "seed": 743292,
          "score": 0.3
        },
        {
          "seed": 743293,
          "score": 0.25
        },
        {
          "seed": 743294,
          "score": 0.15
        },
        {
          "seed": 743295,
          "score": 0.1
        }
      ]
    },
    {
      "a": "Residual supervised",
      "b": "Full CNN RL latest1200",
      "wins": 23,
      "losses": 77,
      "draws": 0,
      "score": 0.23,
      "by_color": {
        "white": {
          "wins": 15,
          "losses": 35,
          "draws": 0
        },
        "black": {
          "wins": 8,
          "losses": 42,
          "draws": 0
        }
      },
      "reasons": {
        "King capturable": 100
      },
      "mean_plies": 55.75,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.2
        },
        {
          "seed": 743292,
          "score": 0.15
        },
        {
          "seed": 743293,
          "score": 0.25
        },
        {
          "seed": 743294,
          "score": 0.1
        },
        {
          "seed": 743295,
          "score": 0.45
        }
      ]
    },
    {
      "a": "Residual supervised",
      "b": "Full CNN RL best1170",
      "wins": 30,
      "losses": 69,
      "draws": 1,
      "score": 0.305,
      "by_color": {
        "white": {
          "wins": 17,
          "losses": 33,
          "draws": 0
        },
        "black": {
          "wins": 13,
          "losses": 36,
          "draws": 1
        }
      },
      "reasons": {
        "King capturable": 99,
        "Threefold repetition": 1
      },
      "mean_plies": 55.24,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.375
        },
        {
          "seed": 743292,
          "score": 0.2
        },
        {
          "seed": 743293,
          "score": 0.45
        },
        {
          "seed": 743294,
          "score": 0.25
        },
        {
          "seed": 743295,
          "score": 0.25
        }
      ]
    },
    {
      "a": "Residual supervised",
      "b": "Residual RL latest1200",
      "wins": 31,
      "losses": 69,
      "draws": 0,
      "score": 0.31,
      "by_color": {
        "white": {
          "wins": 15,
          "losses": 35,
          "draws": 0
        },
        "black": {
          "wins": 16,
          "losses": 34,
          "draws": 0
        }
      },
      "reasons": {
        "King capturable": 100
      },
      "mean_plies": 68.67,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.35
        },
        {
          "seed": 743292,
          "score": 0.2
        },
        {
          "seed": 743293,
          "score": 0.2
        },
        {
          "seed": 743294,
          "score": 0.35
        },
        {
          "seed": 743295,
          "score": 0.45
        }
      ]
    },
    {
      "a": "Full CNN RL latest1200",
      "b": "Full CNN RL best1170",
      "wins": 45,
      "losses": 52,
      "draws": 3,
      "score": 0.465,
      "by_color": {
        "white": {
          "wins": 22,
          "losses": 26,
          "draws": 2
        },
        "black": {
          "wins": 23,
          "losses": 26,
          "draws": 1
        }
      },
      "reasons": {
        "King capturable": 97,
        "Threefold repetition": 3
      },
      "mean_plies": 66.81,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.55
        },
        {
          "seed": 743292,
          "score": 0.475
        },
        {
          "seed": 743293,
          "score": 0.475
        },
        {
          "seed": 743294,
          "score": 0.4
        },
        {
          "seed": 743295,
          "score": 0.425
        }
      ]
    },
    {
      "a": "Full CNN RL latest1200",
      "b": "Residual RL latest1200",
      "wins": 44,
      "losses": 52,
      "draws": 4,
      "score": 0.46,
      "by_color": {
        "white": {
          "wins": 20,
          "losses": 27,
          "draws": 3
        },
        "black": {
          "wins": 24,
          "losses": 25,
          "draws": 1
        }
      },
      "reasons": {
        "King capturable": 96,
        "Threefold repetition": 4
      },
      "mean_plies": 73.31,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.375
        },
        {
          "seed": 743292,
          "score": 0.525
        },
        {
          "seed": 743293,
          "score": 0.575
        },
        {
          "seed": 743294,
          "score": 0.35
        },
        {
          "seed": 743295,
          "score": 0.475
        }
      ]
    },
    {
      "a": "Full CNN RL best1170",
      "b": "Residual RL latest1200",
      "wins": 49,
      "losses": 49,
      "draws": 2,
      "score": 0.5,
      "by_color": {
        "white": {
          "wins": 21,
          "losses": 27,
          "draws": 2
        },
        "black": {
          "wins": 28,
          "losses": 22,
          "draws": 0
        }
      },
      "reasons": {
        "King capturable": 98,
        "Threefold repetition": 2
      },
      "mean_plies": 67.87,
      "seed_scores": [
        {
          "seed": 743291,
          "score": 0.375
        },
        {
          "seed": 743292,
          "score": 0.325
        },
        {
          "seed": 743293,
          "score": 0.6
        },
        {
          "seed": 743294,
          "score": 0.55
        },
        {
          "seed": 743295,
          "score": 0.65
        }
      ]
    }
  ],
  "wall_seconds": 60.72525596618652,
  "standings": {
    "Full CNN supervised": {
      "wins": 119,
      "losses": 277,
      "draws": 4,
      "games": 400,
      "score": 0.3025
    },
    "Residual supervised": {
      "wins": 139,
      "losses": 260,
      "draws": 1,
      "games": 400,
      "score": 0.34875
    },
    "Full CNN RL latest1200": {
      "wins": 236,
      "losses": 154,
      "draws": 10,
      "games": 400,
      "score": 0.6025
    },
    "Full CNN RL best1170": {
      "wins": 243,
      "losses": 150,
      "draws": 7,
      "games": 400,
      "score": 0.61625
    },
    "Residual RL latest1200": {
      "wins": 249,
      "losses": 145,
      "draws": 6,
      "games": 400,
      "score": 0.63
    }
  }
}
```

## Primary artifact index

- [First400 analysis](out/rl/paired-400-analysis.json) and [fresh evaluation](out/rl/paired-400-fresh-evaluation.json).
- [Historical value warmup](out/rl/value-pretrain-after-400.json).
- [After800 analysis](out/rl/paired-after-value-warmup-800-analysis.json) and [evaluation](out/rl/paired-after-value-warmup-800-evaluation.json).
- [Fresh critic diagnostic](out/experiments/fresh-update800-critic-head-vs-independent/results.json), including epoch-level train/validation metrics, test-by-stage errors and bootstrap intervals.
- [Lambda1 through1200 analysis](out/rl/paired-lambda1-1200-analysis.json), [evaluation](out/rl/paired-lambda1-1200-evaluation.json), and [verification](out/experiments/paired-ppo-lambda1-updates801-1200/verification.json).
- [Current round robin](out/experiments/current-checkpoints-round-robin-update1200/results.json).
- Full1200-update histories, losses, value errors, norms, entropy, KL, outcomes, timings, evaluations and probe references: [fullCNN metrics](out/rl/cnn-c64-h256-8ebd1cf653/metrics.json), [residual metrics](out/rl/residual-c32-h128-518d814dce/metrics.json).
- [RL operation and checkpoint usage](rl/README.md). This is operational documentation, not a parallel findings report.

Full per-update JSON is linked rather than duplicating1200 large records per model in Markdown. Historical smoke-run results, implementation checks, literature links and source-model provenance remain in the chronological stages above.

<a id="stage-45"></a>

## 45. Paired PPO continuation1201–2200 (2026-11-05; completed)

User ran the provided resume command, retaining lambda1 and the existing recipe. Both models completed1000 additional updates,16000 unique shared training games,35200 cumulative unique training games. No architecture, LR, optimizer, draw reward, opponent policy or warmup change was introduced. This is a continuation of the same trajectory, not an independent seed.

### Repeated frozen-baseline benchmark

Post-completion evaluation: seed572940,50 games against each original supervised baseline,100 per candidate, temperature0.9, max256 plies. Same protocol as the400/800/1200 benchmark; these are repeated known opponents/seed. Score counts draws0.5. The table is not the stage44 all-opponent round-robin aggregate.

| Candidate | Selected update | W/L/D | Score at2200 | Corresponding score at1200 | Nominal score SE |
| --- | --- | --- | --- | --- | --- |
| full_cnn_ppo_latest | 2200 | 82/17/1 | 82.5% | 80.0% | 0.0379 |
| full_cnn_ppo_best | 1170 | 67/32/1 | 67.5% | 67.5% | 0.0468 |
| residual_ppo_latest | 2200 | 58/41/1 | 58.5% | 78.0% | 0.0493 |
| residual_ppo_best | 1200 | 78/22/0 | 78.0% | 78.0% | 0.0416 |

### Fresh training-rollout windows

Pre-update MC RMSE pools squared error by decision count; EV averages per-update EV. Each50-update window contains800 shared games. Policies, opponents and return distributions evolve, so these are not fixed validation-set errors.

| Model | Window | W/L/D | Score | Mean reward | MC RMSE | MC EV |
| --- | --- | --- | --- | --- | --- | --- |
| cnn-c64-h256-8ebd1cf653 | pre_change_last50 | 423/365/12 | 53.6250% | 0.074000 | 0.837583 | 0.025297 |
| cnn-c64-h256-8ebd1cf653 | phase2_first50 | 386/385/29 | 50.0625% | 0.004875 | 0.824027 | 0.025915 |
| cnn-c64-h256-8ebd1cf653 | last50 | 481/289/30 | 62.0000% | 0.243750 | 0.808479 | 0.038663 |
| cnn-c64-h256-8ebd1cf653 | phase2 | 8687/6903/410 | 55.5750% | 0.114062 | 0.830589 | 0.027719 |
| residual-c32-h128-518d814dce | pre_change_last50 | 365/423/12 | 46.3750% | -0.071000 | 0.830451 | 0.017392 |
| residual-c32-h128-518d814dce | phase2_first50 | 385/386/29 | 49.9375% | 0.002375 | 0.830258 | 0.023113 |
| residual-c32-h128-518d814dce | last50 | 289/481/30 | 38.0000% | -0.236250 | 0.801668 | 0.022160 |
| residual-c32-h128-518d814dce | phase2 | 6903/8687/410 | 44.4250% | -0.108938 | 0.821731 | 0.017500 |

For this analysis artifact pre_change_last50=1151–1200, phase2_first50=1201–1250, last50=2151–2200, phase2=1201–2200. Gamma/lambda are unchanged at0.99/1.

### Final frozen head-to-head

100 games, five seeds743291–743295,50 games per color; same protocol as stage44 latest-versus-latest.

```json
{
  "a": "Full CNN RL latest2200",
  "b": "Residual RL latest2200",
  "wins": 54,
  "losses": 43,
  "draws": 3,
  "score": 0.555,
  "by_color": {
    "white": {
      "wins": 34,
      "losses": 15,
      "draws": 1
    },
    "black": {
      "wins": 20,
      "losses": 28,
      "draws": 2
    }
  },
  "reasons": {
    "King capturable": 97,
    "Threefold repetition": 3
  },
  "mean_plies": 76.31,
  "seed_scores": [
    {
      "seed": 743291,
      "score": 0.55
    },
    {
      "seed": 743292,
      "score": 0.4
    },
    {
      "seed": 743293,
      "score": 0.675
    },
    {
      "seed": 743294,
      "score": 0.525
    },
    {
      "seed": 743295,
      "score": 0.625
    }
  ]
}
```

### Optimization, feature health and verification


#### cnn-c64-h256-8ebd1cf653

```json
{
  "phase_ppo": {
    "policy_loss": -0.011570924306611866,
    "value_loss": 0.310514960639097,
    "total_loss": 0.13047865206645348,
    "endpoint_kl": 0.003828349356073886,
    "endpoint_entropy": 1.3201950360536576,
    "ratio_clip_fraction": 0.017418791295320035,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 23.307913818842923,
    "gradient_global_max": 58.34077270889282
  },
  "last50_ppo": {
    "policy_loss": -0.011402719680735833,
    "value_loss": 0.29386623390770616,
    "total_loss": 0.12150575554615331,
    "endpoint_kl": 0.003991724716033786,
    "endpoint_entropy": 1.402155249118805,
    "ratio_clip_fraction": 0.01739006962888529,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 22.63875496520841,
    "gradient_global_max": 55.27048973083496
  },
  "early_stop_updates": 6,
  "largest_raw_gradient": 961.3421630859375,
  "largest_endpoint_kl": 0.014484427869319916,
  "optimizer_steps": 20076,
  "decisions": 579180,
  "mean_plies": 72.4573125,
  "termination_reasons": {
    "King capturable": 15590,
    "Threefold repetition": 409,
    "Move limit reached": 1
  },
  "final_evaluation": {
    "update": 2200,
    "mean_score": 0.75,
    "mean_reward": 0.505,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 13,
        "losses": 5,
        "draws": 2,
        "failures": 0,
        "score": 0.7,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 7,
            "losses": 1,
            "draws": 2
          },
          "black": {
            "wins": 6,
            "losses": 4,
            "draws": 0
          }
        },
        "reasons": {
          "Threefold repetition": 1,
          "King capturable": 18,
          "Move limit reached": 1
        },
        "mean_reward": 0.41,
        "mean_plies": 79.35,
        "value_monte_carlo": {
          "mse": 0.5280382633209229,
          "rmse": 0.7266623973846436,
          "mae": 0.6364797353744507,
          "explained_variance": -0.09489107131958008,
          "prediction_mean": 0.06514744460582733,
          "prediction_std": 0.3360937237739563,
          "target_mean": 0.2723221778869629,
          "target_std": 0.6656374335289001,
          "samples": 791
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 16,
        "losses": 4,
        "draws": 0,
        "failures": 0,
        "score": 0.8,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 8,
            "losses": 2,
            "draws": 0
          },
          "black": {
            "wins": 8,
            "losses": 2,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.6,
        "mean_plies": 57.8,
        "value_monte_carlo": {
          "mse": 0.5907264351844788,
          "rmse": 0.7685872912406921,
          "mae": 0.70185387134552,
          "explained_variance": -0.14932847023010254,
          "prediction_mean": 0.15574923157691956,
          "prediction_std": 0.26876604557037354,
          "target_mean": 0.6168216466903687,
          "target_std": 0.5735924243927002,
          "samples": 575
        }
      }
    }
  },
  "best_evaluation": {
    "update": 1170,
    "mean_score": 0.85,
    "mean_reward": 0.7,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 19,
        "losses": 1,
        "draws": 0,
        "failures": 0,
        "score": 0.95,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 9,
            "losses": 1,
            "draws": 0
          },
          "black": {
            "wins": 10,
            "losses": 0,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.9,
        "mean_plies": 58.85,
        "value_monte_carlo": {
          "mse": 0.8343842625617981,
          "rmse": 0.9134463667869568,
          "mae": 0.8914403915405273,
          "explained_variance": -0.025522470474243164,
          "prediction_mean": -0.07725625485181808,
          "prediction_std": 0.16252480447292328,
          "target_mean": 0.7153176665306091,
          "target_std": 0.4484180510044098,
          "samples": 584
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 15,
        "losses": 5,
        "draws": 0,
        "failures": 0,
        "score": 0.75,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 10,
            "losses": 0,
            "draws": 0
          },
          "black": {
            "wins": 5,
            "losses": 5,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.5,
        "mean_plies": 68.75,
        "value_monte_carlo": {
          "mse": 0.7834465503692627,
          "rmse": 0.8851251602172852,
          "mae": 0.8536926507949829,
          "explained_variance": 0.05083513259887695,
          "prediction_mean": -0.1198059543967247,
          "prediction_std": 0.16016212105751038,
          "target_mean": 0.37958431243896484,
          "target_std": 0.7501057982444763,
          "samples": 685
        }
      }
    }
  },
  "features1200": {
    "encoder.1": {
      "zero_fraction": 0.5113863945007324,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.05668428258908208,
      "top4_activity_share": 0.1739516882333672
    },
    "encoder.3": {
      "zero_fraction": 0.9575161933898926,
      "dormant_fraction": 0.21875,
      "top1_activity_share": 0.23732393195244889,
      "top4_activity_share": 0.7972560787752949
    },
    "encoder.6": {
      "zero_fraction": 0.6273651123046875,
      "dormant_fraction": 0.17578125,
      "top1_activity_share": 0.3024195593863404,
      "top4_activity_share": 0.9076698504109654
    }
  },
  "features2200": {
    "encoder.1": {
      "zero_fraction": 0.5114965438842773,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.05651645244555318,
      "top4_activity_share": 0.1749294645972145
    },
    "encoder.3": {
      "zero_fraction": 0.9595170021057129,
      "dormant_fraction": 0.1875,
      "top1_activity_share": 0.2502503366212013,
      "top4_activity_share": 0.8025328316965644
    },
    "encoder.6": {
      "zero_fraction": 0.6435546875,
      "dormant_fraction": 0.19921875,
      "top1_activity_share": 0.30458393251690097,
      "top4_activity_share": 0.9136504820753797
    }
  },
  "verification": {
    "update": 2200,
    "source_sha256": "7ba9870bb4151241c50c7d288e892725806f9fab6e7f3844aaa445b812d4656c",
    "canonical_sha256": "b06b765441fcf4335923d765f3e6d1e704d4f6b92f34c2f878e29fb92bf19352",
    "finite": true,
    "one_rl_checkpoint": true,
    "update1200_row_preserved": true
  }
}
```


#### residual-c32-h128-518d814dce

```json
{
  "phase_ppo": {
    "policy_loss": -0.012005753600140426,
    "value_loss": 0.32791692026354086,
    "total_loss": 0.13818075408206718,
    "endpoint_kl": 0.00829648078524042,
    "endpoint_entropy": 1.3764951940774917,
    "ratio_clip_fraction": 0.04189782436527317,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 138.16229518864068,
    "gradient_global_max": 441.32716264343264
  },
  "last50_ppo": {
    "policy_loss": -0.011877679651914804,
    "value_loss": 0.31153947682020233,
    "total_loss": 0.13017941596251878,
    "endpoint_kl": 0.009696955597028135,
    "endpoint_entropy": 1.3731897211074828,
    "ratio_clip_fraction": 0.04748548820081166,
    "gradient_clipped_fraction": 1.0,
    "pre_clip_gradient_l2": 143.6267972299387,
    "gradient_global_max": 378.21587951660155
  },
  "early_stop_updates": 34,
  "largest_raw_gradient": 5890.8232421875,
  "largest_endpoint_kl": 0.03055417351424694,
  "optimizer_steps": 19941,
  "decisions": 580137,
  "mean_plies": 72.4573125,
  "termination_reasons": {
    "King capturable": 15590,
    "Threefold repetition": 409,
    "Move limit reached": 1
  },
  "final_evaluation": {
    "update": 2200,
    "mean_score": 0.65,
    "mean_reward": 0.3,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 16,
        "losses": 4,
        "draws": 0,
        "failures": 0,
        "score": 0.8,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 8,
            "losses": 2,
            "draws": 0
          },
          "black": {
            "wins": 8,
            "losses": 2,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.6,
        "mean_plies": 65.2,
        "value_monte_carlo": {
          "mse": 0.9204868674278259,
          "rmse": 0.9594200849533081,
          "mae": 0.942147433757782,
          "explained_variance": -0.060378313064575195,
          "prediction_mean": -0.1528095155954361,
          "prediction_std": 0.15677547454833984,
          "target_mean": 0.5524688959121704,
          "target_std": 0.6316482424736023,
          "samples": 649
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 10,
        "losses": 10,
        "draws": 0,
        "failures": 0,
        "score": 0.5,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 6,
            "losses": 4,
            "draws": 0
          },
          "black": {
            "wins": 4,
            "losses": 6,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.0,
        "mean_plies": 60.2,
        "value_monte_carlo": {
          "mse": 0.7528155446052551,
          "rmse": 0.8676494359970093,
          "mae": 0.8365158438682556,
          "explained_variance": 0.01839500665664673,
          "prediction_mean": -0.14039821922779083,
          "prediction_std": 0.16226746141910553,
          "target_mean": 0.036664120852947235,
          "target_std": 0.8573123812675476,
          "samples": 602
        }
      }
    }
  },
  "best_evaluation": {
    "update": 1200,
    "mean_score": 0.85,
    "mean_reward": 0.7,
    "opponents": {
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/residual-medium-27890681/best_policy.pt": {
        "wins": 15,
        "losses": 5,
        "draws": 0,
        "failures": 0,
        "score": 0.75,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 8,
            "losses": 2,
            "draws": 0
          },
          "black": {
            "wins": 7,
            "losses": 3,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.5,
        "mean_plies": 75.75,
        "value_monte_carlo": {
          "mse": 0.7501545548439026,
          "rmse": 0.866114616394043,
          "mae": 0.8334201574325562,
          "explained_variance": 0.07632690668106079,
          "prediction_mean": -0.12789799273014069,
          "prediction_std": 0.16366198658943176,
          "target_mean": 0.3582569658756256,
          "target_std": 0.7458323836326599,
          "samples": 755
        }
      },
      "/Users/jesuscaraball0/code/school_code/67980/fow/out/experiments/full-data-plain-cnn-adamw-bn-8epochs-job27800942/best_policy.pt": {
        "wins": 19,
        "losses": 1,
        "draws": 0,
        "failures": 0,
        "score": 0.95,
        "games": 20,
        "by_color": {
          "white": {
            "wins": 9,
            "losses": 1,
            "draws": 0
          },
          "black": {
            "wins": 10,
            "losses": 0,
            "draws": 0
          }
        },
        "reasons": {
          "King capturable": 20
        },
        "mean_reward": 0.9,
        "mean_plies": 70.15,
        "value_monte_carlo": {
          "mse": 0.8840565085411072,
          "rmse": 0.9402427673339844,
          "mae": 0.923287570476532,
          "explained_variance": -0.08369863033294678,
          "prediction_mean": -0.10998078435659409,
          "prediction_std": 0.15546000003814697,
          "target_mean": 0.748103678226471,
          "target_std": 0.36923739314079285,
          "samples": 697
        }
      }
    }
  },
  "features1200": {
    "encoder.1": {
      "zero_fraction": 0.4449586868286133,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.07405003519423335,
      "top4_activity_share": 0.2543121089496898
    },
    "encoder.2.0.relu1": {
      "zero_fraction": 0.5848751068115234,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.0880653695532022,
      "top4_activity_share": 0.29365345811662336
    },
    "encoder.2.0.relu2": {
      "zero_fraction": 0.7070102691650391,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.1633749400028517,
      "top4_activity_share": 0.44672100498451167
    },
    "encoder.2.1.relu1": {
      "zero_fraction": 0.48252296447753906,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.22109280103641973,
      "top4_activity_share": 0.504125527058736
    },
    "encoder.2.1.relu2": {
      "zero_fraction": 0.9228420257568359,
      "dormant_fraction": 0.25,
      "top1_activity_share": 0.514514277469061,
      "top4_activity_share": 0.9016219663817893
    },
    "encoder.2.2.relu1": {
      "zero_fraction": 0.6735725402832031,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.1616185140642077,
      "top4_activity_share": 0.43140818103141443
    },
    "encoder.2.2.relu2": {
      "zero_fraction": 0.9601659774780273,
      "dormant_fraction": 0.5625,
      "top1_activity_share": 0.8989174527950764,
      "top4_activity_share": 0.9801234437013668
    },
    "encoder.6": {
      "zero_fraction": 0.476959228515625,
      "dormant_fraction": 0.1015625,
      "top1_activity_share": 0.06731348542623257,
      "top4_activity_share": 0.1737231813665922
    }
  },
  "features2200": {
    "encoder.1": {
      "zero_fraction": 0.4503917694091797,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.07635926117248898,
      "top4_activity_share": 0.2584614446610005
    },
    "encoder.2.0.relu1": {
      "zero_fraction": 0.5664234161376953,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.09083958709518158,
      "top4_activity_share": 0.2779435996467783
    },
    "encoder.2.0.relu2": {
      "zero_fraction": 0.726008415222168,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.19481757947717163,
      "top4_activity_share": 0.48659553170813497
    },
    "encoder.2.1.relu1": {
      "zero_fraction": 0.49628257751464844,
      "dormant_fraction": 0.0,
      "top1_activity_share": 0.24025661285812427,
      "top4_activity_share": 0.5080069204570928
    },
    "encoder.2.1.relu2": {
      "zero_fraction": 0.9206752777099609,
      "dormant_fraction": 0.28125,
      "top1_activity_share": 0.498007835806129,
      "top4_activity_share": 0.9053716232957082
    },
    "encoder.2.2.relu1": {
      "zero_fraction": 0.6726474761962891,
      "dormant_fraction": 0.03125,
      "top1_activity_share": 0.1510794207441133,
      "top4_activity_share": 0.4192257276888645
    },
    "encoder.2.2.relu2": {
      "zero_fraction": 0.9514760971069336,
      "dormant_fraction": 0.46875,
      "top1_activity_share": 0.892261653593173,
      "top4_activity_share": 0.979898024403335
    },
    "encoder.6": {
      "zero_fraction": 0.54205322265625,
      "dormant_fraction": 0.1015625,
      "top1_activity_share": 0.07836352030151748,
      "top4_activity_share": 0.19451994696011451
    }
  },
  "verification": {
    "update": 2200,
    "source_sha256": "203d399c41d13259d6b2d2e78fe34072aa555e7674114521d7ef111d92a8fedd",
    "canonical_sha256": "c7b114b2514feaa5976169566ab11b0c0ff4d01d104086c9b49635e0379b3b44",
    "finite": true,
    "one_rl_checkpoint": true,
    "update1200_row_preserved": true
  }
}
```


### Takeaways and next steps

1. Additional training did not establish a large, consistent frozen-baseline strength gain. Latest fullCNN moved80→82.5%; latest residual78→58.5% on the repeated100-game benchmark. The fullCNN change is small relative to sampling uncertainty. Residual declined19.5 percentage points, a material warning of benchmark regression; confirm across more seeds before attributing it to forgetting or a specific training defect. This does not support another long unchanged continuation for both models.

2. FullCNN gained relative ground against residual: phase score55.575%, late50-update score62%. The independent final100-game match was54W43L3D for fullCNN (55.5%), compared with44W52L4D (46%) at1200 on the same five-seed protocol. Both samples remain uncertain and compare two changing actors; this does not isolate absolute improvement or residual forgetting.

3. Critics improved modestly in error but remain weak. Late MC RMSE0.8376→0.8085/full and0.8305→0.8017/residual, EV0.0253→0.0387 and0.0174→0.0222. Better RMSE can also reflect changed outcome/trajectory distributions. The next intervention should be measured against fresh outcome metrics and playing strength, not training loss alone.

4. Gradient clipping remains100%; late KL about0.0040/full and0.0097/residual, finite checkpoints, no logged execution errors. Raw gradient spikes (phase maxima961/full,5891/residual) did not imply uncontrolled policy updates. Feature concentration persists: full secondconv top4 about80.3%, dense top4 about91.4%; residual lasttrunk top1 about89.2%, dormancy46.875%. No evidence that simply training longer repairs the representation pattern.

5. Saved best actors remain full1170 and residual1200. Their repeated benchmark values should reproduce prior results; no new selector peak exceeded the existing85%. Current selection remains40-game fixed-opponent score, not incumbent head-to-head promotion. Preserve them; the noisy selector and absence of best updates do not establish that later policies are worse.

6. Before another long unchanged run, prioritize an enlarged multi-seed evaluation of latest versus incumbent policies. A focused training experiment worth trying next is64 versus16 games/update at matched total games (four times fewer updates for64), testing whether less noisy on-policy estimates help. This is a hypothesis, not a demonstrated fix. If doing critic-specific fitting, more independent fresh games plus short fitting/early stopping is supported more than additional epochs on a small fixed dataset. Avoid simultaneous architecture/optimizer changes. No further training is launched here.

Artifacts: out/rl/paired-lambda1-2200-analysis.json; paired-lambda1-2200-evaluation.json; out/experiments/paired-ppo-updates1201-2200/comparison.png, verification.json, head-to-head/results.json. Full per-opponent value metrics, color outcomes and termination details are retained in evaluation JSON. Scripts work/analyze_pair_2200.py, evaluate_pair_2200.py, report_pair_2200.py. Logs logs/ppo-pair-1201-2200.out/.err, ppo-pair-2200-evaluation.out/.err and ppo-pair-2200-head-to-head.out/.err. Previous reports/evaluation artifacts remain unchanged.


### Supplement: direct incumbent checks and four-actor tournament

Following the residual benchmark warning, evaluated all four distinct latest/incumbent actors directly:6 matchups ×100 games=600 unique games,300 appearances/actor. Temperature0.9, max256 plies, five seeds743291–743295,20 games/seed, balanced colors. The latest-vs-latest result reproduces the earlier stage45 match on the same protocol, so it is not independent extra evidence and should not be counted twice. This tournament adds the missing direct incumbent comparisons. No checkpoint was promoted or reset.

| A | B | A W/L/D | A score | Approximate95% score interval |
| --- | --- | --- | --- | --- |
| Full latest2200 | Full best1170 | 58/37/5 | 60.5% | 51.1%–69.9% |
| Full latest2200 | Residual latest2200 | 54/43/3 | 55.5% | 45.9%–65.1% |
| Full latest2200 | Residual best1200 | 61/37/2 | 62.0% | 52.5%–71.5% |
| Full best1170 | Residual latest2200 | 61/39/0 | 61.0% | 51.4%–70.6% |
| Full best1170 | Residual best1200 | 49/49/2 | 50.0% | 40.2%–59.8% |
| Residual latest2200 | Residual best1200 | 36/61/3 | 37.5% | 28.1%–46.9% |

Normal-approximation per-game intervals are descriptive, unadjusted for multiple comparisons, and omit training-seed uncertainty. They favor full latest over its incumbent (~51.2–69.8% score) and residual incumbent over residual latest (latest~28.1–46.9%). Stronger evidence than the previous close1200/1170 match, but broader opponent robustness remains unmeasured.

Updated interpretation: the residual regression is supported by direct incumbent play, not just the single repeated-baseline sample. FullCNN latest shows evidence of improvement over its stored incumbent. This does not identify whether moving-opponent adaptation, forgetting, optimization noise or another mechanism caused the different trajectories. Do not conclude that lower critic RMSE guarantees actor improvement.

Updated recommendation: prioritize a reliable challenger-versus-incumbent promotion gate and a broader fixed evaluation panel before another long unchanged continuation. Preserve residual incumbent1200; latest2200 remains available for intentional continuation. A controlled small follow-up can test a frozen incumbent opponent or a simple mixture of frozen/current opponents at matched games, isolating opponent nonstationarity without requiring PSRO. The earlier16-vs64-games/update experiment remains an alternative variance-reduction hypothesis, not a demonstrated fix. No new training or selector change is performed here.

Additional artifacts: out/experiments/rl2200-latest-vs-saved-best-round-robin/results.json (all per-color and per-seed outcomes and actor hashes); out/rl/paired-lambda1-2200-comparison.png (fresh value trends and four-actor standings). Analysis now also contains last200 and ten100-update windows; old report window labels pre_change_last50/phase2 correspond to previous50/phase in the extended JSON schema.


<a id="stage-46"></a>

## 46. Literature review: robust fog-of-war play and whether to adopt PSRO (2026-11-05; proposal only)

User asked how to build a strong fog-of-war bot and whether to move to PSRO after the nonuniform2200 continuation. No new training, architecture, payoff solver or archive was implemented in this review.

### What the literature supports

- Lanctot et al., *A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning* (NeurIPS2017), https://arxiv.org/abs/1711.00832, full text https://arxiv.org/html/1711.00832. PSRO grows an empirical game of policies: estimate payoffs, compute a meta-strategy, use RL to train approximate responses against sampled fixed opponents, add responses and update the empirical game. PPO can remain the response-learning algorithm. The learned solution is a policy mixture; it need not be a single dominant network. A restricted-game solution is not evidence of low exploitability in the full game, especially with approximate response learning and noisy payoffs. Frozen-history sampling alone is useful but does not implement the proposed payoff/meta-solver/population-expansion loop.
- Vinyals et al., *Grandmaster level in StarCraft II using multi-agent reinforcement learning* (Nature2019), primary lab account https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/, paper https://doi.org/10.1038/s41586-019-1724-z. Historical opponents and exploiters address forgetting and strategic weaknesses that ordinary self-play can leave behind. This supports population training, but does not establish that our measured residual regression was caused by forgetting or a strategic cycle. We have not demonstrated a nontransitive cycle in our small pool.
- Perolat et al., *Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning* (Science2022), https://arxiv.org/abs/2206.15378, full text https://arxiv.org/html/2206.15378. DeepNash demonstrates strong imperfect-information play without search, using regularized Nash dynamics rather than PSRO. Its CNN inputs encode accumulated knowledge plus the last40 moves. The transferable lessons are history/information representation and stabilized game-theoretic learning; adopting a recurrent network or PSRO is not the only possible route. Its scale and game differ substantially from ours.
- Brown et al., *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games* (NeurIPS2020), https://arxiv.org/abs/2007.13544, paper https://proceedings.neurips.cc/paper/2020/file/c61f571dbd2fb949d3fe5ae1608dd48b-Paper.pdf. ReBeL uses public belief states and equilibrium-aware search rather than a single assumed full board. It is not a drop-in AlphaZero/MCTS replacement for our private fog observations. Its theoretical results have assumptions about two-player zero-sum games, representations and solving accuracy which do not carry over automatically to our PPO implementation.
- *Scalable decision-making for games of imperfect information* (Nature2026; Ataraxos), https://www.nature.com/articles/s41586-026-11036-y, open full text https://pmc.ncbi.nlm.nih.gov/articles/PMC13626944/. Demonstrates a different successful route in Stratego: controlled/regularized self-play updates, a separately trained hidden-information belief model, and test-time search over plausible hidden states. The paper reports a15W1L4D match against a decorated human champion. It supports treating beliefs/search as a serious longer-term direction, not evidence that our problem requires full PSRO. Performance and compute numbers are specific to that system; its custom simulator is not a recommendation to reimplement our game.

### Mapping to our measured setup

Our current CNN actor and value network receive only64 current observation tokens expanded to14 planes. ObservationTracker keeps limited history for candidate moves (including castling/en-passant information), but that information is not directly provided to the learned CNN features/value. The action mask may convey some history indirectly. The previous recurrent imitation model exists but recurrent PPO was explicitly excluded from the present infrastructure. Thus current PPO is restricted to a largely reactive representation: distinct observation histories with identical visible boards can yield the same unmasked policy/value predictions. More population diversity cannot recover information the network never receives.

The current residual decline is supported both by its repeated-baseline score78→58.5% and its36W61L3D loss to incumbent1200. That motivates robust opponent coverage, but does not identify forgetting, nontransitivity or moving-opponent instability as the cause. Weak critics, approximation/optimization limitations and sampling uncertainty remain relevant. We should not describe PSRO as a demonstrated repair or guarantee.

### Recommended staged design

1. Adopt a small frozen-policy population now rather than another long two-moving-agent continuation. Initial candidates can be both supervised baselines, full latest2200, full incumbent1170, residual latest2200, residual incumbent1200. Deduplicate identical actor states. Preserve older useful strategies, not only the latest and the highest noisy selector score. The canonical continuation policy per variant can remain one file; immutable population snapshots require their own persistent representation, for example embedded inference states with manifest/hashes. That storage design is not implemented by this review.
2. If implementing actual PSRO, retain the pairwise payoff table and compute a population meta-strategy, then train a fullCNN response and a residual response against frozen opponents sampled once per episode from a fixed distribution for that response round. Learners can train in parallel; neither should chase the other's changing weights inside that round. Warm-start from strong existing actors rather than discarding imitation/PPO progress. Add evaluated responses and recompute the meta-strategy. PPO remains the inner optimizer, lambda1 remains provisional, and actor architecture can initially remain unchanged. Solver weights must account for noisy payoffs; a numerical equilibrium of a noisy small matrix is not an exact game solution.
3. First compare that proposed meta-strategy sampling against a simple uniform frozen-population baseline at matched actual game/decision budgets, using held-out seeds and cross-play. This tests whether solver weighting adds value beyond merely retaining opponents. Current in-pool average score, worst matchup and response-training attacks are useful measures; empirical attacker success is not exact exploitability. Do not select solely by beating the previous incumbent, since a specialist may do that while losing broader coverage.
4. Independently test observation history as an architecture/input ablation: a short observation stack or explicit last-seen piece information plus age/visibility (stale information is not known current truth), or later recurrent PPO. Build from each player's own observation/action history only, keep train-time/inference contracts consistent, and never expose hidden true boards to the deployed actor. This is not evidence that history will necessarily fix the critic; test it on games and prediction metrics. Do not change history inputs and opponent scheduling in the same first experiment if attribution matters.
5. For substantially stronger fog-of-war play, consider a learned belief distribution over hidden states, followed by information-aware planning using the existing game engine where supported. Self-play true states may supervise belief targets during training, while belief-model inputs and deployed decisions use only allowed observations. Simply filling unseen squares with one guessed board and treating it as certain is not a principled solution. Search is a later engineering/research stage; no engine reimplementation is proposed.

### Payoff convention to settle before claiming Nash guarantees

Current training terminal utilities are win+1/loss−1/draw+0.1 for both players. Their sum is0 for decisive games and0.2 for draws, so this reward game is not zero-sum/constant-sum. Standard score (1/.5/0) is constant-sum; centered outcome (1/0/−1) is zero-sum. Gamma0.99 per own decision additionally gives a timing preference. PSRO can be formulated for general-sum games, but a zero-sum matrix solver on standard scores plus PPO optimizing the current discounted draw-bonus reward is an approximate response setup for a different objective; ordinary zero-sum convergence claims do not apply. Preserve the user's draw preference unless explicitly changed, and document the distinction rather than silently changing rewards. The game-theoretic target and response objective should be chosen together.

Recommendation: move toward a small PSRO/population loop as the next training-system experiment, while prioritizing history-aware information representation for stronger fog-of-war decisions. PSRO is a plausible response to restricted opponent coverage; it does not solve missing information, critic capacity or approximate best-response quality. No implementation or further run was authorized by this question or launched here.

<a id="stage-47"></a>

## 47. Premature local history pilot (2026-11-05; cancelled by user)

User endorsed history-enriched imitation pretraining, then PPO against existing full/residual CNNs, then evaluation before PSRO. Implemented and launched a local controlled pilot to establish the pipeline and assess the representation before a full-data training commitment. No PSRO code or large full-data run is launched here.

### Representation and contract

Feedforward CNN input is the current player observation plus the previous3 observations received by that player, sampled at every game ply. Newest frame first; four64-token views expand to56 one-hot channels. Game-start missing history is entirely unknown, not copied from future observations. Views include the player's observations on opponent turns, as supplied by the existing referee protocol. This is a short observation stack, not a full belief state or persistent last-seen memory. A hidden piece that disappears longer than the stack horizon is not retained explicitly.

Shared ObservationHistory logic is used by imitation examples, held-out probes, in-process agents and standalone bots. PPO records complete stacked tokens before action selection and uses exactly those tokens during optimization. No hidden board is used as model input; data targets remain recorded actions, and the canonical referee handles all game rules. Tests verify color separation, reset, ordering/padding, no future/hidden-board dependence, actual gameplay-to-replay alignment and standalone/in-process encoding compatibility.

Current-only legacy checkpoints remain compatible, including unchanged canonical variant identities. New history models have distinct history4-prefixed identities, one canonical RL file per new architecture, independently preserved latest/best state, normal optimizer/RNG resume, and supported standalone export. The original full/residual canonical files are never written by the pipeline. Frozen inference-only opponent snapshots are preserved under this experiment.

### Controlled imitation pretraining

Both current-only and history4 models use a medium residual CNN32/128,3 residual blocks, convolutional BN and dropout0.1. Initial predictions and all shared weights are exactly matched: the current-view stem uses ordinary initialization, additional history kernels are zero-initialized and trainable, and their creation preserves the RNG stream. This is not He reinitialization. New history input adds only first-convolution parameters; later layers are identical. One training seed7980, not a multi-seed architecture result.

Exact file split copied from the prior2560-game architecture pilot:2560 selected training files (2559 usable games),248396 action positions;64 validation games,6147 positions. Empty/unusable labeled-game handling is shared between arms. Eight epochs each, batch128 (1941 updates/epoch,15528 total), AdamW startLR1e-4, peak3e-4 after3 warmup epochs (5823 updates), cosine floor1e-5, betas0.9/0.99, epsilon1e-8, decay0.01, global gradient clip5. Selected imitation checkpoint maximizes legal top1 accuracy on a fixed2048-position validation probe. Loss/top1/top5, layer gradients, update ratios, LR, activation statistics/histograms and plots reuse the existing trainer. This is a matched new control and a new conservative LR recipe, not an exact replay of earlier pilot optimizer trajectories.

The small pilot has far less imitation data than existing full-data baseline models; losing to those models would not prove history is harmful. Compare history versus current-only control first. This is preliminary validation evidence, not an independent final test of imitation generalization.

### PPO and evaluation sequence

1. Complete both imitation arms; evaluate each against frozen fullCNN latest2200 and residual incumbent1200 on seeds743291/743292,50 games/opponent/seed (200 per candidate).
2. Initialize only the history model's actor from its selected imitation weights and zero its previously unsupervised value head using the existing PPO rule. Train400 PPO updates,16 games/update (6400 fresh games), lambda1, gamma0.99, unchanged PPO optimizer/clip/entropy recipe, against the two frozen opponents sampled by the existing pool. Existing full/residual actors do not update. In-loop evaluations every20 updates,20 games per opponent. Store the history RL model in its separate canonical variant directory.
3. Run a five-seed100-game/pair color-balanced round robin including current-only imitation, history imitation, history PPO latest/best (deduplicate if equal), full2200 and residual1200. Record before/after playing strength, value metrics, rewards/outcomes, stability and feature diagnostics. The control does not receive matched PPO; therefore a final history-PPO versus control-imitation advantage cannot be attributed solely to history inputs. Imitation comparison isolates the representation more directly; post-PPO comparison measures the whole pipeline.

### Verification, launch and artifacts

All67 unit tests passed in23.7s after implementation. Includes meaningful new history leakage/alignment/reset tests, matched-function/gradient tests, one tiny imitation/PPO integration path, exported game-protocol play and legacy canonical-ID checks. No training hyperparameter was changed based on an early pilot outcome.

Pipeline: work/run_history_cnn_experiment.py. Launched exec session76469. Master logs logs/history-cnn-pipeline.out/.err; phase logs logs/history-pilot-current-control-imitation.out/.err, logs/history-pilot-history4-imitation.out/.err, logs/history-pilot-ppo.out/.err. Experiment directory: out/experiments/history4-cnn-imitation-then-ppo-pilot/. pipeline.json stores recipe/progress/source/opponent hashes; split.json stores exact selected games; current-control/ and history4/ hold imitation diagnostics/checkpoints; imitation-arena.json holds pre-PPO evaluations; tournament/results.json holds final cross-play. Canonical history RL location is added to pipeline.json when its model identity is known.

Pipeline skips completed phases and resumes interrupted imitation epoch/PPO update boundaries; do not start a duplicate while it is active. The existing heartbeat monitor was repurposed to watch completion/recover concrete execution defects, append measured results and recommendations, and pause after final delivery. Results pending. No empirical claim that history helps has been made at launch.

### Cancellation and corrected scope

User clarified that feature design should be discussed before implementation/training, and imitation pretraining should run on the server. The local launch was premature. Cancelled the pipeline and active child trainer (both logs record KeyboardInterrupt; exec session76469 exited130) and paused the heartbeat monitor. Only the current-view control completed2 epochs, with epoch3 interrupted. The actual history model, PPO and planned evaluations never started. No results from this partial control establish anything about history inputs. Both original full/residual canonical checkpoint hashes verified unchanged. Partial artifacts and unreviewed prototype code remain for traceability, not an approved architecture or ready-to-run training plan. Do not resume this pilot or deploy its input design without the user choosing features and server recipe. pipeline.json and partial metrics.json are marked cancelled_by_user.

<a id="stage-48"></a>

## 48. History features: Obscuro and neural blind-chess literature (2026-11-05; design discussion only)

User clarified that history should include our own past moves, and favored current observation + last-seen observation + age. This is a design discussion; the cancelled prototype remains unapproved, no model input was changed for this review, and no training restarted.

Primary sources checked:

1. Zhang and Sandholm, *General search techniques without common knowledge for imperfect-information games, and application to superhuman Fog of War chess* (Obscuro), https://arxiv.org/abs/2506.01242; full text https://arxiv.org/html/2506.01242, section3 and appendixC.1. Obscuro retains the full set of positions compatible with its accumulated observations (including castling/en-passant rights), samples positions for imperfect-information subgame search, and uses Stockfish14 leaf evaluations. It is not a directly comparable history-plane CNN/PPO architecture. The paper does not establish the specific current+last-seen+age representation proposed here. Its success supports retaining historical information but not a claim that it tested our chosen neural features.
2. Bertram, Fürnkranz and Müller, *Supervised and Reinforcement Learning from Observations in Reconnaissance Blind Chess* (CoG2022), https://arxiv.org/abs/2208.02029; full text https://arxiv.org/html/2208.02029, sectionIII/tableI. Uses20 observations,90 spatial planes each (1800 total), including own last action encoded with73 move planes, own-piece positions, sensing information, known capture squares, illegal-move result and player color. Imitation pretraining precedes PPO with past self opponents. The authors report recurrent alternatives taking longer without better prediction accuracy in their setup, not a universal argument against recurrence. This is close methodological precedent for our imitation→PPO idea, but RBC has different sensing/action feedback from our FoW rules, so sensing/illegal-action channels do not transfer mechanically. Their modest playing strength also does not establish that observation stacking alone produces superhuman play.
3. Bertram, Fürnkranz and Müller, *Neural Network-based Information Set Weighting for Playing Reconnaissance Blind Chess* (IEEE Transactions on Games2024), https://arxiv.org/abs/2407.05864; full text https://arxiv.org/html/2407.05864, sectionV-B/tableI. Again uses20 recent turns with own requested-move encoding and observation feedback, here to weight possible board states with neural networks rather than directly choose moves. Direct evidence that action-observation history is useful input design precedent; not an ablation proving an optimal history length or our last-seen/age channels.

Proposed input groups, pending agreement:

- Current observed board, including empty versus unknown distinctions.
- Per-square last-observed content (piece type/color or observed empty), never a guessed current board. Refresh whenever that square is visible. Never-observed squares remain separately identifiable as unknown.
- Per-square age since observation, counted in plies, with a chosen bounded/log scaling and consistent game-start/never-seen handling. Age0 marks current observations; an explicit visibility plane is useful though partly redundant with current unknown encoding.
- Explicit recent own completed moves, initially considering8 moves, newest first. Compact CNN encoding can use a from-square plane and a to-square plane for each move (16 planes for8 moves), with promotion/special-action metadata where the task requires it. Unlike the73-plane directional encoding in RBC, this factors origin and destination and is a proposed lightweight encoding, not a reproduction. Only earlier own moves are inputs; the move being predicted is the target and must never appear among its own input history.
- Own color and known castling rights, which are already available to the protocol/tracker. Other known game metadata can be considered separately; do not infer unavailable opponent moves from hidden replay boards.

Own move history is complementary: board/last-seen/age compress prior observations and can discard the path of actions that led there. Explicit actions may help recognize repeated maneuvers, plans, recent pawn moves and the consequences of revealing/moving pieces. In some situations actions are reconstructible from sufficiently rich observation histories, so improvement is a hypothesis, not guaranteed new information in every position. Last-seen+age is also not a full belief distribution: stale sightings on multiple squares need not be simultaneously true, and it does not encode correlations between hidden pieces.

Recommended provisional representation: current + last-seen + age/visibility + recent own actions + small own-side metadata, rather than making a short stack of raw board views the whole history mechanism. Eight own moves is an initial design suggestion, not a literature-derived optimum. Need decide exact channels/history horizon and then prepare full-dataset server pretraining, followed by PPO/evaluation. No local training or server job was launched by this review.

<a id="stage-49"></a>

## 49. Boosted CNN: server pretraining infrastructure (2026-11-05; implemented, not launched)

Supervised model experiment; the complete record is maintained in [EXPERIMENT_HISTORY.md, stage 49](EXPERIMENT_HISTORY.md#stage-49).


<a id="stage-50"></a>

## 50. Boosted CNN full historical pretraining: 50 epochs and readiness for PPO (2026-11-05; completed)

Supervised model experiment; the complete record is maintained in [EXPERIMENT_HISTORY.md, stage 50](EXPERIMENT_HISTORY.md#stage-50).


<a id="stage-51"></a>

## 51. Boosted CNN PPO against a balanced frozen mixture (2026-11-05; launched)

User authorized400 PPOupdates for the completed boosted imitation model against a half-full-CNN/half-residual mixture, with results/report afterward. Pipeline `work/run_boosted_ppo_400.py`; logs `logs/boosted-ppo-400-balanced.out/.err`; manifest/results `out/experiments/boosted-cnn-ppo-400-balanced-frozen-opponents/`. Canonical future policy: `out/rl/boosted-cnn-c32-h128-f4004a979d/policy.pt`. Source boosted-cnn-28018186 best epoch50. Frozen opponents: full latest2200 and residual embedded-best1200, preserving both original canonical files. Only boosted updates.

The existing uniform opponent sampler would give50/50 in expectation. The run-specific driver instead uses the unchanged canonical training_steps/PPO/referee lifecycle and calls the existing collector for shuffled opponent/color-pairs, concatenating all Batch fields. This guarantees8games/opponent/update,4white/4black,6400games total/3200each. No game rules or observations are reimplemented. A mock collector check confirmed exact counts/colors and complete batch concatenation. No production RL implementation was changed.

Recipe:400updates,16games/update,lambda1,gamma.99,temperature.9,actorLR1e-5,criticLR1e-3,Adam eps1e-5,4passes,minibatch128,ratio clip.2,value coefficient.5,entropy coefficient.01,globalclip.5,targetKL.02,256ply cap,seed7980,CPU1thread. Zero-initialize the imitation-unsupervised value head; freeze dropout/BN buffers using eval mode while training weights. Checkpoints atomically update the single canonical boosted RL file, embedding selected best. In-loop checkpoint selection evaluates20games/opponent every20updates (40games,total),seed9137. Before and after/latest+best benchmarks use50games/opponent (100games/policy),seed572940,balanced colors,with the same frozen RL incumbents; this is distinct from older benchmarks against supervised models.

The pipeline records launch source/opponent/implementation hashes, skips already completed benchmarks, resumes update-boundary PPO after recoverable failures, and uses file locks to prevent duplicate writers. Completion analysis/report script `work/report_boosted_ppo_400.py` verifies finite tensors, identical source hashes,unchanged BN buffers,exact opponent/color counts and one canonical RL file; writes50-update-window metrics,plots via the canonical trainer,and measured stage52. Results pending; do not infer improvement from launch. No PSRO or unrequested model training is included.

<a id="stage-52"></a>

## 52. Completed boosted PPO400 against an exactly balanced frozen mixture (2026-11-05)

Completed400 PPO **updates** (each up to4 minibatch passes),6400 fresh training games,3200 against each frozen opponent. Each update contains8 games/opponent,4 per color. Total learner decisions=202228; actual optimizer minibatch steps=7046. Best selected update=320; latest=400. Only boosted updates; full/residual actors stay frozen. Canonical file: `out/rl/boosted-cnn-c32-h128-f4004a979d/policy.pt`. Training wall time=0.744hours; total pipeline including benchmarks=0.761hours.

### Recipe and opponent identity

Actor starts from boosted imitation epoch50; the untrained value head is zero initialized, with no historical value warmup. Opponents are full CNN latest2200 and residual saved-best1200 exported as weights-only immutable evaluation/training snapshots. These are stronger RL incumbents, not the old supervised baselines. Thus absolute scores are not interchangeable with stages36–45 benchmarks against supervised actors. Adam actor LR1e-5,value-head LR1e-3,epsilon1e-5;4PPO passes,minibatch128,ratio clip.2,value coefficient.5,entropy coefficient.01,gradient clip.5,target KL.02,temperature.9,gamma.99,lambda1,256-ply cap,seed7980,CPU1thread. Dropout off/BNbuffers fixed for both collection and optimization; encoder and policy weights trainable. Outcomes win+1/loss−1/draw+.1; score=(W+.5D)/N. This is one training seed, not a controlled feature ablation or PSRO.

### Independent before/after benchmark

100games/policy,50/opponent,balanced colors,seed572940,temperature.9,max256plies. Same protocol before/after, separate from the40-game checkpoint selector (seed9137,every20updates). The pre-PPO actor has its value head zeroed, preserving actor predictions; arbitrary imitation value outputs are not treated as a trained critic. Repeated benchmark is not an untouched seed once examined.

|Actor|Update|W/L/D|Score|Mean reward|
|---|---:|---:|---:|---:|
|Imitation starting actor|0|33/65/2|34.00%|-0.31800|
|Latest PPO|400|50/43/7|53.50%|0.07700|
|Selected PPO best|320|60/33/7|63.50%|0.27700|

|Actor|Frozen opponent|W/L/D|Score|MC value RMSE|MC value EV|Mean plies|
|---|---|---:|---:|---:|---:|---:|
|Imitation|full-latest2200|17/31/2|36.00%|0.81726|0.00000|63.56|
|Imitation|residual-best1200|16/34/0|32.00%|0.83529|0.00000|73.86|
|Latest400|full-latest2200|26/19/5|57.00%|0.77522|0.00835|55.24|
|Latest400|residual-best1200|24/24/2|50.00%|0.82737|0.02528|68.08|
|Selected best|full-latest2200|35/12/3|73.00%|0.78822|0.06508|49.08|
|Selected best|residual-best1200|25/21/4|54.00%|0.79266|0.01004|77.04|

### On-policy learning and critic diagnostics

MC RMSE is decision-weighted pooled squared error against full discounted outcomes; EV below is the mean of per-update EVs, not a pooled cross-update EV. Fresh means rollout predictions before fitting that batch. Same-batch after-update results are in analysis.json and must not be presented as held-out generalization. Lambda1 GAE targets equal discounted MC returns up to rounding.

|Window|W/L/D|Score|Reward|Critic loss|Fresh MC RMSE|Fresh mean-update EV|Mean endpoint KL|Mean entropy|Gradient clipped|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|All400|3368/2748/284|54.84%|0.10131|0.31939|0.84798|0.00895|0.00672|1.46154|100.00%|
|First10|59/89/12|40.62%|-0.18000|0.29568|0.80361|-0.00522|0.00602|1.74685|100.00%|
|First50|341/431/28|44.38%|-0.10900|0.32138|0.84082|0.00322|0.00604|1.62064|100.00%|
|Last50|467/279/54|61.75%|0.24175|0.30286|0.82971|0.00497|0.00709|1.16562|100.00%|

Training outcomes by opponent:

|Opponent|Games|W/L/D|Score|Mean reward|Mean plies|
|---|---:|---:|---:|---:|---:|---:|
|full-latest2200|3200|1791/1301/108|57.66%|0.15650|52.94|
|residual-best1200|3200|1577/1447/176|52.03%|0.04612|73.56|

Maximum endpoint KL=0.02859; KL early-stop updates=9; maximum pre-clipping global gradient=2535.15698. Mean PPO ratio-clipped fraction=3.10%; mean actor loss=-0.01583;mean total loss=0.12923. Termination reason totals: `{"King capturable": 6116, "Threefold repetition": 284}`. Games finish through the canonical referee; any protocol/bot failure aborts rather than becoming a training win.

### Representation diagnostics and integrity

Probe sampled from first training rollout and held fixed; this is not the imitation validation probe or held-out rollout set.

|Layer|Initial zero fraction|Final zero fraction|Initial largest activity share|Final largest activity share|
|---|---:|---:|---:|---:|
|encoder.1|55.82%|55.14%|9.32%|8.68%|
|encoder.2.2.relu2|95.45%|96.11%|78.11%|72.59%|
|encoder.6|53.60%|38.68%|96.24%|90.99%|

Finite latest/best tensors verified; original supervised source and both original full/residual canonical RL files match launch SHA256s. Frozen opponent snapshots also match. One canonical boosted RL model file with embedded best, no duplicate latest/best files. BN running buffers unchanged=True; actor/encoder state L2 change=0.512809. Initial source hash recorded in canonical training log matches launch source.

### Measured takeaways and next recommendations

Latest benchmark score changes by+19.50percentage points and selected-best benchmark by+29.50points versus the starting imitation actor. These are observed100-game changes, not proof of statistical significance or a history-specific PPO effect. No current-only model received a matched400-update PPO run, and no unchanged-training control was run. The best selector maximizes a different40-game sample, so selected-best need not win the100-game benchmark. No benchmark result automatically promotes or overwrites a checkpoint.

The latest actor shows a substantial descriptive gain against the same frozen incumbents; verify it across additional seeds before committing a longer continuation. Fresh critic EV remains low in the last50updates; lower same-batch fitting error does not establish a useful generalizing critic. Interpret gradient clipping alongside actual KL, score and finite tensors; do not infer collapse from clipping frequency or activity concentration alone. Keep opponent mixture fixed for any immediate diagnostic continuation, preserve the independently evaluated incumbent, and defer PSRO until useful complementary play is established. No additional training beyond400 is authorized by this report.

Artifacts: `out/experiments/boosted-cnn-ppo-400-balanced-frozen-opponents/{pipeline.json,analysis.json,before-ppo-benchmark.json,latest400-benchmark.json,selected-best-benchmark.json}`. Canonical metrics/plots in the boosted out/rl directory include rewards,outcomes,losses,value_quality,policy_health,throughput,training,evaluation_opponents; per-checkpoint activation/weight diagnostics in diagnostics/. Master logs: `logs/boosted-ppo-400-balanced.out/.err`. Analysis retains50-update windows, all selector evaluations, per-opponent results, source verification and features.

### Review of completed results and specific next step

The independent100-game benchmark improved from33W65L2D (34.0%) before PPO to50W43L7D (53.5%) at400 and60W33L7D (63.5%) for the saved-best actor at320. The saved-best scored73% versus full latest2200 (35W12L3D) and54% versus residual best1200 (25W21L4D). Latest400 scored57% and50%, respectively. This supports describing boosted as competitive with the tested established RL opponents after training. It does not establish universal superiority, nor isolate the contribution of history versus added parameters/pretraining.

Training score rose44.375% in the first50updates to61.750% in the last50; mean reward−.109→+.24175. Fresh value RMSE barely moved (.84082→.82971), and fresh mean-update EV remained near zero (.00322→.00497). Thus actor improvement is much clearer than critic improvement; current history/last-seen inputs did not by themselves solve value prediction in this run. Critic loss.32138→.30286 and better same-batch post-fit RMSE are weaker evidence than transfer to subsequent fresh rollouts.

Mean endpoint KL in the last50=.00709,mean entropy1.16562 versus1.62064 in the first50,ratio-clipped fraction about3.1%,with100%of optimizer steps globally clipped. No nonfinite checkpoint tensors were found. Those measurements support controlled probability changes and increasing policy concentration despite frequent clipping; they do not demonstrate a cured gradient/feature-scale problem. Dense largest-unit activity share fell96.24%→90.99% on the fixed rollout probe, while final-trunk zero fraction rose95.45%→96.11%; concentration remains substantial.

The40-game selector chose320 at75% score; its separate100-game score is63.5%. The10-point best320/latest400 benchmark gap warrants further checking, but100games/candidate with different induced trajectories is insufficient to diagnose definite late regression. Recommended next action: repeat balanced benchmarks across several seeds and directly compare320 versus400, retaining320 as the provisional incumbent rather than promoting latest automatically. If those comparisons support continued learning, continue the same canonical boosted run from latest with the incumbent preserved and more reliable checkpoint comparisons; do not blindly extend based only on update count. A critic-focused controlled experiment can follow if fresh EV remains flat, without claiming the improving actor must wait for a strong critic. No extra training or additional arena games were launched by this review.

Completed training took44.63minutes and the full pipeline45.66minutes. Exactly3200games/opponent,balanced colors,source/opponent hashes,finite tensors andunchanged BNbuffers were verified by the completion script. The monitor is paused after final delivery.

<a id="stage-53"></a>

## 53. Fresh-game value-head-only experiment on boosted (2026-11-05; launched)

User authorized400steps of the proposed actor-frozen critic experiment,with results/report afterward. This means400value-fitting rounds,16fresh games/round (6400training games),not400actor updates. Latest boosted PPO400 actor is the fixed subject; saved-best320 is preserved. Same frozen full latest2200/residual best1200 snapshots as stage52,exact8games/opponent/round,4/color. Source canonical `out/rl/boosted-cnn-c32-h128-f4004a979d/policy.pt`; one canonical RL file remains. No original source/opponent files are updated.

Freeze encoder,policy head and BN buffers; fit only existing linear+tanh value readout. Reuse current critic Adam moments/LR1e-3/eps1e-5,4passes over each fresh batch,minibatch128,head gradient clip.5,gamma.99,temperature.9,256ply cap. Actor's Adam state and root PPO RNG are preserved; fresh fitting uses an independent resumable phase RNG(seed684100). No policy loss,entropy bonus or actor update occurs. Bootstrap labels are complete-game discounted returns,not current critic estimates. Encoder features can be cached within each fresh batch because the encoder/eval buffers remain fixed.

Collect128validation games(seed684101) and256untouched test games(seed684102),game-level balanced opponents/colors; independent game-instance split using separate RNG streams; natural repeated trajectories are counted and retained. Evaluate validation every20rounds,select minimum MSE including unchanged head round0,compare selected/latest head to unchanged critic and training-mean constant on the untouched test features. Paired whole-game bootstrap stratified by opponent,3000resamples,seed684103. The original phase,latest fitted head and selected head are small embedded states,not extra RL model files. Commit the validation-selected value head/matching optimizer for later continuation; PPO actor update count stays400.

Driver `work/run_boosted_value_only_400.py`; report `work/report_boosted_value_only_400.py`; manifest/metrics under `out/experiments/boosted-value-head-only-400-fresh-games`; logs `logs/boosted-value-head-only-400.out/.err`; execsession46567. Driver locks the canonical file,checks actor/BN/saved-best hashes,checkpoints phase RNG/optimizer/metrics each round and resumes only this phase after recoverable errors. A small head-fitting smoke check verified16minibatch optimizer steps for4passes,head changes and bitwise frozen encoder/policy. Live results pending. Game scores should remain statistically unchanged because the actor is frozen; do not interpret sampling fluctuations as stronger play. No extra actor PPO,PSRO or independent encoder is launched.

Stage53 setup correction before any optimizer step: an initial assertion incorrectly treated naturally identical sampled trajectories as reused held-out game instances. Independent seeded draws produced3distinct shared short trajectories between validation/test (7-ply games against fullCNN). The check was corrected to retain and count natural repeats rather than censor common games and alter the target distribution. Holdout draws remain excluded from fitting/selection as specified; test is not used for model selection. No actor or critic parameter changed during the aborted setup.

<a id="stage-54"></a>

## 54. Completed boosted value-head-only fitting:400 fresh-game rounds (2026-11-06)

400 value-fitting rounds completed,6400 new training games,3200/opponent,4games/color/opponent/round. PPO actor update remains400; this is **not**400 additional actor/PPO updates. Only the linear+tanh value head was fitted,4passes/minibatch128 each round,Adam LR1e-3/eps1e-5,head-only clip.5. Latest400 actor/encoder/BN buffers stayed bitwise unchanged; saved-best320 actor/critic snapshot was untouched. The canonical PPO file retains the validation-selected head from round60 and matching optimizer state; original/latest/selected small value-head states and phase RNG/metrics are embedded for audit/resume. The400-round fitting phase does not advance PPO update numbering or change actor sampling weights. No historical data,PSRO or extra actor training was used.

Game-level holdouts:128 validation games/4268positions and256 test games/8537positions,balanced by opponent/color. Dedicated collection seeds684101/684102; fresh fitting seed684100. Different RNG streams produce independent game draws. Identical trajectories can naturally recur, especially short games; these are retained to avoid conditioning evaluation on novelty/outcome. No held-out game instance enters optimization. Natural trajectory overlap counts are recorded in analysis.json, not falsely described as novel states. Validation measured every20rounds selects minimum MSE,including original round0 as a candidate. Test predictions were measured initially and after fitting,never used for updates/selection. Constant predictor is mean discounted return over all fresh training positions (0.139423),not the test mean. Both original and selected heads use the exact same test features/returns,so comparisons do not suffer from changed actor trajectories. Feature caching is valid because the encoder is frozen in eval mode.

### Untouched test returns

|Critic|Test RMSE|Test MAE|Test EV|Prediction mean|Prediction std|
|---|---:|---:|---:|---:|---:|
|Unchanged PPO400 critic|0.755710|0.691162|0.029089|0.118177|0.135366|
|Latest value round400|0.774338|0.718183|-0.007911|-0.015496|0.124614|
|Validation-selected head|0.763842|0.684606|0.024077|0.182290|0.196871|
|Training-mean constant|0.768353|0.701469|0.000000|0.139423|0.000000|

|Reference|Selected minus reference test MSE|Paired95%interval|
|---|---:|---|
|unchanged|0.012357|[-0.017992,0.042067]|
|constant|-0.006912|[-0.031427,0.017239]|

Bootstrap resamples256 whole test games3000times,stratified by frozen opponent. Negative difference favors fitted head. These intervals condition on one training run and one held-out sample,exclude training-seed uncertainty,and are not multiplicity-adjusted.

|Frozen opponent|Unchanged test RMSE|Selected test RMSE|Constant test RMSE|
|---|---:|---:|---:|
|full-latest2200|0.738275|0.753681|0.748032|
|residual-best1200|0.767712|0.770894|0.782303|

### Fresh-stream fitting versus generalization

|Window|Fresh pre-fit RMSE|Fresh mean-round EV|Same-batch post-fit RMSE|Value loss|Head clip fraction|Mean pre-clip L2|
|---|---:|---:|---:|---:|---:|---:|
|All400|0.793002|0.021518|0.741070|0.290110|97.63%|15.555574|
|First50|0.777677|0.008006|0.728450|0.278517|97.03%|14.259687|
|Last50|0.806650|0.003753|0.765735|0.303704|97.17%|15.662228|

Total training positions=204073;actual optimizer minibatch steps=7164;maximum raw head gradient=144.298080;elapsed=35.34minutes. Fresh-stream summary RMSE pools squared errors by decision; EV is a mean of round EVs. Test EV is pooled on the fixed test set. Same-batch post-fit error is not held-out generalization. Fixed actor training scores/rewards fluctuate from sampling and are **not policy improvement**.

### Findings and next recommendation

The selected head does not establish a test-MSE improvement over the unchanged critic under the paired game-cluster interval. More fresh-game value-only fitting is therefore not yet demonstrated to solve the critic problem. Explained variance remains low even after head-only training,so a generalizing outcome predictor is still limited. A frozen shared representation constrains what a linear readout can learn; this experiment does not isolate representational limits from irreducible hidden-information/outcome noise or optimization/conditioning. No separate critic encoder was trained. Compare selected and unchanged critic initialization in a matched subsequent PPO experiment before attributing better actor learning to head fitting. If additional fitting is unsupported,avoid repeating400more rounds by default; a small independent-critic/nonlinear-readout diagnostic is more informative than enlarging the actor. No subsequent actor update or new experiment was launched by this report.

Artifacts: `out/experiments/boosted-value-head-only-400-fresh-games/{pipeline.json,metrics.json,analysis.json,value-only-training.png,validation-features.pt,test-features.pt}`. Canonical boosted `policy.pt` retains one model file, original actor optimizer state/root PPO RNG and embedded best; phase metadata holds the fitted-head lifecycle separately. Original supervised/full/residual checkpoints and frozen opponents match launch hashes. Logs: `logs/boosted-value-head-only-400.out/.err`. Reproduction/analysis scripts in `work/run_boosted_value_only_400.py` and `work/report_boosted_value_only_400.py`.

### Review: what the400rounds actually resolved

The unchanged critic test RMSE=.755710/EV=.029089;validation-selected round60=.763842/.024077;latest400=.774338/−.007911;constant=.768353/0. The selected critic is numerically worse than the unchanged head against **both** opponents,though the paired test MSE difference+.012357 has95%CI[−.017992,+.042067],so a definite degradation is not established. Its advantage over a training-mean constant is also unresolved (MSE−.006912,95%CI[−.031427,+.017239]). This experiment supplies no clear held-out benefit from extra head-only fitting. The512-game critic diagnostic on older models and this fresh-stream boosted run differ in data refresh/model; their shared lack of clear gains should not be presented as identical replications.

Validation-selected round60 improved validation RMSE only.755320→.751930. Later validation fluctuates substantially;at340 prediction mean=.786566 while fixed validation target mean=.133613,producingRMSE=.997079 withEV=.025821. The head is often adjusting the mean prediction rather than learning much more outcome discrimination. On test,initial mean bias=.041389 versus selected=.105502. Squared bias rises.001713→.011131,accounting for about76%of the selected head's.012357higher MSE;residual error variance also increases slightly. This is a calibration/conditioning warning,not proof that actor interference was the sole problem or that a separate encoder is required.

With continually refreshed data and frozen actor,first50→last50 fresh pre-fit RMSE worsened.777677→.806650 andmean-round EV fell.008006→.003753. Same-batch post-fit errors are lower,but transfer is weak. Roughly97%of head optimizer steps clipped despite removing the actor entirely;the high inherited feature scale and retained1e-3head LR are plausible optimization factors. The experiment did not isolate LR,feature scaling,readout capacity,partial observability or outcome noise. It therefore rules out neither a better-conditioned head nor a separately trained critic.

Recommendation:do not repeat another400head-only rounds with this unchanged recipe. Keep the original critic as the reference for future comparisons. The next small controlled diagnostic should address value-head conditioning (lower head LR or standardized frozen features),using the same train/validation/test protocol and guarding against test-driven selection;only then escalate to a nonlinear readout/independent critic if supported. A subsequent matched actor-PPO continuation would be needed to show that any fitted critic improves policy learning. No such additional run was launched.

Storage state is explicit:canonical boosted actor remainslatest400;embedded best320 remainsunchanged;active value head is the predeclared validation-selected round60,with matching optimizer state. The initial value head and latest fitted head remain embedded in the phase metadata for recovery/control. There is no claim that the selected critic won the untouched test comparison,and no actor-strength promotion occurred. Source/opponent hashes and finite tensors passed;monitor paused after final delivery.

<a id="stage-55"></a>

## 55. Boosted outcome CNN: matched joint-pretraining infrastructure (2026-11-06; implemented, not launched)

Supervised model experiment; the complete record is maintained in [EXPERIMENT_HISTORY.md, stage 55](EXPERIMENT_HISTORY.md#stage-55).


<a id="stage-56"></a>

## 56. Wide boosted outcome CNN setup (2026-11-06; implemented, not launched)

Supervised model experiment; the complete record is maintained in [EXPERIMENT_HISTORY.md, stage 56](EXPERIMENT_HISTORY.md#stage-56).


<a id="stage-57"></a>

## 57. Wide outcome baseline reduced to64/256 (2026-11-06; setup revision, not launched)

Supervised model experiment; the complete record is maintained in [EXPERIMENT_HISTORY.md, stage 57](EXPERIMENT_HISTORY.md#stage-57).

<a id="stage-59"></a>

## 59. Outcome-pretrained boosted: baseline100games then balanced PPO400 (2026-11-06; launched)

User authorized100baseline games againstfullCNN/residualCNN,then400PPOupdates againsta50/50mixture,andreevaluation/metrics/takeawaysintheRLreport. Source:completedmediumoutcome-pretrainingselectedepoch43,out/experiments/boosted-outcome-cnn-28094904/best_policy.pt (supervisedresultsinEXPERIMENT_HISTORY.md#stage-58). Itsoutcome-supervisedvalueheadispreservedatPPOinitialization,andthepipelinecheckstensorsagainstsourcebeforefirsttrainingbatch. Separatecanonicalvariant:out/rl/boosted-outcome-cnn-c32-h128-4abf38ebb1/policy.pt. Ordinaryboosted'sPPO400/value-onlyphaseandfull/residualpoliciesareuntouched.

Reusestage52balancedcollectorandcanonicaltraining_steps/PPO/referee:exact8games/opponent/update,4percolor,6400traininggames/3200each. FrozenfullCNNlatest2200andresidualsavedbest1200areexportedwithlaunchhashes;neitherupdates. Initialbenchmark100gamestotal(50/opponent),seed572940,temp.9,max256plies,gamma.99;latest400andsavedbestgetthesame100gamebenchmarkaftertraining. Thismatchesearlierordinaryboostedprotocol,nottheolderbenchmarksagainstsupervisedopponents. In-loopbestselectionuses20games/opponent,every20updates,seed9137;40gameselectornoisecanranklatest/bestdifferentlyfromthe100gamebenchmark.

SamePPOrecipe:actorLR1e-5,valueheadLR1e-3,Adam eps1e-5,4passes,minibatch128,ratio clip.2,value coefficient.5,entropy coefficient.01,globalclip.5,targetKL.02,gamma.99,lambda1,16games/update,seed7980,CPU1thread. Eval-modekeepsdropoutoffandBNbuffersfixedwhileencoder/policy/valueweightsaretrainable. Historicaloutcometargetswereundiscountedandfromdifferentpolicies;freshvalueRMSE/EVmeasuretransfer,ratherthanassuminghistoricalEV.22willcarryover. Noextraheadwarmup,noPSRO,nochangeinactorarchitecture.

Driverwork/run_boosted_outcome_ppo_400.py,execsession18725,logslogs/boosted-outcome-ppo-400-balanced.out/.err. Manifest/benchmarks/analysisunderout/experiments/boosted-outcome-cnn-ppo-400-balanced-frozen-opponents. Itskipscompletedbenchmarks,resumescanonicalupdateboundaries,usesfilelocks,andverifiessource/opponenthashes. Completionreportwork/report_boosted_outcome_ppo_400.pywritesmeasuredstage60,all/first10/first50/last50metrics,features,integritychecks,andcomparisonwithretainedordinaryboostedPPO400results. Cross-runcomparisonscombinechangesinencoder,policyandvaluepretraining;theyarenotanisolatedcritic-initializationablationandhaveonlyonetrainingseed. Noresultsareclaimedatlaunch.

Stage59 baseline completed before the first PPO update:35wins,63losses,2draws in100games,score36.0%,mean reward−.278. AgainstfullCNNlatest2200:score32%;againstresidualbest1200:40%. Fresh discounted-return critic EV was−.04497/full and+.01469/residual,RMSE.82430/.83289. These are initial transfer measurements,not post-PPO results;historical validationEV.22 did not carry over directly to this opponent/policy distribution. Driver verification confirmed source value tensors were preserved exactly at update0. Training is active;interpretation waits for the completed400-update trajectory and post benchmarks.

<a id="stage-60"></a>

## 60. Completed outcome-pretrained boosted PPO400 against an exactly balanced frozen mixture (2026-11-06)

Completed400 PPO **updates** (each up to4 minibatch passes),6400 fresh training games,3200 against each frozen opponent. Each update contains8 games/opponent,4 per color. Total learner decisions=216166; actual optimizer minibatch steps=7496. Best selected update=320; latest=400. Only boosted updates; full/residual actors stay frozen. Canonical file: `out/rl/boosted-outcome-cnn-c32-h128-4abf38ebb1/policy.pt`. Training wall time=0.792hours; total pipeline including benchmarks=1.198hours.

### Recipe and opponent identity

Actor starts from jointly outcome-supervised boosted imitation epoch43. The pretrained value head is preserved, not zero initialized; there is no extra head-only warmup. Historical pretraining targets are undiscounted outcomes, while PPO targets discounted current-policy returns. Opponents are full CNN latest2200 and residual saved-best1200 exported as weights-only immutable evaluation/training snapshots. These are stronger RL incumbents, not the old supervised baselines. Thus absolute scores are not interchangeable with stages36–45 benchmarks against supervised actors. Adam actor LR1e-5,value-head LR1e-3,epsilon1e-5;4PPO passes,minibatch128,ratio clip.2,value coefficient.5,entropy coefficient.01,gradient clip.5,target KL.02,temperature.9,gamma.99,lambda1,256-ply cap,seed7980,CPU1thread. Dropout off/BNbuffers fixed for both collection and optimization; encoder and policy weights trainable. Outcomes win+1/loss−1/draw+.1; score=(W+.5D)/N. This is one training seed, not a controlled feature ablation or PSRO.

### Independent before/after benchmark

100games/policy,50/opponent,balanced colors,seed572940,temperature.9,max256plies. Same protocol before/after, separate from the40-game checkpoint selector (seed9137,every20updates). The pre-PPO actor retains its learned historical outcome head. Its fresh-game value scores directly test transfer to discounted current-policy outcomes. Repeated benchmark is not an untouched seed once examined.

|Actor|Update|W/L/D|Score|Mean reward|
|---|---:|---:|---:|---:|
|Imitation starting actor|0|35/63/2|36.00%|-0.27800|
|Latest PPO|400|58/39/3|59.50%|0.19300|
|Selected PPO best|320|68/29/3|69.50%|0.39300|

|Actor|Frozen opponent|W/L/D|Score|MC value RMSE|MC value EV|Mean plies|
|---|---|---:|---:|---:|---:|---:|
|Imitation|full-latest2200|15/33/2|32.00%|0.82430|-0.04497|62.36|
|Imitation|residual-best1200|20/30/0|40.00%|0.83289|0.01469|68.66|
|Latest400|full-latest2200|28/21/1|57.00%|0.80498|0.12094|47.42|
|Latest400|residual-best1200|30/18/2|62.00%|0.85519|0.17584|77.16|
|Selected best|full-latest2200|34/14/2|70.00%|0.70840|0.10392|58.24|
|Selected best|residual-best1200|34/15/1|69.00%|0.74211|0.05956|77.68|

### On-policy learning and critic diagnostics

MC RMSE is decision-weighted pooled squared error against full discounted outcomes; EV below is the mean of per-update EVs, not a pooled cross-update EV. Fresh means rollout predictions before fitting that batch. Same-batch after-update results are in analysis.json and must not be presented as held-out generalization. Lambda1 GAE targets equal discounted MC returns up to rounding.

|Window|W/L/D|Score|Reward|Critic loss|Fresh MC RMSE|Fresh mean-update EV|Mean endpoint KL|Mean entropy|Gradient clipped|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|All400|3338/2692/370|55.05%|0.10672|0.28928|0.78484|0.05756|0.00555|1.47844|100.00%|
|First10|56/101/3|35.94%|-0.27938|0.32809|0.83321|-0.01086|0.00507|1.79490|100.00%|
|First50|340/429/31|44.44%|-0.10738|0.30484|0.81116|0.03102|0.00529|1.68078|100.00%|
|Last50|469/295/36|60.88%|0.22200|0.28586|0.78268|0.06802|0.00552|1.37235|100.00%|

Training outcomes by opponent:

|Opponent|Games|W/L/D|Score|Mean reward|Mean plies|
|---|---:|---:|---:|---:|---:|---:|
|full-latest2200|3200|1673/1334/193|55.30%|0.11197|58.02|
|residual-best1200|3200|1665/1358/177|54.80%|0.10147|77.19|

Maximum endpoint KL=0.03401; KL early-stop updates=10; maximum pre-clipping global gradient=1734.71460. Mean PPO ratio-clipped fraction=2.86%; mean actor loss=-0.01130;mean total loss=0.11853. Termination reason totals: `{"King capturable": 6030, "Threefold repetition": 370}`. Games finish through the canonical referee; any protocol/bot failure aborts rather than becoming a training win.

### Representation diagnostics and integrity

Probe sampled from first training rollout and held fixed; this is not the imitation validation probe or held-out rollout set.

|Layer|Initial zero fraction|Final zero fraction|Initial largest activity share|Final largest activity share|
|---|---:|---:|---:|---:|
|encoder.1|49.78%|49.62%|12.94%|12.62%|
|encoder.2.2.relu2|96.50%|95.17%|54.63%|53.52%|
|encoder.6|52.81%|46.54%|48.48%|40.41%|

Finite latest/best tensors verified; original supervised source and both original full/residual canonical RL files match launch SHA256s. Frozen opponent snapshots also match. One canonical boosted RL model file with embedded best, no duplicate latest/best files. BN running buffers unchanged=True; actor/encoder state L2 change=0.583635. Initial source hash recorded in canonical training log matches launch source.

### Comparison with the prior move-only boosted PPO experiment

Both runs use the same frozen actor identities,100-game benchmark seed and400-update recipe. This compares joint-pretraining versus move-only starting models;it does not isolate value-head initialization from altered encoder/policy features. Outcomes within games are correlated and one training seed does not measure run-to-run variation.

|Model|Before score|Latest400 score|Selected-best score|Last50 fresh MC EV|Last50 fresh MC RMSE|
|---|---:|---:|---:|---:|---:|
|Prior move-only boosted|34.00%|53.50%|63.50%|0.00497|0.82971|
|Joint outcome-pretrained boosted|36.00%|59.50%|69.50%|0.06802|0.78268|

### Measured takeaways and next recommendations

Latest benchmark score changes by+23.50percentage points and selected-best benchmark by+33.50points versus the starting imitation actor. These are observed100-game changes, not proof of statistical significance or a history-specific PPO effect. The previous move-only boosted experiment used this same400-update recipe/opponent identities/seed but a different imitation-trained actor and initially zero critic. It is a single-seed training-objective comparison, not an isolated value-head-initialization control. No unchanged-training control was run. The best selector maximizes a different40-game sample, so selected-best need not win the100-game benchmark. No benchmark result automatically promotes or overwrites a checkpoint.

The latest actor shows a substantial descriptive gain against the same frozen incumbents; verify it across additional seeds before committing a longer continuation. Fresh critic EV remains low in the last50updates; lower same-batch fitting error does not establish a useful generalizing critic. Interpret gradient clipping alongside actual KL, score and finite tensors; do not infer collapse from clipping frequency or activity concentration alone. Keep opponent mixture fixed for any immediate diagnostic continuation, preserve the independently evaluated incumbent, and defer PSRO until useful complementary play is established. No additional training beyond400 is authorized by this report.

Artifacts: `out/experiments/boosted-outcome-cnn-ppo-400-balanced-frozen-opponents/{pipeline.json,analysis.json,before-ppo-benchmark.json,latest400-benchmark.json,selected-best-benchmark.json}`. Canonical metrics/plots in the boosted out/rl directory include rewards,outcomes,losses,value_quality,policy_health,throughput,training,evaluation_opponents; per-checkpoint activation/weight diagnostics in diagnostics/. Master logs: `logs/boosted-outcome-ppo-400-balanced.out/.err`. Analysis retains50-update windows, all selector evaluations, per-opponent results, source verification and features.

### Review: critic transfer, calibration and practical next steps

Joint outcome pretraining followed by PPO is a promising starting recipe here: benchmark latest/best scores are each six percentage points above the retained move-only boosted experiment, and last50 fresh-rollout mean EV is 0.06802 rather than 0.00497, with RMSE 0.78268 rather than 0.82971. These are descriptive comparisons between different actors and trajectories, not a matched critic dataset or an isolated initialization experiment. One training seed and100 benchmark games do not establish superiority.

Historical undiscounted outcome EV around0.22 did not transfer immediately to current-policy discounted returns: first10 fresh-rollout EV was -0.01086. The critic subsequently acquired some useful signal, but calibration remains a concrete problem. In the latest400 residual-opponent benchmark, mean prediction was -0.25637 while mean discounted return was +0.23101, despite positive EV0.17584; RMSE was0.85519. At best320 the same opponent benchmark had mean prediction+0.33404 versus return+0.29505 and RMSE0.74211. EV ignores additive prediction bias, so report mean prediction/return and RMSE alongside EV. The best checkpoint's stronger actor and better calibration here do not prove a general late-training regression from a100-game comparison.

Dense-layer activity concentration decreased from48.48% to40.41% for the largest feature, while the final residual trunk remains sparse and concentrated. Together with finite tensors, improving play and mean endpoint KL around0.0055, this argues against declaring a catastrophic collapse. Every measured training minibatch still required gradient clipping, and the largest pre-clipping norm was1734.71; clipping is containing large gradients rather than eliminating their cause. Do not increase the clipping threshold merely to reduce the reported clipping fraction.

Treat embedded best320 as the provisional candidate for deployment/comparison, retaining latest400 for continuation. The next evidence-gathering step should be additional game seeds and direct matches of best320 versus latest400 and the retained move-only best actor, with balanced colors and uncertainty assessed at game level. That distinguishes checkpoint-selection noise from reproducible playing strength before another long continuation. When the wider supervised model is actually available, benchmark it using the same opponent suite before deciding whether extra capacity merits PPO. PSRO can later address opponent coverage; it does not by itself fix critic calibration. These are recommendations only: this experiment stopped at400 and no extra matches or training were started.

<a id="stage-61"></a>

## 61. Authorized outcome-boosted fixed-opponent continuation: updates401–800 (2026-11-06; launched)

Continue canonical medium outcome boosted latest400, preserving optimizer and RNG state, to800 total updates:400 additional updates/6400 fresh games. Same immutable full latest2200/residual embedded-best1200 opponents, exact8games/opponent/update and4/color. Same recipe as stage60:gamma.99,lambda1,actorLR1e-5,criticLR1e-3,Adam eps1e-5,4passes,minibatch128,clipratio.2,gradclip.5,targetKL.02,temperature.9,max256plies,seed7980,CPU1thread. Resume-parser check confirmed all prior recipe settings unchanged. No boosted actors added as opponents, no PSRO, no wide training or extra value warmup.

Prior stage60 metrics/benchmarks remain untouched; update400 complete metrics and prior analysis are copied into out/experiments/boosted-outcome-cnn-ppo-updates401-800-balanced-frozen-opponents. Before-continuation benchmark reuses the completed latest400 result(59.5%); previous best320 result(69.5%) remains the incumbent comparison. At800 evaluate latest and embedded best using100games each,50/opponent,seed572940,then compare preceding50/first10/first50/last50/all400 update metrics and features. Preserve original source/opponent hashes and best state through canonical checkpointing. Report script will append measured stage62; interpretation and recommendations require post-completion review. This is an additional-training experiment, not an isolated algorithm change; no claim of improvement before results.

Driver work/run_boosted_outcome_ppo_800.py; report work/report_boosted_outcome_ppo_800.py. Logs logs/boosted-outcome-ppo-401-800-balanced.out/.err. Canonical learner out/rl/boosted-outcome-cnn-c32-h128-4abf38ebb1/policy.pt.
