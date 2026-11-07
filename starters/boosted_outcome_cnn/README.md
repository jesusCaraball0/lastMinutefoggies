# Boosted outcome CNN

A separately named joint-pretraining variant of boosted_cnn. Same medium residual
backbone:32 channels,128 hidden units,3 residual blocks,convBN,ReLU,dropout0.1;
1,370,625parameters. Same1819 observation planes:20 own-turn observation/action
records plus current observable metadata,last-seen contents,age,ever-seen,visibility.
No hidden board is an actor or critic input. Game outcome is a target,not a feature.

From-scratch training uses `move_cross_entropy + 0.5 * outcome_MSE` per batch.
The existing linear+tanh value head predicts the acting player's final outcome:
win+1,loss−1,draw+0.1. These historical targets are **undiscounted**; PPO later
adapts them to current-policy gamma0.99 returns. Both objectives backpropagate
through the encoder. Only recognized fow-v1 gameplay outcomes supervise value;
protocol failures/missing metadata are masked for value,while their existing
valid move examples remain,keeping the old move-training split/count unchanged.

The cluster script `training/boosted_outcome_cnn.sbatch` uses50epochs and the
original boosted run's exact split/recipe. Select best by legal move accuracy,
not by a newly chosen outcome metric. Metrics include separate move CE,joint
loss,outcome MSE/RMSE/MAE/EV,win/loss accuracy excluding draws,and validation
training-mean constant baseline. New `outcome_training.png` joins the existing
loss/LR/gradient/activation/weight diagnostics. Masked label counts are recorded.

Metadata `model_config.outcome_supervision=true` gives this variant its own
canonical PPO identity; old IDs remain unchanged. `value_pretrained=true` prevents
PPO from zeroing the trained historical head. Standard game protocol and bundle
export work unchanged. No trained weights are installed in this starter until
explicitly copied/exported. See training/CLUSTER.md for commands.
