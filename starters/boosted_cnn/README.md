# Boosted CNN

A medium residual CNN (32 channels,128 hidden features,3 residual blocks,BN,
dropout0.1) with Bertram-inspired20-own-turn observation/action records plus
last-seen content, age, current visibility and known own castling rights.

The learned policy is feedforward. Input history is maintained by the shared
runtime, using only this player's observations and executed own moves.
Current target moves and hidden referee boards are never inputs.

Train with `sbatch training/boosted_cnn.sbatch` from the repository root.
Results go to `out/experiments/boosted-cnn-JOBID/`; copy the selected
`best_policy.pt` here as `policy.pt` when you want to use this directory in
an arena. It uses the same game protocol and shared bot runtime as other CNNs.
PPO uses a distinct `boosted-cnn-...` canonical variant and preserves existing
plain/residual policies. See the root RL report for experimental findings.

No trained weights are included until server pretraining completes.
