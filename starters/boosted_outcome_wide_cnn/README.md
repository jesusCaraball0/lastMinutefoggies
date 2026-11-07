# Wide boosted outcome CNN

Width comparison against boosted_outcome_cnn:64 convolution channels and256
shared dense features,keeping3 residual blocks,convBN,ReLU,dropout0.1 and the
same1819 boosted input planes. Total3,372,033parameters versus1,370,625for32/128.
This doubles both widths relative to the32/128 medium model;the residual blocks,
shared dense layer and two prediction heads retain their existing design.
This tests the combined effect of widening the stem/trunk and dense representation;
it does not isolate which bottleneck matters or reproduce the paper architecture.

Input1819x8x8 → Conv3x3(64)/BN/ReLU →3 residual blocks(64) → flatten4096
→ Linear256/ReLU/Dropout.1 → policy4096 logits + linear/tanh outcome1.
Same move CE +0.5*outcome MSE,own-side undiscounted win+1/loss−1/draw+.1,
label eligibility/masking,full-data split,seed7980,50epochs and AdamW schedule.
Train from scratch;best checkpoint remains selected by legal move accuracy.
See ../boosted_outcome_cnn/README.md for shared loss/feature details.

Launch `sbatch training/boosted_outcome_wide_cnn.sbatch` from repo root.
Existing .venv-cnn is activated by the job. Outputs
out/experiments/boosted-outcome-wide-cnn-JOBID,logs of the same prefix.
OneGPU,4CPUs,32GiB,6hours per allocation;50epochs may require multiple
allocations. Resume with --resume first and the original last_policy.pt.
No automatic chained submissions are installed.

Future canonical RL identity:boosted-outcome-cnn-c64-h256-68076ee2fe.
Medium outcome,ordinary boosted and all older RL files keep their identities.
No weights are supplied by this starter;standard bot export uses shared inference.
