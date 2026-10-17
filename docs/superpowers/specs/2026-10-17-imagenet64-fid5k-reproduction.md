# ImageNet-64 FID Reproduction Specification

## Objective

Launch one paired evaluation from the cloned PEEK repository to estimate the manuscript's largest reported FID reduction using a smaller 5k evaluation, then return immediately without monitoring. The pair comprises the compute-matched baseline and PEEK measurements required to calculate one relative reduction.

## Paper-derived protocol

- Model: NVIDIA conditional EDM checkpoint for ImageNet at 64 by 64 resolution.
- Solver: first-order DPM-Solver++ with the log-SNR timestep schedule.
- Ordinary baseline: 9 solver evaluations.
- PEEK: 6 ordinary solver evaluations plus the fixed binary gate schedule `1,1,1,0,0`, for 9 total denoiser evaluations.
- Samples: the same deterministic latent and class-label stream for both methods.
- Metric: StyleGAN Inception features and the official ImageNet-64 reference statistics.
- Evaluation size: 5,000 generated images per method, as requested.
- Repetitions: one paired run only.
- Output: baseline FID, PEEK FID, absolute reduction, relative reduction, full configuration, and artifact paths in machine-readable JSON.

The manuscript reports FID 19.71 for the 9-evaluation ordinary baseline and FID 9.99 for PEEK under its full final-evaluation protocol, corresponding to a 49.3 percent relative reduction. The 5k run is a lower-cost estimate and is therefore not expected to reproduce those absolute FID values exactly.

## Acceptance criterion

The 5k result supports the stated basic effect when PEEK has lower FID than the compute-matched baseline and its relative reduction is in the broad neighborhood of the manuscript claim. Record the exact values without post-hoc schedule changes. If the reduction is materially outside that neighborhood, report the result as non-reproduction and retain all configuration and logs for diagnosis.

## Operational constraints

- Work in `/home/yunwei/MultiStroke/PEEK`, cloned from `https://github.com/WendyBaiYunwei/PEEK` at revision `e0d93ad4df2a957e62d7cc6dd6036d777a54fe7e`.
- Do not modify unrelated files in the parent MultiStroke worktree.
- Do not launch demanding GPU computation interactively.
- Put the complete launch sequence in a `.txt` command file and provide one exact `auto_runner.py` command.
- Use all eight available H200 GPUs and the largest safe effective batch cap for this 5k run.
- Reuse the local pretrained checkpoint when its identity is verified; do not duplicate the 1.2 GB file.
- Run only lightweight unit tests and a backend dry run before launch.
- Launch through `auto_runner.py` and do not wait for completion.

## Non-goals

- Re-running schedule search.
- Reproducing the five-run full paper table.
- Evaluating the DiT ImageNet-256 result.
- Changing the PEEK method or tuning its fixed schedule against the 5k test set.
