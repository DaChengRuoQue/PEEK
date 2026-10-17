# Self-Contained ImageNet-64 FID Evaluation Specification

## Objective

Replace the external solver-tree launcher with a self-contained evaluator in the PEEK repository, then launch one paired ImageNet-64 evaluation using 50,000 generated images for each method. The run must compare the fixed-schedule PEEK sampler with its compute-matched ordinary baseline and preserve enough metadata to audit the result.

## Evaluation protocol

- Model: NVIDIA conditional EDM checkpoint for ImageNet at 64 by 64 resolution.
- Solver: first-order DPM-Solver++ with the log-SNR timestep schedule.
- Ordinary baseline: 9 denoiser evaluations.
- PEEK: 6 ordinary solver evaluations with the fixed gate schedule `1,1,1,0,0`, giving 9 total denoiser evaluations.
- Pairing: both methods use the same deterministic latent and class-label stream.
- Metric: NVIDIA StyleGAN Inception features and the official ImageNet-64 reference statistics.
- Evaluation size: 50,000 generated images per method.
- Repetitions: one paired comparison.
- Output: exact configuration, artifact paths, baseline FID, PEEK FID, and relative reduction in machine-readable JSON.

The fixed protocol is determined before the run. The evaluator must not perform schedule search, tune against the generated samples, or execute any additional solver rows.

## Repository boundary

All executable evaluation code must reside under the PEEK repository root. The launch command must run from that root and must not accept or resolve an external solver directory. The implementation will include the minimal required components:

- ImageNet-64 EDM checkpoint loading and distributed deterministic sample generation.
- First-order DPM-Solver++ baseline sampling.
- The PEEK correction needed by the fixed schedule.
- NVIDIA-compatible Inception feature extraction and FID calculation.
- A paired orchestration command and an in-repository automatic runner.

The model checkpoint, reference statistics, detector cache, generated samples, logs, and run summaries will live in gitignored directories inside the repository. The existing local checkpoint may be copied into that in-repository cache after its hash is recorded. Runtime imports and subprocesses must resolve only repository code, the active Python environment, PyTorch launch utilities, and explicitly identified downloaded model assets.

No source file, option, log field, import, documentation entry, or launch command added for this evaluator may refer to the previous implementation. Existing PEEK attribution text that incorrectly describes the package as derived from that implementation will also be removed.

## Source provenance

Model-loading and FID utilities derived from NVIDIA EDM or StyleGAN must retain their original copyright and license notices. Solver logic derived from the public diff-sampler implementation must be reduced to the ImageNet-64 DPM-Solver++ path and documented with its upstream source revision. PEEK-specific logic will use this repository's `peek` package or a small adapter whose behavior is covered by tests.

## Components and data flow

1. The launch file invokes the repository-local automatic runner.
2. The paired evaluator validates the fixed protocol and local asset locations.
3. Distributed sampling generates the ordinary baseline samples, then the PEEK samples, using the same indexed latent and label stream.
4. The metric component calculates one FID for each generated set against the same reference statistics.
5. The evaluator writes a backend summary followed by a concise reproduction summary containing the relative reduction.

Completed artifacts are reusable. An interrupted run may resume only when the existing artifact metadata exactly matches the requested protocol; otherwise it must fail rather than mix incompatible samples.

## Validation and tests

Implementation follows test-driven development. Tests must first fail against the current external-backend launcher, then cover:

- rejection of any external backend-root option or path;
- construction of exactly one fixed paired protocol at the requested evaluation size;
- equal indexed latent and class-label streams for both methods;
- correct model-call accounting for the baseline and PEEK schedule;
- numerical agreement of the extracted first-order sampler and PEEK adapter with the previously used implementation on a small deterministic tensor fixture;
- summary validation and reduction calculation;
- dry-run output containing only paths under the PEEK repository root; and
- absence of the previous implementation name from tracked repository content.

Before launch, run the complete lightweight test suite, Python compilation checks, a no-GPU dry run, and a repository-boundary audit. The demanding evaluation begins only after these checks pass.

## Launch behavior

Use GPUs 2 through 7, leave occupied GPUs 0 and 1 untouched, and retain the previously validated per-GPU batch cap of 128. The full command is stored in a `.txt` file and invoked through the repository-local `auto_runner.py`. Launch it in a persistent session, report the process and log path, and return without waiting for completion.

## Acceptance criteria

- The live process has the PEEK repository as its working directory.
- No evaluator source, runtime import, command argument, or subprocess points to the external solver tree.
- Exactly one baseline-versus-PEEK paired comparison is scheduled.
- Both methods use the fixed paper-derived protocol and 50,000 generated images.
- A persistent log is visible immediately after launch.
- Final artifacts remain under the PEEK repository and include an auditable JSON summary.

## Non-goals

- Schedule search or post-hoc tuning.
- Repeated evaluations or additional solver comparisons.
- DiT ImageNet-256 evaluation.
- Refactoring unrelated PEEK package functionality.
- Deleting the prior completed run or its logs.
