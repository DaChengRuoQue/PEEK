# Self-Contained ImageNet-64 FID Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch one repository-local, paired 50,000-image ImageNet-64 evaluation of first-order DPM-Solver++ with and without fixed-schedule PEEK.

**Architecture:** Add a narrow ImageNet-64 evaluation package that owns deterministic distributed sampling, the required first-order solver, NVIDIA-compatible FID, and paired orchestration. Vendor the minimal licensed NVIDIA pickle-loading support, keep all runtime assets and artifacts under gitignored repository directories, and expose one repository-relative command through a local automatic runner.

**Tech Stack:** Python 3.10, PyTorch distributed launch, NumPy, SciPy, Pillow, NVIDIA EDM/StyleGAN utilities, unittest.

**Spec:** `docs/superpowers/specs/2026-10-17-self-contained-imagenet64-fid50k.md`

## Global Constraints

- Run exactly one paired comparison with 50,000 generated images per method.
- Use conditional EDM ImageNet-64, a log-SNR schedule, first-order DPM-Solver++, and identical indexed latent and class-label streams.
- Use 9 ordinary denoiser evaluations for the baseline and 6 ordinary evaluations plus fixed PEEK gates `1,1,1,0,0` for 9 total evaluations.
- Do not search schedules, tune the fixed schedule, run additional solver rows, or repeat the evaluation.
- Runtime code, model assets, metric assets, generated images, logs, and summaries must resolve under the PEEK repository root.
- Preserve upstream license and provenance notices in vendored NVIDIA and solver-derived code.
- Use GPUs 2 through 7, leave GPUs 0 and 1 untouched, and use total batch 768 to retain the validated per-GPU cap of 128.
- Put the demanding command in a `.txt` file and launch it with the repository-local `auto_runner.py` in a persistent session.
- Preserve the prior completed run and its untracked logs.

## Review Focus

- Partial output directories must not be resumed unless their manifest matches method, image count, schedule, model digest, and indexed stream; Task 3 tests rejection.
- Distributed sharding must produce every index exactly once when the image count is not divisible by world size or batch size; Task 3 tests coverage and uniqueness.
- A stale FID JSON must not be reused when its image manifest or reference digest differs; Task 4 tests rejection.
- Asset preparation must reject a checkpoint or downloaded metric asset whose digest differs from the recorded digest; Task 2 tests rejection.
- Dry-run and live subprocess plans must reject resolved executable or asset paths outside the repository root; Task 5 tests repository containment.

---

### Task 1: Fixed protocol and first-order sampler

**Files:**
- Create: `peek/imagenet64/__init__.py`
- Create: `peek/imagenet64/protocol.py`
- Create: `peek/imagenet64/sampler.py`
- Create: `tests/test_imagenet64_protocol.py`
- Create: `tests/test_imagenet64_sampler.py`

**Interfaces:**
- Consumes: a callable EDM denoiser, indexed latents and one-hot labels.
- Produces: `EvaluationProtocol`, `build_logsnr_schedule(...)`, `sample_dpmpp_first_order(...)`, and `model_call_count(...)`.

- [ ] **Step 1: Write failing protocol tests**

Add tests asserting the immutable defaults: 50,000 images, baseline NFE 9, PEEK base NFE 6, gates `(1, 1, 1, 0, 0)`, total PEEK NFE 9, and rejection of schedule or count overrides.

- [ ] **Step 2: Run protocol tests and verify RED**

Run: `python -m unittest tests.test_imagenet64_protocol -v`

Expected: FAIL because `peek.imagenet64.protocol` does not exist.

- [ ] **Step 3: Implement the immutable protocol**

Define `EvaluationProtocol` as a frozen dataclass and `paper_protocol() -> EvaluationProtocol`. Validate the fixed values during construction and expose `model_call_count(base_nfe: int, gates: tuple[int, ...]) -> int`.

- [ ] **Step 4: Run protocol tests and verify GREEN**

Run: `python -m unittest tests.test_imagenet64_protocol -v`

Expected: PASS.

- [ ] **Step 5: Write failing sampler parity tests**

Add deterministic CPU tensor tests for the six-step PEEK and nine-step baseline paths. Assert exact model-call counts and numerical agreement with an independent test-local transcription of the previously validated first-order DPM-Solver++ update and PEEK correction.

- [ ] **Step 6: Run sampler tests and verify RED**

Run: `python -m unittest tests.test_imagenet64_sampler -v`

Expected: FAIL because sampler functions do not exist.

- [ ] **Step 7: Implement the sampler**

Implement `build_logsnr_schedule(num_evaluations: int, sigma_min: float, sigma_max: float, device: torch.device) -> torch.Tensor` and `sample_dpmpp_first_order(net, latents, class_labels, *, num_evaluations: int, gates: tuple[int, ...]) -> torch.Tensor`. Keep only the fixed first-order data-prediction path and use the repository PEEK interpolation primitive.

- [ ] **Step 8: Run focused and full tests**

Run: `python -m unittest tests.test_imagenet64_protocol tests.test_imagenet64_sampler -v && python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add peek/imagenet64 tests/test_imagenet64_protocol.py tests/test_imagenet64_sampler.py
git commit -m "feat: add fixed ImageNet sampler protocol"
```

### Task 2: Repository-local model and metric assets

**Files:**
- Create: `dnnlib/__init__.py`
- Create: `dnnlib/util.py`
- Create: `torch_utils/__init__.py`
- Create: `torch_utils/misc.py`
- Create: `torch_utils/persistence.py`
- Create: `peek/imagenet64/assets.py`
- Create: `scripts/prepare_imagenet64_assets.py`
- Create: `tests/test_imagenet64_assets.py`
- Modify: `.gitignore`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the known NVIDIA checkpoint, detector, and reference-statistics URLs or a user-supplied local checkpoint during preparation only.
- Produces: `AssetManifest`, `prepare_assets(...)`, and `load_asset_manifest(...)`; runtime paths under `checkpoints/imagenet64/`.

- [ ] **Step 1: Write failing asset tests**

Test repository-root containment, SHA-256 calculation, atomic copy/download, rejection of mismatched digests, and rejection of incomplete manifests.

- [ ] **Step 2: Run asset tests and verify RED**

Run: `python -m unittest tests.test_imagenet64_assets -v`

Expected: FAIL because the asset module does not exist.

- [ ] **Step 3: Vendor minimal NVIDIA support with notices intact**

Add the EDM/StyleGAN `dnnlib` and `torch_utils` files required to unpickle the network and detector. Record upstream project URLs and license terms in file headers and `README.md`.

- [ ] **Step 4: Implement asset preparation**

Define `prepare_assets(repo_root: Path, checkpoint_source: str | None) -> AssetManifest`. Store the model, detector, reference statistics, and JSON manifest below `checkpoints/imagenet64/`; verify every digest before atomic rename. The live evaluator consumes only the manifest and never accepts an external source path.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m unittest tests.test_imagenet64_assets -v && python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add .gitignore pyproject.toml dnnlib torch_utils peek/imagenet64/assets.py scripts/prepare_imagenet64_assets.py tests/test_imagenet64_assets.py README.md
git commit -m "feat: prepare repository-local EDM assets"
```

### Task 3: Deterministic distributed image generation

**Files:**
- Create: `peek/imagenet64/random.py`
- Create: `peek/imagenet64/generate.py`
- Create: `tests/test_imagenet64_generate.py`

**Interfaces:**
- Consumes: `EvaluationProtocol`, `AssetManifest`, and `sample_dpmpp_first_order(...)`.
- Produces: `StackedRandomGenerator`, `shard_indices(...)`, `GenerationManifest`, and CLI `python -m peek.imagenet64.generate`.

- [ ] **Step 1: Write failing generation tests**

Test deterministic per-index latent and one-hot label generation, exact non-overlapping sharding for uneven counts, method-specific sampler selection, atomic PNG writes, manifest matching, and rejection of incompatible partial output.

- [ ] **Step 2: Run generation tests and verify RED**

Run: `python -m unittest tests.test_imagenet64_generate -v`

Expected: FAIL because generation modules do not exist.

- [ ] **Step 3: Implement indexed generation**

Define `StackedRandomGenerator`, `shard_indices(num_images: int, rank: int, world_size: int) -> list[int]`, and `generate_rank(config: GenerationConfig) -> GenerationManifest`. Initialize distributed execution from `torchrun`, load the local checkpoint, generate deterministic indexed samples, and save them under `runs/imagenet64_edm_fid50k/{baseline,peek}/images/`.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_imagenet64_generate -v && python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add peek/imagenet64/random.py peek/imagenet64/generate.py tests/test_imagenet64_generate.py
git commit -m "feat: generate paired ImageNet samples"
```

### Task 4: Distributed NVIDIA-compatible FID

**Files:**
- Create: `peek/imagenet64/fid.py`
- Create: `tests/test_imagenet64_fid.py`

**Interfaces:**
- Consumes: repository-local detector and reference statistics plus a validated `GenerationManifest`.
- Produces: `FeatureMoments`, `calculate_fid(...)`, validated `fid.json`, and CLI `python -m peek.imagenet64.fid`.

- [ ] **Step 1: Write failing FID tests**

Test moment accumulation and merging, FID against a small known NumPy/SciPy fixture, exact image-count enforcement, and rejection of stale FID metadata when image or reference digests differ.

- [ ] **Step 2: Run FID tests and verify RED**

Run: `python -m unittest tests.test_imagenet64_fid -v`

Expected: FAIL because the FID module does not exist.

- [ ] **Step 3: Implement distributed feature statistics and FID**

Define `calculate_feature_moments(...) -> FeatureMoments` and `calculate_fid(moments: FeatureMoments, reference: FeatureMoments) -> float`. Load the repository-local NVIDIA detector, reduce float64 sums across ranks, and write provenance-checked JSON only on rank zero.

- [ ] **Step 4: Run focused and full tests**

Run: `python -m unittest tests.test_imagenet64_fid -v && python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add peek/imagenet64/fid.py tests/test_imagenet64_fid.py
git commit -m "feat: compute repository-local ImageNet FID"
```

### Task 5: Paired orchestration and repository-boundary cleanup

**Files:**
- Replace: `examples/imagenet64_edm_fid5k.py` with `examples/imagenet64_edm_fid50k.py`
- Replace: `tests/test_imagenet64_edm_fid5k.py` with `tests/test_imagenet64_edm_fid50k.py`
- Create: `peek/imagenet64/evaluate.py`
- Create: `auto_runner.py`
- Replace: `commands_imagenet64_peek_fid5k.txt` with `commands_imagenet64_peek_fid50k.txt`
- Modify: `peek/core.py`
- Modify: `peek/search.py`
- Modify: `README.md`
- Delete: `docs/superpowers/plans/2026-10-17-imagenet64-fid5k-reproduction.md`
- Delete: `docs/superpowers/specs/2026-10-17-imagenet64-fid5k-reproduction.md`

**Interfaces:**
- Consumes: repository-local generation and FID CLIs.
- Produces: `build_evaluation_commands(...)`, `validate_repo_command(...)`, `reproduction_summary.json`, a one-line command file, and repository-local launch logs.

- [ ] **Step 1: Write failing orchestration tests**

Assert exactly four ordered subprocesses: baseline generation, baseline FID, PEEK generation, and PEEK FID. Assert the fixed values, six-process launch, total batch 768, rejection of external resolved paths, summary arithmetic, and absence of schedule-search options.

- [ ] **Step 2: Run orchestration tests and verify RED**

Run: `python -m unittest tests.test_imagenet64_edm_fid50k -v`

Expected: FAIL because the new entry point does not exist and the old one accepts an external solver root.

- [ ] **Step 3: Implement orchestration and local runner**

Define `build_evaluation_commands(repo_root: Path, protocol: EvaluationProtocol, nproc: int, batch: int) -> tuple[list[str], ...]` and `validate_repo_command(repo_root: Path, command: list[str]) -> None`. The example delegates to `peek.imagenet64.evaluate`; the local runner executes the single command file and writes its log immediately.

- [ ] **Step 4: Remove obsolete external-backend material**

Delete the old launcher, tests, command, plan, and specification; remove incorrect attribution from `peek/core.py` and `peek/search.py`; rewrite README instructions with repository-relative paths and the exact local automatic-runner command.

- [ ] **Step 5: Run focused and full tests**

Run: `python -m unittest tests.test_imagenet64_edm_fid50k -v && python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 6: Run static and boundary verification**

Run compilation and the evaluator dry run, then build the three retired-name fragments dynamically and confirm `git grep` finds none of them in tracked content.

Expected: exit 0; dry run prints exactly one paired evaluation with four repository-contained subprocesses; forbidden tracked references are absent.

- [ ] **Step 7: Commit**

```bash
git add -A README.md peek examples tests scripts auto_runner.py commands_imagenet64_peek_fid50k.txt docs/superpowers
git commit -m "feat: make ImageNet evaluation self-contained"
```

### Task 6: Prepare, smoke-check, and launch

**Files:**
- Create ignored assets under: `checkpoints/imagenet64/`
- Create ignored outputs under: `runs/imagenet64_edm_fid50k/`
- Create ignored logs under: `runner_logs_imagenet64_peek_fid50k/`

**Interfaces:**
- Consumes: the complete repository-local evaluator and existing verified local NVIDIA checkpoint as a preparation source.
- Produces: a live automatic-runner process and an immediately visible log path.

- [ ] **Step 1: Prepare repository-local assets**

Run `scripts/prepare_imagenet64_assets.py` from the repository root, copying the existing checkpoint into the ignored asset directory and downloading or copying the detector and reference statistics. Record and inspect the manifest digests.

- [ ] **Step 2: Run the allocation smoke check**

Run one distributed batch for each method on GPUs 2 through 7 with the production per-rank batch cap while suppressing metric calculation and final artifact reuse. Keep total batch 768 if both pass; reduce only on an observed out-of-memory failure.

- [ ] **Step 3: Run final pre-launch verification**

Run the complete unit suite, compilation, dry run, repository-boundary audit, asset-manifest verification, GPU availability check, command-file line-count check, and `git diff --check`.

Expected: zero test failures, GPUs 2 through 7 available, GPUs 0 and 1 absent from the launch environment, one non-comment command, no tracked forbidden references, and every resolved runtime path inside the repository.

- [ ] **Step 4: Launch in a persistent session**

Run from the repository root:

```bash
python auto_runner.py --commands commands_imagenet64_peek_fid50k.txt --log_dir runner_logs_imagenet64_peek_fid50k --stop_on_error
```

Return after confirming the process remains live and the log contains the resolved fixed protocol. Do not wait for evaluation completion.
