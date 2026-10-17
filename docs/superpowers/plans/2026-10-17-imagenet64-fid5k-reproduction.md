# ImageNet-64 FID Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch one paired 5k ImageNet-64 EDM evaluation that compares fixed-schedule PEEK with its compute-matched baseline and reports the relative FID reduction.

**Architecture:** Add a narrow PEEK-side launcher that validates and invokes the exact solver-native backend used for the manuscript experiment. The backend runs only the DPM-Solver++ PEEK row with its fixed schedule; that row generates both the 9-NFE ordinary baseline and 9-NFE PEEK samples, computes their two FIDs, and writes one paired result. A small summarizer adds the relative reduction to the backend JSON.

**Tech Stack:** Python 3.10, argparse, subprocess, existing PEEK package, existing solver-native PyTorch backend, unittest, `auto_runner.py`.

**Spec:** `docs/superpowers/specs/2026-10-17-imagenet64-fid5k-reproduction.md`

## Global Constraints

- Use the cloned PEEK repository at revision `e0d93ad4df2a957e62d7cc6dd6036d777a54fe7e`.
- Run exactly one paired evaluation: one compute-matched baseline FID and one PEEK FID.
- Use ImageNet-64 conditional EDM, first-order DPM-Solver++, base NFE 6, fixed PEEK schedule `1,1,1,0,0`, and 9 total model calls for both methods.
- Use 5,000 generated images per method, all eight H200 GPUs, and batch cap 1,024.
- Do not run schedule search, extra solver rows, repeated evaluations, or DiT ImageNet-256.
- Launch via the parent workspace's `auto_runner.py` and do not monitor the demanding job.

## Review Focus

- The launcher must select exactly one backend row and a fixed schedule, so no proxy search can start.
- The backend must retain its default NFE-matched baseline rather than skipping it.
- Output paths must resolve inside the PEEK clone even though the backend executes from its own root.
- The summary must calculate the reduction from the paired baseline and PEEK FIDs without rounding the stored values.
- Dry-run validation must not download models or start GPU sampling.

---

### Task 1: Minimal paired-evaluation launcher

**Files:**
- Create: `examples/imagenet64_edm_fid5k.py`
- Create: `tests/test_imagenet64_edm_fid5k.py`

**Interfaces:**
- Consumes: `run_cifar_peek.py` from an explicit solver backend root.
- Produces: a validated backend command and `reproduction_summary.json` after successful completion.

- [ ] Write tests for the exact fixed protocol command and paired-result arithmetic.
- [ ] Run the focused tests and observe failure because the module does not exist.
- [ ] Implement argument validation, exact command construction, subprocess execution, and paired-result summarization.
- [ ] Run the focused tests and full lightweight suite.

### Task 2: Launch file and dry-run verification

**Files:**
- Create: `commands_imagenet64_peek_fid5k.txt`
- Modify: `README.md`

**Interfaces:**
- Consumes: the launcher from Task 1 and the local solver backend.
- Produces: one command for `auto_runner.py`, logs, backend artifacts, and the final paired summary.

- [ ] Create one command using GPUs 0 through 7, 5,000 images per method, and batch cap 1,024.
- [ ] Run unit tests, compile checks, and launcher `--dry-run` validation.
- [ ] Launch `auto_runner.py` detached and return without monitoring.

## Launch command

```bash
python /home/yunwei/MultiStroke/auto_runner.py --commands /home/yunwei/MultiStroke/PEEK/commands_imagenet64_peek_fid5k.txt --log_dir /home/yunwei/MultiStroke/PEEK/runner_logs_imagenet64_peek_fid5k --stop_on_error
```
