# PEEK

PEEK is a training-free correction for diffusion sampling that uses one extra denoiser query to look one step ahead. At selected sampler steps, it combines the current prediction with the next-step prediction before committing the current update. PEEK-G applies the same idea to the solver direction. A greedy calibration procedure selects where to spend the additional queries while comparing each candidate with an ordinary sampler at matched inference cost.

This repository is the minimal public implementation accompanying **“PEEK: One-Step-Look-Ahead Exposure Bias Correction for Diffusion Sampling”** by Yunwei Bai, Ying Kiat Tan, and Tsuhan Chen at the National University of Singapore.

Project page: [wendybaiyunwei.github.io/PEEK-Video](https://wendybaiyunwei.github.io/PEEK-Video)

## Installation

PEEK requires Python 3.10 or newer. The core package and toy example have no third-party runtime dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

For the ImageNet-64 EDM example:

```bash
python -m pip install -e ".[imagenet]"
```

## Minimal runnable usage

The dependency-free example runs the same correction loop used by model backends:

```bash
python -m examples.toy
```

Expected output:

```json
{
  "ordinary_final_state": 5.0,
  "peek_final_state": 9.0,
  "peek_used_at_steps": [
    0
  ]
}
```

The core API accepts a sampler backend and one interpolation value for every step except the final step:

```python
from peek import sample

result = sample(
    initial_state,
    timesteps,
    alpha_schedule,
    backend,
    method="peek",  # use "peek_g" for direction interpolation
    record_trace=False,
)
final_state = result.final_state
```

`backend` implements model prediction, scheduler stepping, clean-prediction conversion, and scheduler snapshot and restore. See `examples/toy.py` for the complete interface. The repository's model-backed sampler is implemented in `peek/imagenet64/sampler.py`.

## ImageNet-64 EDM example

`examples.imagenet64_edm_fid50k.py` launches one paired comparison for the manuscript's largest reported FID reduction. It uses NVIDIA's conditional ImageNet-64 EDM checkpoint, first-order DPM-Solver++, a 9-evaluation ordinary baseline, and a 6-step PEEK sampler with the fixed schedule `1,1,1,0,0`, also totaling 9 evaluations. It does not run schedule search or repeated evaluations. Sampling, metric calculation, and orchestration are implemented inside this repository.

Prepare the pinned NVIDIA model and metric assets under the ignored
`checkpoints/imagenet64/` directory. A local checkpoint can be supplied to
avoid downloading it again:

```bash
python scripts/prepare_imagenet64_assets.py --checkpoint-source /path/to/edm-imagenet-64x64-cond-adm.pkl
```

Launch the checked-in command from the repository root. It uses GPUs 2 through
7, leaving occupied GPUs 0 and 1 untouched, and writes a visible timestamped
log immediately:

```bash
python auto_runner.py \
  --commands commands_imagenet64_peek_fid50k.txt \
  --log_dir runner_logs_imagenet64_peek_fid50k \
  --stop_on_error
```

After completion, the paired FIDs and relative reduction are written to `runs/imagenet64_edm_fid50k/reproduction_summary.json`. The manuscript reports a 49.3 percent relative reduction for this protocol.

| Method | Manuscript FID | Reproduced FID |
| --- | ---: | ---: |
| Ordinary DPM-Solver++ | 19.71 | 19.71 |
| PEEK | 9.99 | 9.98 |

The reproduced relative reduction is 49.35 percent.

## Schedule search and caches

`peek.greedy_schedule_search` implements the paper protocol. It starts from the all-zero schedule and accepts a nonzero gate only when that candidate strictly improves both the current PEEK schedule and an ordinary sampler with matched denoiser-evaluation cost. The evaluator is application-defined, so it can use FID or another lower-is-better validation metric.

The search memoizes matched-baseline metrics. It also supports a selected-prefix state supplied by the evaluator, allowing later gates to start from the cached noisy prefix rather than rerunning decided steps. `peek.rebase_first_order` provides the framework-neutral operation needed to move an ordinary first-order baseline cache to a new noise level.

## Data and checkpoint prerequisites

- The core package, unit tests, and toy example require no datasets or checkpoints.
- The ImageNet-64 EDM example requires the pinned NVIDIA model, detector, and reference-statistics assets prepared by `scripts/prepare_imagenet64_assets.py`, plus CUDA-capable GPUs.
- Full paper-table reproduction requires the corresponding pretrained model checkpoints, evaluation datasets or reference statistics, and compute for proxy schedule search and final evaluation. These assets are not distributed in this repository.
- No generated samples, FID caches, datasets, model weights, or experiment logs are included.

## Repository structure

```text
peek/
  core.py        PEEK and PEEK-G correction loop and backend protocol
  search.py      greedy compute-matched interpolation-schedule search
  cache.py       first-order baseline cache rebasing
  imagenet64/    conditional EDM sampling and FID evaluation
examples/
  toy.py                       dependency-free smoke example
  imagenet64_edm_fid50k.py    checkpoint-backed paired evaluation
scripts/
  prepare_imagenet64_assets.py    pinned asset preparation
tests/                         deterministic lightweight unit tests
```

## Vendored NVIDIA utilities

The repository-local ImageNet evaluator includes the minimal `dnnlib` and
`torch_utils` modules needed to load NVIDIA EDM and StyleGAN metric pickles.
These files originate from [NVLabs EDM](https://github.com/NVlabs/edm) and
retain NVIDIA's Creative Commons Attribution-NonCommercial-ShareAlike 4.0
copyright and license notices.

## Citation

The source manuscript does not yet provide an archival identifier. Until one is available, use the following provisional citation:

```bibtex
@misc{bai2026peek,
  title = {PEEK: One-Step-Look-Ahead Exposure Bias Correction for Diffusion Sampling},
  author = {Yunwei Bai and Ying Kiat Tan and Tsuhan Chen},
  year = {2026},
  note = {Manuscript}
}
```

## License status

No open-source license has been declared for this code, so this repository intentionally does not include a license file. Unless and until the copyright holders add a license, no permission to use, modify, or redistribute the code is granted beyond rights provided by applicable law.
