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

For the optional Diffusers example:

```bash
python -m pip install -e ".[diffusers]"
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

`backend` implements model prediction, scheduler stepping, clean-prediction conversion, and scheduler snapshot and restore. See `examples/toy.py` for the complete interface and `peek/diffusers.py` for a model-backed adapter.

## Diffusers example

The following command downloads an epsilon-prediction CIFAR-10 checkpoint and writes one image. It uses a first-order deterministic DPM-Solver++ scheduler, matching the backend assumptions in the example.

```bash
python -m examples.diffusers_cifar10 \
  --model-id google/ddpm-cifar10-32 \
  --steps 9 \
  --schedule 1,1,1,1,1,1,1,1 \
  --output peek_cifar10.png
```

This command is intentionally not part of the lightweight test suite because it downloads a checkpoint and performs GPU inference.

## Schedule search and caches

`peek.greedy_schedule_search` implements the paper protocol. It starts from the all-zero schedule and accepts a nonzero gate only when that candidate strictly improves both the current PEEK schedule and an ordinary sampler with matched denoiser-evaluation cost. The evaluator is application-defined, so it can use FID or another lower-is-better validation metric.

The search memoizes matched-baseline metrics. It also supports a selected-prefix state supplied by the evaluator, allowing later gates to start from the cached noisy prefix rather than rerunning decided steps. `peek.rebase_first_order` provides the framework-neutral operation needed to move an ordinary first-order baseline cache to a new noise level.

## Data and checkpoint prerequisites

- The core package, unit tests, and toy example require no datasets or checkpoints.
- The Diffusers example requires network access on first use, the `google/ddpm-cifar10-32` checkpoint or an equivalent local epsilon-prediction checkpoint, and enough accelerator memory for that model. Use `--device cpu` only for a slow functional run.
- Full paper-table reproduction requires the corresponding pretrained model checkpoints, evaluation datasets or reference statistics, and compute for proxy schedule search and final evaluation. These assets are not distributed in this repository.
- No generated samples, FID caches, datasets, model weights, or experiment logs are included.

## Repository structure

```text
peek/
  core.py        PEEK and PEEK-G correction loop and backend protocol
  search.py      greedy compute-matched interpolation-schedule search
  cache.py       first-order baseline cache rebasing
  diffusers.py   optional epsilon-prediction Diffusers adapter
examples/
  toy.py                  dependency-free smoke example
  diffusers_cifar10.py    checkpoint-backed sampling example
tests/                    deterministic lightweight unit tests
```

## Validation

Run the lightweight suite with:

```bash
python -m unittest discover -v
python -m compileall -q peek examples tests
python -m examples.toy
```

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
