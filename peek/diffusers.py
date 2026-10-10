"""Optional Diffusers adapter for the framework-neutral PEEK loop.

Importing this module does not import PyTorch or Diffusers. The adapter loads
PyTorch only when instantiated, so the base package and toy example remain
dependency-free.
"""

from __future__ import annotations

import copy
import importlib
from typing import Any

from .core import Prediction


class DiffusersEpsilonBackend:
    """Adapt an epsilon-prediction UNet and deterministic scheduler to PEEK.

    The scheduler must already have its inference timesteps configured. PEEK's
    provisional step mutates scheduler internals, so snapshots include the
    scheduler's complete Python state.
    """

    def __init__(self, unet: Any, scheduler: Any, *, clip_clean: bool = True) -> None:
        try:
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError(
                "DiffusersEpsilonBackend requires the 'diffusers' optional dependencies; "
                "install with `pip install -e '.[diffusers]'`."
            ) from exc
        prediction_type = getattr(getattr(scheduler, "config", None), "prediction_type", "epsilon")
        if str(prediction_type) != "epsilon":
            raise ValueError(
                "DiffusersEpsilonBackend currently supports epsilon-prediction schedulers only"
            )
        if not hasattr(scheduler, "alphas_cumprod"):
            raise ValueError("scheduler must expose alphas_cumprod")
        self.unet = unet
        self.scheduler = scheduler
        self.clip_clean = bool(clip_clean)

    @staticmethod
    def _timestep_index(timestep: Any) -> int:
        return int(timestep.item()) if hasattr(timestep, "item") else int(timestep)

    def _alpha_product(self, state: Any, timestep: Any) -> Any:
        value = self.scheduler.alphas_cumprod[self._timestep_index(timestep)]
        if hasattr(value, "to"):
            value = value.to(device=state.device, dtype=state.dtype)
        else:
            value = self._torch.as_tensor(value, device=state.device, dtype=state.dtype)
        return value

    def predict(self, state: Any, timestep: Any) -> Prediction[Any]:
        with self._torch.inference_mode():
            model_input = self.scheduler.scale_model_input(state, timestep)
            output = self.unet(model_input, timestep)
            direction = output.sample if hasattr(output, "sample") else output[0]
            if tuple(direction.shape) != tuple(state.shape):
                raise ValueError(
                    "epsilon prediction shape does not match sampler state; learned-variance "
                    "outputs require a model-specific adapter"
                )
            alpha_product = self._alpha_product(state, timestep)
            noise_product = 1.0 - alpha_product
            clean = (state - noise_product.sqrt() * direction) / alpha_product.sqrt()
            if self.clip_clean:
                clean = clean.clamp(-1.0, 1.0)
            return Prediction(clean=clean, direction=direction)

    def step(self, state: Any, timestep: Any, direction: Any) -> Any:
        with self._torch.inference_mode():
            output = self.scheduler.step(
                model_output=direction,
                timestep=timestep,
                sample=state,
            )
            return output.prev_sample if hasattr(output, "prev_sample") else output[0]

    def direction_from_clean(self, state: Any, timestep: Any, clean: Any) -> Any:
        alpha_product = self._alpha_product(state, timestep)
        noise_product = 1.0 - alpha_product
        return (state - alpha_product.sqrt() * clean) / noise_product.sqrt()

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(vars(self.scheduler))

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.scheduler.__dict__.clear()
        self.scheduler.__dict__.update(copy.deepcopy(snapshot))
