"""Generate one CIFAR-10 image with a fixed PEEK schedule and Diffusers."""

from __future__ import annotations

import argparse
from pathlib import Path

from peek import sample
from peek.diffusers import DiffusersEpsilonBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/ddpm-cifar10-32")
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--schedule", default="1,1,1,1,1,1,1,1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("peek_cifar10.png"))
    return parser.parse_args()


def main() -> None:
    import torch
    from diffusers import DDPMPipeline, DPMSolverMultistepScheduler
    from PIL import Image

    args = parse_args()
    alphas = [float(value) for value in args.schedule.split(",") if value.strip()]
    if len(alphas) != args.steps - 1:
        raise ValueError(f"--schedule must contain {args.steps - 1} values")

    pipeline = DDPMPipeline.from_pretrained(args.model_id)
    unet = pipeline.unet.to(args.device).eval()
    scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        solver_order=1,
        algorithm_type="dpmsolver++",
    )
    scheduler.set_timesteps(args.steps, device=args.device)
    sample_size = unet.config.sample_size
    if not isinstance(sample_size, int):
        raise ValueError("this example expects a square UNet sample size")
    generator = torch.Generator(device=args.device)
    generator.manual_seed(0)
    state = torch.randn(
        (1, unet.config.in_channels, sample_size, sample_size),
        generator=generator,
        device=args.device,
    )
    state = state * scheduler.init_noise_sigma
    backend = DiffusersEpsilonBackend(unet, scheduler)
    result = sample(state, list(scheduler.timesteps), alphas, backend)

    image = (result.final_state / 2.0 + 0.5).clamp(0.0, 1.0)
    array = (image[0].permute(1, 2, 0) * 255.0).round().byte().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
