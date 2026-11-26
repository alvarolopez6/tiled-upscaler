"""Command-line interface for the tiled upscaler."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import typer
from PIL import Image

from .engine import UpscalerEngine
from .tiling import compute_tile_coordinates


def upscale_command(
    input: Path = typer.Option(
        ...,
        "--input",
        "-i",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="Path to the input image.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path where the upscaled image will be saved.",
    ),
    scale: float = typer.Option(2.0, min=0.01, help="Output scale factor."),
    creativity: float = typer.Option(
        0.35,
        min=0.0,
        max=1.0,
        help="How much freedom diffusion has to change the input.",
    ),
    resemblance: float = typer.Option(
        0.85,
        min=0.0,
        help="ControlNet conditioning strength.",
    ),
    steps: int = typer.Option(20, min=1, help="Number of scheduler steps."),
    tile_size: int = typer.Option(
        512,
        min=8,
        help="Tile size in output-image pixels.",
    ),
    overlap: int = typer.Option(
        256,
        min=0,
        help="Overlap in output-image pixels.",
    ),
    prompt: str = typer.Option(
        "high resolution, sharp focus, intricate textures, photorealistic",
        help="Positive text prompt.",
    ),
    negative_prompt: str = typer.Option(
        "",
        help="Negative text prompt.",
    ),
    guidance_scale: float = typer.Option(
        3.0,
        min=0.0,
        help="Classifier-free guidance strength.",
    ),
) -> None:
    """Upscale an image and save the result."""
    if input.resolve() == output.resolve():
        raise typer.BadParameter("input and output paths must be different")

    try:
        with Image.open(input) as source_image:
            image = source_image.convert("RGB")
    except OSError as error:
        raise typer.BadParameter(f"could not open input image: {error}") from error

    engine = UpscalerEngine()
    typer.echo(f"Input: {input} ({image.width}x{image.height})")
    typer.echo(f"Device: {engine.device}; dtype: {engine.dtype}")
    typer.echo("Running tiled diffusion...")

    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.device)

    start_time = time.perf_counter()
    try:
        result = engine.upscale(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            scale=scale,
            creativity=creativity,
            resemblance=resemblance,
            steps=steps,
            guidance_scale=guidance_scale,
            tile_size=tile_size,
            overlap=overlap,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    elapsed = time.perf_counter() - start_time

    output_array = result[0].permute(1, 2, 0).float().cpu().numpy()
    output_array = np.clip(np.rint(output_array * 255), 0, 255).astype(np.uint8)
    output_image = Image.fromarray(output_array)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_image.save(output)

    vae = engine.load_vae()
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_height = result.shape[-2] // vae_scale_factor
    latent_width = result.shape[-1] // vae_scale_factor
    latent_tile_size = tile_size // vae_scale_factor
    latent_stride = (tile_size - overlap) // vae_scale_factor
    tile_count = len(
        compute_tile_coordinates(
            latent_height,
            latent_width,
            latent_tile_size,
            latent_stride,
        )
    )

    typer.echo(f"Output: {output} ({output_image.width}x{output_image.height})")
    typer.echo(f"Tiles per denoising step: {tile_count}")
    effective_steps = min(
        int(steps * engine.creativity_to_strength(creativity)),
        steps,
    )
    typer.echo(
        f"Timesteps processed: {effective_steps} of {steps}; elapsed: {elapsed:.2f}s"
    )

    if engine.device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(engine.device)
        typer.echo(f"CUDA peak allocated memory: {peak_bytes / 1024**3:.2f} GiB")
    elif engine.device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        current_bytes = torch.mps.current_allocated_memory()
        typer.echo(f"MPS currently allocated memory: {current_bytes / 1024**3:.2f} GiB")


def main() -> None:
    """Run the CLI command."""
    typer.run(upscale_command)


if __name__ == "__main__":
    main()
