# Tiled Latent Diffusion Upscaler

[English](README.md) | [Español](README.es.md)

An educational image-upscaling project built with Stable Diffusion 1.5 and
ControlNet Tile. Model components and the diffusion loop are implemented
explicitly; the project does not use an opaque high-level image-generation
pipeline.

**Status: experimental.** The end-to-end engine has been exercised on Apple
Silicon with MPS, including tiled denoising. Results may change image details
and are not guaranteed to preserve every input pixel.

## What it does

The engine resizes an input image, encodes it into VAE latents, adds noise
according to `creativity`, then denoises with the UNet, ControlNet Tile,
Classifier-Free Guidance, and the Euler Ancestral scheduler. Noise predictions
are computed per tile and blended before the global latents are updated.

```text
image + prompt
      ↓
VAE → noisy latents
      ↓
ControlNet Tile + UNet + scheduler, per tile
      ↓
VAE → upscaled image
```

## Requirements

- Python 3.12 (specified in `.python-version`).
- [`uv`](https://docs.astral.sh/uv/).
- Enough memory to load Stable Diffusion 1.5 and ControlNet Tile.
- Hugging Face Hub access to download the models on first use.

The device is selected automatically: CUDA, MPS, then CPU. GPU inference uses
`float16`; CPU inference uses `float32`. Tiling reduces activation memory, but
**the model weights still require several GB of memory**. CPU inference can be
very slow, and GPUs with little VRAM may need smaller tiles.

## Installation

```bash
uv sync
```

Model weights are downloaded to the Hugging Face cache on first use. Set
`HF_HOME` to use a different cache location. Model weights and input images are
not stored in this repository.

## CLI usage

Show the available options without running inference:

```bash
uv run tiled-upscaler --help
```

Example:

```bash
uv run tiled-upscaler \
  --input photo.jpg \
  --output "$HOME/Desktop/photo_upscaled.png" \
  --scale 2 \
  --creativity 0.25 \
  --resemblance 0.85 \
  --guidance-scale 3.0 \
  --steps 20 \
  --tile-size 512 \
  --overlap 256 \
  --prompt "high resolution landscape photograph, sharp focus, detailed textures"
```

`tile-size` and `overlap` are specified in output-image pixels and must be
compatible with the VAE spatial factor (8 for SD 1.5). `steps` is the schedule
length; `creativity` determines how many steps are actually used. The CLI
reports the device, tiles per denoising step, effective steps, and elapsed time.
On CUDA it reports peak allocated memory; on MPS it reports current allocated
memory, not the peak.

Current defaults are `guidance_scale=3.0` and a linear `creativity` to `strength`
mapping of `[0,1] → [0.05,0.45]`. These are provisional educational settings,
not universally optimal values.

The engine can also be used from Python with `UpscalerEngine.upscale()`. It
accepts a PIL image and returns an RGB tensor `[1, 3, H, W]` in `[0, 1]`.

## Early visual evaluation

This is an educational diffusion prototype, not a production super-resolution
system. We compared local beach (`256×256 → 512×512`) and lake
(`512×512 → 1024×1024`) photos with Lanczos and VAE-only baselines. The private
input/output images are not included in the repository.

Controlled runs used fixed seeds, 20 scheduler steps, `tile_size=512`,
`overlap=256`, and prompts describing each scene. With the current creativity
mapping (`[0,1] → strength [0.05,0.45]`):

| Creativity | Strength | Effective steps (of 20) | Observation |
|---:|---:|---:|---|
| 0.00 | 0.05 | 1 | Very conservative; close to the VAE reconstruction. |
| 0.25 | 0.15 | 3 | Preserves broad composition, but can remain soft. |
| 0.50 | 0.25 | 5 | Allows more reinterpretation; no consistent sharpness gain. |

A CFG comparison at creativity `0.25` found that scales `1–3` generally kept
the composition better than `7.5`; high guidance could create visible artifacts
with generic prompts. Across these two photos, diffusion did **not consistently
outperform Lanczos in sharpness**. This is an important limitation of the
current model/configuration; adding more steps alone did not reliably fix it.

## Docker on Linux

The Docker image targets Linux. The base Compose configuration does not require
a GPU, so PyTorch can fall back to CPU. On a Linux host with NVIDIA Container
Toolkit, add the NVIDIA override to expose the GPU:

```bash
docker compose build
docker compose run --rm tiled-upscaler --help
```

CPU inference example:

```bash
docker compose run --rm tiled-upscaler \
  --input /input/photo.jpg \
  --output /output/photo_upscaled.png \
  --steps 20
```

NVIDIA GPU inference example:

```bash
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml run --rm tiled-upscaler \
  --input /input/photo.jpg \
  --output /output/photo_upscaled.png \
  --steps 20
```

The host must have a compatible NVIDIA driver and NVIDIA Container Toolkit.
The image is based on CUDA 13.2.1 to align with the CUDA 13.2 dependencies in
the current Linux `uv.lock` resolution. Docker on macOS cannot pass Apple MPS to
this Linux container; use native `uv` execution on a Mac for MPS acceleration.

## Tests

```bash
uv run pytest
```

Tests cover tile coordinates, Gaussian masks, 2D NumPy blending, PyTorch latent
blending, image preprocessing, and the creativity-to-strength mapping. The
latest reported run passed 33 test cases. Unit tests do not load model weights;
inference validation is separate and requires substantial memory.

## Current structure

```text
src/tiled_upscaler/
├── device.py          # device and precision selection
├── preprocessing.py   # PIL image → VAE and ControlNet tensors
├── tiling.py          # coordinates, masks, and blending
├── engine.py          # models and manual tiled diffusion loop
└── cli.py             # Typer command-line interface
Dockerfile             # Linux/CUDA image
docker-compose.yml     # CPU-compatible Compose service
docker-compose.nvidia.yml # optional NVIDIA GPU override
tests/
├── test_tiling.py
├── test_engine.py
└── test_preprocessing.py
assets/input/          # local images; ignored by Git
```

## Status and limitations

- Phases 0–2: implemented and validated.
- Phase 3: tiled end-to-end flow has been tested with synthetic input and local
  photos. Visual evaluation shows that current diffusion outputs can be softer
  than Lanczos; quality still needs work.
- Phase 4: CLI `--help` and real inference were tested on MPS and NVIDIA CUDA.
  Profiling is partial: MPS reports current allocated memory, not peak.
- Phase 5: Docker CPU/NVIDIA configurations are implemented and an NVIDIA
  inference ran technically. Rebuild and revalidate after the latest
  guidance/creativity calibration; CPU-only Docker is not yet tested.
- This is an experimental project. Results can vary with the prompt,
  `creativity`, `resemblance`, and `steps`.

Inference runs locally; Hugging Face access is used to download model files.
Check each model's license and terms before use or redistribution.

## License

The project's license has not yet been selected.
