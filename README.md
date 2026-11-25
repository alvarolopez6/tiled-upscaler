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

The engine can also be used from Python with `UpscalerEngine.upscale()`. It
accepts a PIL image and returns an RGB tensor `[1, 3, H, W]` in `[0, 1]`.

## Tests

```bash
uv run pytest
```

Tests cover tile coordinates, Gaussian masks, 2D NumPy blending, PyTorch latent
blending, and image preprocessing. The last reported run passed 28 test cases;
that run preceded recent CLI changes. Unit tests do not load model weights;
inference validation is separate and requires substantial memory.

## Current structure

```text
src/tiled_upscaler/
├── device.py          # device and precision selection
├── preprocessing.py   # PIL image → VAE and ControlNet tensors
├── tiling.py          # coordinates, masks, and blending
├── engine.py          # models and manual tiled diffusion loop
└── cli.py             # Typer command-line interface
tests/
├── test_tiling.py
└── test_preprocessing.py
assets/input/          # local images; ignored by Git
```

## Status and limitations

- Phases 0–2: implemented and validated.
- Phase 3: tiled end-to-end flow is implemented and has been tested with a
  synthetic input and a local photograph. Quality, memory use, and different
  input sizes still need further evaluation.
- Phase 4: CLI implemented and `--help` checked. A CLI-driven inference and
  complete profiling are still pending.
- Phase 5: Docker is pending.
- This is an experimental project. Results can vary with the prompt,
  `creativity`, `resemblance`, and `steps`.

Inference runs locally; Hugging Face access is used to download model files.
Check each model's license and terms before use or redistribution.

## License

The project's license has not yet been selected.
