"""Image preprocessing helpers used before VAE and ControlNet inference."""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image


def preprocess_image(
    image: Image.Image,
    scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize a PIL image and return VAE and ControlNet input tensors.

    The VAE tensor has shape ``[1, 3, height, width]`` and values in ``[-1, 1]``.
    The ControlNet tensor has the same shape and values in ``[0, 1]``.
    Output dimensions are rounded to the nearest multiple of 8.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")

    image = image.convert("RGB")
    original_width, original_height = image.size

    target_width = max(8, round(original_width * scale / 8) * 8)
    target_height = max(8, round(original_height * scale / 8) * 8)
    resized_image = image.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )

    image_array = np.asarray(resized_image, dtype=np.float32).copy() / 255.0
    control_image = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
    control_image = control_image.contiguous()
    vae_image = control_image * 2.0 - 1.0

    return vae_image, control_image
