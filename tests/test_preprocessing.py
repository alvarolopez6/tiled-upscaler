import numpy as np
import pytest
import torch
from PIL import Image

from tiled_upscaler.preprocessing import preprocess_image


def test_preprocess_image_returns_expected_tensors() -> None:
    image = Image.new("RGB", (64, 48), color=(255, 128, 0))

    vae_image, control_image = preprocess_image(image, scale=2.0)

    assert vae_image.shape == (1, 3, 96, 128)
    assert control_image.shape == (1, 3, 96, 128)
    assert vae_image.dtype == torch.float32
    assert control_image.dtype == torch.float32
    assert torch.allclose(vae_image, control_image * 2.0 - 1.0)
    assert control_image.min() >= 0.0
    assert control_image.max() <= 1.0


def test_preprocess_image_converts_non_rgb_images() -> None:
    grayscale_image = Image.fromarray(
        np.full((16, 16), 127, dtype=np.uint8),
    )

    vae_image, control_image = preprocess_image(grayscale_image)

    assert vae_image.shape == (1, 3, 32, 32)
    assert control_image.shape == (1, 3, 32, 32)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan")])
def test_preprocess_image_rejects_invalid_scale(scale: float) -> None:
    image = Image.new("RGB", (16, 16))

    with pytest.raises(ValueError):
        preprocess_image(image, scale)
