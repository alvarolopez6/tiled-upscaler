import pytest
import numpy as np
import torch

from tiled_upscaler.tiling import (
    blend_tiles,
    blend_latent_tiles,
    compute_tile_coordinates,
    create_gaussian_mask,
)


def test_coordinates_align_the_last_tile_with_both_edges() -> None:
    coordinates = compute_tile_coordinates(
        height=10,
        width=10,
        tile_size=6,
        stride=4,
    )

    assert coordinates == [
        (0, 6, 0, 6),
        (0, 6, 4, 10),
        (4, 10, 0, 6),
        (4, 10, 4, 10),
    ]


def test_all_tiles_stay_inside_the_image() -> None:
    height, width = 17, 23
    tile_size, stride = 8, 5

    coordinates = compute_tile_coordinates(height, width, tile_size, stride)

    for y1, y2, x1, x2 in coordinates:
        assert 0 <= y1 < y2 <= height
        assert 0 <= x1 < x2 <= width
        assert y2 - y1 <= tile_size
        assert x2 - x1 <= tile_size


def test_every_pixel_is_covered() -> None:
    height, width = 17, 23
    coordinates = compute_tile_coordinates(
        height=height,
        width=width,
        tile_size=8,
        stride=5,
    )

    for y in range(height):
        for x in range(width):
            is_covered = any(
                y1 <= y < y2 and x1 <= x < x2
                for y1, y2, x1, x2 in coordinates
            )
            assert is_covered, f"Pixel ({y}, {x}) is not covered"


@pytest.mark.parametrize(
    ("height", "width", "tile_size", "stride"),
    [
        (0, 10, 6, 4),
        (10, 0, 6, 4),
        (10, 10, 0, 4),
        (10, 10, 6, 0),
        (10, 10, 6, 7),
    ],
)
def test_invalid_parameters_raise_value_error(
    height: int,
    width: int,
    tile_size: int,
    stride: int,
) -> None:
    with pytest.raises(ValueError):
        compute_tile_coordinates(height, width, tile_size, stride)


def test_gaussian_mask_has_requested_shape_and_normalized_maximum() -> None:
    mask = create_gaussian_mask(tile_height=5, tile_width=7)

    assert mask.shape == (5, 7)
    assert mask.dtype == np.float32
    assert np.isclose(mask.max(), 1.0)
    assert np.all(mask > 0)


def test_gaussian_mask_is_symmetric() -> None:
    mask = create_gaussian_mask(tile_height=4, tile_width=6)

    assert np.allclose(mask, np.flipud(mask))
    assert np.allclose(mask, np.fliplr(mask))


def test_gaussian_mask_gives_more_weight_to_the_center() -> None:
    mask = create_gaussian_mask(tile_height=5, tile_width=5)

    center = mask[2, 2]
    corner = mask[0, 0]

    assert np.isclose(center, 1.0)
    assert center > corner


@pytest.mark.parametrize(
    ("tile_height", "tile_width", "sigma"),
    [
        (0, 5, 0.3),
        (5, 0, 0.3),
        (5, 5, 0),
        (5, 5, -0.3),
    ],
)
def test_invalid_mask_parameters_raise_value_error(
    tile_height: int,
    tile_width: int,
    sigma: float,
) -> None:
    with pytest.raises(ValueError):
        create_gaussian_mask(tile_height, tile_width, sigma)


def test_blend_tiles_reconstructs_a_constant_image() -> None:
    coordinates = [
        (0, 2, 0, 2),
        (0, 2, 1, 3),
        (1, 3, 0, 2),
        (1, 3, 1, 3),
    ]
    tiles = [np.full((2, 2), 5.0, dtype=np.float32) for _ in coordinates]
    mask = np.array(
        [
            [1.0, 0.5],
            [0.5, 0.25],
        ],
        dtype=np.float32,
    )

    result = blend_tiles(3, 3, coordinates, tiles, mask)

    assert np.allclose(result, 5.0)


def test_blend_tiles_rejects_different_coordinate_and_tile_counts() -> None:
    with pytest.raises(ValueError):
        blend_tiles(
            2,
            2,
            coordinates=[(0, 2, 0, 2)],
            tiles=[],
            mask=np.ones((2, 2), dtype=np.float32),
        )


def test_blend_tiles_rejects_a_tile_with_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        blend_tiles(
            2,
            2,
            coordinates=[(0, 2, 0, 2)],
            tiles=[np.ones((3, 3), dtype=np.float32)],
            mask=np.ones((2, 2), dtype=np.float32),
        )


def test_blend_tiles_rejects_uncovered_output_pixels() -> None:
    with pytest.raises(ValueError):
        blend_tiles(
            3,
            3,
            coordinates=[(0, 2, 0, 2)],
            tiles=[np.ones((2, 2), dtype=np.float32)],
            mask=np.ones((2, 2), dtype=np.float32),
        )


def test_blend_latent_tiles_reconstructs_constant_multi_channel_latents() -> None:
    coordinates = [
        (0, 2, 0, 2),
        (0, 2, 1, 3),
        (1, 3, 0, 2),
        (1, 3, 1, 3),
    ]
    tiles = [
        torch.full((2, 4, 2, 2), 5.0, dtype=torch.float16)
        for _ in coordinates
    ]
    mask = np.array(
        [[1.0, 0.5], [0.5, 0.25]],
        dtype=np.float32,
    )

    result = blend_latent_tiles(3, 3, coordinates, tiles, mask)

    assert result.shape == (2, 4, 3, 3)
    assert result.dtype == torch.float16
    assert torch.allclose(result, torch.full_like(result, 5.0))


def test_blend_latent_tiles_rejects_a_tile_with_the_wrong_shape() -> None:
    with pytest.raises(ValueError):
        blend_latent_tiles(
            2,
            2,
            coordinates=[(0, 2, 0, 2)],
            tiles=[torch.ones((1, 4, 3, 2))],
            mask=np.ones((2, 2), dtype=np.float32),
        )


def test_blend_latent_tiles_rejects_uncovered_output_pixels() -> None:
    with pytest.raises(ValueError):
        blend_latent_tiles(
            3,
            3,
            coordinates=[(0, 2, 0, 2)],
            tiles=[torch.ones((1, 4, 2, 2))],
            mask=np.ones((2, 2), dtype=np.float32),
        )
