"""Utilities for splitting images and latent tensors into overlapping tiles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np


TileCoordinates: TypeAlias = tuple[int, int, int, int]


def _compute_axis_starts(
    length: int,
    tile_size: int,
    stride: int,
) -> list[int]:
    """Return valid starting positions for one image axis."""
    last_start = max(0, length - tile_size)
    starts = list(range(0, last_start + 1, stride))

    # The regular stride may not land exactly on the edge. Adding the final
    # valid start guarantees that the last tile ends at ``length``.
    if not starts or starts[-1] != last_start:
        starts.append(last_start)

    return starts


def compute_tile_coordinates(
    height: int,
    width: int,
    tile_size: int,
    stride: int,
) -> list[TileCoordinates]:
    """Calculate overlapping, in-bounds tile coordinates.

    Coordinates use Python's half-open slicing convention:
    ``image[y1:y2, x1:x2]``. Therefore ``y2`` and ``x2`` are excluded.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if stride > tile_size:
        raise ValueError("stride must not be greater than tile_size")

    y_starts = _compute_axis_starts(height, tile_size, stride)
    x_starts = _compute_axis_starts(width, tile_size, stride)

    coordinates: list[TileCoordinates] = []
    for y1 in y_starts:
        y2 = min(y1 + tile_size, height)
        for x1 in x_starts:
            x2 = min(x1 + tile_size, width)
            coordinates.append((y1, y2, x1, x2))

    return coordinates


def create_gaussian_mask(
    tile_height: int,
    tile_width: int,
    sigma: float = 0.3,
) -> np.ndarray:
    """Create a normalized 2D Gaussian blending mask.

    Coordinates are normalized relative to the tile's half-size, so the same
    ``sigma`` produces the same relative shape for different tile sizes.
    The largest weight is normalized to exactly ``1.0``.
    """
    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("tile_height and tile_width must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    y = np.arange(tile_height, dtype=np.float32) # row indices
    x = np.arange(tile_width, dtype=np.float32)

    y_center = (tile_height - 1) / 2 # center of the tile
    x_center = (tile_width - 1) / 2
    y_scale = max(y_center, 1.0)
    x_scale = max(x_center, 1.0)

    y_normalized = (y - y_center) / y_scale # (-1, 1) range
    x_normalized = (x - x_center) / x_scale
    distance_squared = y_normalized[:, None] ** 2 + x_normalized[None, :] ** 2

    mask = np.exp(-distance_squared / (2 * sigma**2))
    mask /= mask.max() # normalize to 1.0 maximum

    return mask.astype(np.float32)


def blend_tiles(
    output_height: int,
    output_width: int,
    coordinates: Sequence[TileCoordinates], # list of (y1, y2, x1, x2)
    tiles: Sequence[np.ndarray], # list of 2D arrays
    mask: np.ndarray,
) -> np.ndarray:
    """Reconstruct a 2D image from overlapping tiles and a blending mask."""
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output dimensions must be positive")
    if len(coordinates) != len(tiles):
        raise ValueError("coordinates and tiles must have the same length")
    if mask.ndim != 2:
        raise ValueError("mask must be a 2D array")

    value_accum = np.zeros((output_height, output_width), dtype=np.float32)
    weight_accum = np.zeros((output_height, output_width), dtype=np.float32)

    for (y1, y2, x1, x2), tile in zip(coordinates, tiles):
        tile_height = y2 - y1
        tile_width = x2 - x1

        if tile.shape != (tile_height, tile_width):
            raise ValueError("tile shape does not match its coordinates")
        if mask.shape[0] < tile_height or mask.shape[1] < tile_width:
            raise ValueError("mask is smaller than a tile")

        tile_mask = mask[:tile_height, :tile_width] # crop mask to tile size (for the last tiles at the edges)
        value_accum[y1:y2, x1:x2] += tile * tile_mask
        weight_accum[y1:y2, x1:x2] += tile_mask

    if np.any(weight_accum == 0):
        raise ValueError("some output pixels received no tile weight")

    return value_accum / weight_accum
