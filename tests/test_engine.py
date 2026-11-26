import pytest

from tiled_upscaler.engine import UpscalerEngine


@pytest.mark.parametrize(
    ("creativity", "expected_strength"),
    [
        (0.0, 0.05),
        (0.25, 0.15),
        (1.0, 0.45),
    ],
)
def test_creativity_maps_to_conservative_strength(
    creativity: float,
    expected_strength: float,
) -> None:
    assert UpscalerEngine.creativity_to_strength(creativity) == pytest.approx(
        expected_strength
    )


@pytest.mark.parametrize("creativity", [-0.01, 1.01])
def test_creativity_outside_range_is_rejected(creativity: float) -> None:
    with pytest.raises(ValueError):
        UpscalerEngine.creativity_to_strength(creativity)
