"""Device and precision selection for the upscaler.

The priority is CUDA (when available), then Apple's MPS backend, and finally
the CPU. GPU backends use float16 to reduce memory usage; the CPU uses
float32 for compatibility.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_dtype(device: torch.device | None = None) -> torch.dtype:
    """Return the dtype to use for a device.

    GPU inference uses float16 to reduce memory consumption. CPU inference
    remains on float32 because it is the most broadly supported option.
    """
    selected_device = device or get_device()
    return torch.float16 if selected_device.type in {"cuda", "mps"} else torch.float32


def check_device() -> tuple[torch.device, torch.dtype, torch.Tensor]:
    """Run a small matrix operation and return its selected configuration."""
    device = get_device()
    dtype = get_dtype(device)

    left = torch.ones((2, 2), device=device, dtype=dtype)
    right = torch.eye(2, device=device, dtype=dtype) # identity matrix
    result = left @ right

    return device, dtype, result


if __name__ == "__main__":
    device, dtype, result = check_device()
    print(f"device={device}\n")
    print(f"dtype={dtype}\n")
    print(f"matrix_result={result}")