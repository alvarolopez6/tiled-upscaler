"""Tiled Latent Diffusion and ControlNet upscaler."""


def main() -> None:
    """Delegate to the command-line interface."""
    from .cli import main as cli_main

    cli_main()
