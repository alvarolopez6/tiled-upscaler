"""Model engine configuration for the tiled upscaler."""

from __future__ import annotations

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)

from .device import get_device, get_dtype


BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11f1e_sd15_tile"


class UpscalerEngine:
    """Central object that will own the diffusion model components."""

    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = get_dtype(self.device)

        # These components will be loaded incrementally in Phase 2.
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.unet = None
        self.controlnet = None
        self.scheduler = None

    def load_tokenizer(self) -> CLIPTokenizer:
        """Load the CLIP tokenizer used by the Stable Diffusion base model."""
        if self.tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                BASE_MODEL_ID,
                subfolder="tokenizer",
            )

        return self.tokenizer

    def tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Convert a prompt into padded token IDs for the text encoder."""
        tokenizer = self.load_tokenizer()
        encoded_prompt = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt", # Return a dictionary of PyTorch tensors containing the token IDs and attention mask instead of a single python list.
        )
        return encoded_prompt.input_ids

    def load_text_encoder(self) -> CLIPTextModel:
        """Load CLIP's text encoder using the engine's device and dtype."""
        if self.text_encoder is None:
            self.text_encoder = CLIPTextModel.from_pretrained(
                BASE_MODEL_ID,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
            )
            self.text_encoder.to(self.device)
            self.text_encoder.eval()

        return self.text_encoder

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Convert a prompt into CLIP text embeddings."""
        tokenizer = self.load_tokenizer()
        text_encoder = self.load_text_encoder()

        encoded_prompt = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_prompt.input_ids.to(self.device)
        attention_mask = encoded_prompt.attention_mask.to(self.device)

        with torch.no_grad():
            text_outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        return text_outputs.last_hidden_state

    def load_vae(self) -> AutoencoderKL:
        """Load and configure the VAE used to encode and decode images."""
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                BASE_MODEL_ID,
                subfolder="vae",
                torch_dtype=self.dtype,
            )
            self.vae.to(self.device)
            self.vae.enable_tiling()
            self.vae.enable_slicing()
            self.vae.eval()

        return self.vae

    def load_unet(self) -> UNet2DConditionModel:
        """Load the conditional UNet used for noise prediction."""
        if self.unet is None:
            self.unet = UNet2DConditionModel.from_pretrained(
                BASE_MODEL_ID,
                subfolder="unet",
                torch_dtype=self.dtype,
            )
            self.unet.to(self.device)
            self.unet.eval()

        return self.unet

    def load_controlnet(self) -> ControlNetModel:
        """Load the ControlNet Tile model used for structural conditioning."""
        if self.controlnet is None:
            self.controlnet = ControlNetModel.from_pretrained(
                CONTROLNET_MODEL_ID,
                torch_dtype=self.dtype,
            )
            self.controlnet.to(self.device)
            self.controlnet.eval()

        return self.controlnet

    def load_scheduler(self) -> EulerAncestralDiscreteScheduler:
        """Load the scheduler that controls the denoising trajectory."""
        if self.scheduler is None:
            self.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
                BASE_MODEL_ID,
                subfolder="scheduler",
            )

        return self.scheduler
