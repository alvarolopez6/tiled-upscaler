"""Model engine configuration for the tiled upscaler."""

from __future__ import annotations

import torch
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    UNet2DConditionModel,
)

from .device import get_device, get_dtype
from .preprocessing import preprocess_image
from .tiling import (
    blend_latent_tiles,
    compute_tile_coordinates,
    create_gaussian_mask,
)


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

    def encode_prompt_pair(
        self,
        prompt: str,
        negative_prompt: str = "",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode negative and positive prompts for classifier-free guidance."""
        tokenizer = self.load_tokenizer()
        text_encoder = self.load_text_encoder()

        encoded_prompts = tokenizer(
            [negative_prompt, prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded_prompts.input_ids.to(self.device)
        attention_mask = encoded_prompts.attention_mask.to(self.device)

        with torch.no_grad():
            text_outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

        unconditional_embeddings = text_outputs[0:1]
        conditional_embeddings = text_outputs[1:2]
        return unconditional_embeddings, conditional_embeddings

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

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a normalized RGB tensor into scaled VAE latents.

        The input must have shape ``[batch, 3, height, width]`` and values in
        the ``[-1, 1]`` range. Spatial dimensions must be divisible by 8.
        """
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        if image.shape[2] % 8 != 0 or image.shape[3] % 8 != 0:
            raise ValueError("image height and width must be divisible by 8")

        vae = self.load_vae()
        image = image.to(device=self.device, dtype=self.dtype)

        with torch.no_grad():
            latent_distribution = vae.encode(image).latent_dist
            latents = latent_distribution.sample()

        return latents * vae.config.scaling_factor

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode scaled VAE latents into an RGB tensor in the ``[0, 1]`` range."""
        if latents.ndim != 4 or latents.shape[1] != 4:
            raise ValueError("latents must have shape [batch, 4, height, width]")

        vae = self.load_vae()
        latents = latents.to(device=self.device, dtype=self.dtype)
        latents = latents / vae.config.scaling_factor

        with torch.no_grad():
            image = vae.decode(latents).sample

        return (image / 2 + 0.5).clamp(0, 1)

    def inject_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor | int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add scheduler-controlled noise to scaled latents."""
        if latents.ndim != 4 or latents.shape[1] != 4:
            raise ValueError("latents must have shape [batch, 4, height, width]")

        scheduler = self.load_scheduler()
        latents = latents.to(device=self.device, dtype=self.dtype)

        if noise is None:
            noise = torch.randn_like(latents)
        else:
            noise = noise.to(device=self.device, dtype=self.dtype)

        if isinstance(timestep, int):
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=self.device)

        return scheduler.add_noise(latents, noise, timestep)

    @staticmethod
    def creativity_to_strength(creativity: float) -> float:
        """Map creativity from [0, 1] to a conservative diffusion strength."""
        if not 0.0 <= creativity <= 1.0:
            raise ValueError("creativity must be between 0.0 and 1.0")

        minimum_strength = 0.15
        maximum_strength = 0.75
        return minimum_strength + creativity * (
            maximum_strength - minimum_strength
        )

    def prepare_noisy_latents(
        self,
        latents: torch.Tensor,
        creativity: float,
        steps: int,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add creativity-controlled noise and return denoising timesteps."""
        if steps <= 0:
            raise ValueError("steps must be positive")

        scheduler = self.load_scheduler()
        scheduler.set_timesteps(steps, device=self.device)
        strength = self.creativity_to_strength(creativity)

        init_timestep = min(int(steps * strength), steps)
        start_step = max(steps - init_timestep, 0)
        timesteps = scheduler.timesteps[start_step:]
        # add_noise expects one timestep per item in the batch.
        timestep = timesteps[:1].repeat(latents.shape[0])

        noisy_latents = self.inject_noise(latents, timestep, noise)
        return noisy_latents, timesteps

    def denoise_step(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor | int,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run one basic UNet and scheduler denoising step."""
        unet = self.load_unet()
        scheduler = self.load_scheduler()

        latents = latents.to(device=self.device, dtype=self.dtype)
        text_embeddings = text_embeddings.to(
            device=self.device,
            dtype=self.dtype,
        )

        if isinstance(timestep, int):
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=self.device)

        model_input = scheduler.scale_model_input(latents, timestep)

        with torch.no_grad():
            noise_prediction = unet(
                model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample

        return scheduler.step(
            noise_prediction,
            timestep,
            latents,
        ).prev_sample

    def predict_noise_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor | int,
        unconditional_embeddings: torch.Tensor,
        conditional_embeddings: torch.Tensor,
        control_image: torch.Tensor,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict guided noise with ControlNet and CFG, without stepping the scheduler."""
        if guidance_scale < 0:
            raise ValueError("guidance_scale must not be negative")
        if conditioning_scale < 0:
            raise ValueError("conditioning_scale must not be negative")
        if control_image.ndim != 4 or control_image.shape[1] != 3:
            raise ValueError("control_image must have shape [batch, 3, height, width]")
        if control_image.shape[0] != latents.shape[0]:
            raise ValueError("control_image and latents must have the same batch size")

        unet = self.load_unet()
        controlnet = self.load_controlnet()
        scheduler = self.load_scheduler()
        latents = latents.to(device=self.device, dtype=self.dtype)
        control_image = control_image.to(device=self.device, dtype=self.dtype)
        unconditional_embeddings = unconditional_embeddings.to(
            device=self.device,
            dtype=self.dtype,
        )
        conditional_embeddings = conditional_embeddings.to(
            device=self.device,
            dtype=self.dtype,
        )

        if isinstance(timestep, int):
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=self.device)

        latent_model_input = torch.cat([latents, latents])
        latent_model_input = scheduler.scale_model_input(
            latent_model_input,
            timestep,
        )
        control_model_input = torch.cat([control_image, control_image])
        text_embeddings = torch.cat(
            [unconditional_embeddings, conditional_embeddings]
        )

        with torch.no_grad():
            down_block_res_samples, mid_block_res_sample = controlnet(
                sample=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_model_input,
                conditioning_scale=conditioning_scale,
                return_dict=False,
            )
            noise_predictions = unet(
                latent_model_input,
                timestep,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

        unconditional_noise, conditional_noise = noise_predictions.chunk(2)
        guided_noise = unconditional_noise + guidance_scale * (
            conditional_noise - unconditional_noise
        )

        return guided_noise

    def denoise_step_cfg(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor | int,
        unconditional_embeddings: torch.Tensor,
        conditional_embeddings: torch.Tensor,
        control_image: torch.Tensor,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict guided noise, then update latents once with the scheduler."""
        latents = latents.to(device=self.device, dtype=self.dtype)
        if isinstance(timestep, int):
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=self.device)

        guided_noise = self.predict_noise_cfg(
            latents,
            timestep,
            unconditional_embeddings,
            conditional_embeddings,
            control_image,
            guidance_scale,
            conditioning_scale,
        )

        scheduler = self.load_scheduler()
        return scheduler.step(
            guided_noise,
            timestep,
            latents,
        ).prev_sample

    def denoise_step_cfg_tiled(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor | int,
        unconditional_embeddings: torch.Tensor,
        conditional_embeddings: torch.Tensor,
        control_image: torch.Tensor,
        tile_size: int,
        stride: int,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict noise per latent tile, blend it, then update the full latents."""
        if latents.ndim != 4 or latents.shape[1] != 4:
            raise ValueError("latents must have shape [batch, 4, height, width]")
        if control_image.ndim != 4 or control_image.shape[1] != 3:
            raise ValueError(
                "control_image must have shape [batch, 3, height, width]"
            )
        if control_image.shape[0] != latents.shape[0]:
            raise ValueError("control_image and latents must have the same batch size")

        latents = latents.to(device=self.device, dtype=self.dtype)
        control_image = control_image.to(device=self.device, dtype=self.dtype)
        latent_height, latent_width = latents.shape[-2:]

        vae = self.load_vae()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        expected_control_size = (
            latent_height * vae_scale_factor,
            latent_width * vae_scale_factor,
        )
        if control_image.shape[-2:] != expected_control_size:
            raise ValueError(
                "control_image spatial size must match the latents times the "
                f"VAE scale factor ({vae_scale_factor})"
            )

        if isinstance(timestep, int):
            timestep = torch.tensor(timestep, device=self.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=self.device)

        coordinates = compute_tile_coordinates(
            latent_height,
            latent_width,
            tile_size,
            stride,
        )
        mask = create_gaussian_mask(
            min(tile_size, latent_height),
            min(tile_size, latent_width),
        )

        noise_tiles: list[torch.Tensor] = []
        for y1, y2, x1, x2 in coordinates:
            latent_tile = latents[:, :, y1:y2, x1:x2]
            pixel_tile = control_image[
                :,
                :,
                y1 * vae_scale_factor : y2 * vae_scale_factor,
                x1 * vae_scale_factor : x2 * vae_scale_factor,
            ]
            noise_tiles.append(
                self.predict_noise_cfg(
                    latent_tile,
                    timestep,
                    unconditional_embeddings,
                    conditional_embeddings,
                    pixel_tile,
                    guidance_scale,
                    conditioning_scale,
                )
            )

        guided_noise = blend_latent_tiles(
            latent_height,
            latent_width,
            coordinates,
            noise_tiles,
            mask,
        )

        scheduler = self.load_scheduler()
        return scheduler.step(guided_noise, timestep, latents).prev_sample

    def denoise(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Run the basic denoising loop without ControlNet or CFG."""
        if timesteps.ndim != 1 or timesteps.numel() == 0:
            raise ValueError("timesteps must be a non-empty 1D tensor")

        for timestep in timesteps:
            latents = self.denoise_step(
                latents,
                timestep,
                text_embeddings,
            )

        return latents

    def denoise_cfg(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        unconditional_embeddings: torch.Tensor,
        conditional_embeddings: torch.Tensor,
        control_image: torch.Tensor,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run the ControlNet-conditioned denoising loop with CFG."""
        if timesteps.ndim != 1 or timesteps.numel() == 0:
            raise ValueError("timesteps must be a non-empty 1D tensor")

        for timestep in timesteps:
            latents = self.denoise_step_cfg(
                latents,
                timestep,
                unconditional_embeddings,
                conditional_embeddings,
                control_image,
                guidance_scale,
                conditioning_scale,
            )

        return latents

    def denoise_cfg_tiled(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        unconditional_embeddings: torch.Tensor,
        conditional_embeddings: torch.Tensor,
        control_image: torch.Tensor,
        tile_size: int,
        stride: int,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 1.0,
    ) -> torch.Tensor:
        """Run the ControlNet and CFG denoising loop over latent tiles."""
        if timesteps.ndim != 1 or timesteps.numel() == 0:
            raise ValueError("timesteps must be a non-empty 1D tensor")

        for timestep in timesteps:
            latents = self.denoise_step_cfg_tiled(
                latents,
                timestep,
                unconditional_embeddings,
                conditional_embeddings,
                control_image,
                tile_size,
                stride,
                guidance_scale,
                conditioning_scale,
            )

        return latents

    def upscale(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        scale: float = 2.0,
        creativity: float = 0.35,
        resemblance: float = 0.85,
        steps: int = 20,
        guidance_scale: float = 7.5,
        tile_size: int = 512,
        overlap: int = 256,
    ) -> torch.Tensor:
        """Upscale a PIL image and return an RGB tensor in the [0, 1] range.

        ``tile_size`` and ``overlap`` are specified in output-image pixels.
        The returned tensor has shape ``[1, 3, height, width]``.
        """
        if steps <= 0:
            raise ValueError("steps must be positive")
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if overlap < 0 or overlap >= tile_size:
            raise ValueError("overlap must be non-negative and smaller than tile_size")

        vae_image, control_image = preprocess_image(image, scale)
        vae = self.load_vae()
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

        if tile_size % vae_scale_factor != 0:
            raise ValueError("tile_size must be divisible by the VAE scale factor")
        pixel_stride = tile_size - overlap
        if pixel_stride % vae_scale_factor != 0:
            raise ValueError("tile_size - overlap must be divisible by the VAE scale factor")

        latents = self.encode_image(vae_image)
        unconditional_embeddings, conditional_embeddings = self.encode_prompt_pair(
            prompt,
            negative_prompt,
        )
        noisy_latents, timesteps = self.prepare_noisy_latents(
            latents,
            creativity,
            steps,
        )

        final_latents = self.denoise_cfg_tiled(
            noisy_latents,
            timesteps,
            unconditional_embeddings,
            conditional_embeddings,
            control_image,
            tile_size // vae_scale_factor,
            pixel_stride // vae_scale_factor,
            guidance_scale,
            resemblance,
        )

        return self.decode_latents(final_latents)

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
