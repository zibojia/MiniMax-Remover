from typing import Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

import scipy
import numpy as np
import torch.nn.functional as F
from transformer_minimax_remover import Transformer3DModel
from einops import rearrange

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class Minimax_Remover_Pipeline(DiffusionPipeline):

    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 16,
        height: int = 720,
        width: int = 1280,
        num_latent_frames: int = 21,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def expand_masks(self, masks, iterations):
        masks = masks.cpu().detach().numpy()
        # numpy array, masks [0,1], f h w c
        masks2 = []
        for i in range(len(masks)):
            mask = masks[i]
            mask = mask > 0
            mask = scipy.ndimage.binary_dilation(mask, iterations=iterations)
            masks2.append(mask)
        masks = np.array(masks2).astype(np.float32)
        masks = torch.from_numpy(masks)
        masks = masks.repeat(1,1,1,3)
        masks = rearrange(masks, "f h w c -> c f h w")
        masks = masks[None,...]
        return masks

    def resize(self, images, w, h):
        bsz,_,_,_,_ = images.shape
        images = rearrange(images, "b c f w h -> (b f) c w h")
        images = F.interpolate(images, (w,h), mode='bilinear')
        images = rearrange(images, "(b f) c w h -> b c f w h", b=bsz)
        return images

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        images: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        iterations: int = 16
    ):

        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        batch_size = 1
        transformer_dtype = torch.float16

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            num_latent_frames,
            torch.float16,
            device,
            generator,
            latents,
        )

        masks = self.expand_masks(masks, iterations)
        masks = self.resize(masks, height, width).to("cuda:0").half()
        masks[masks>0] = 1
        images = rearrange(images, "f h w c -> c f h w")
        images = self.resize(images[None,...], height, width).to("cuda:0").half()

        masked_images = images * (1-masks)

        latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(self.vae.device, torch.float16)
            )

        latents_std =  1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                self.vae.device, torch.float16
            )

        with torch.no_grad():
            masked_latents = self.vae.encode(masked_images.half()).latent_dist.mode()
            masks_latents = self.vae.encode(2*masks.half()-1.0).latent_dist.mode()

            masked_latents = (masked_latents - latents_mean) * latents_std
            masks_latents = (masks_latents - latents_mean) * latents_std

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = latents.to(transformer_dtype)
                
                #print("latent_model_input, masked_latents, masks_latents", latent_model_input.shape, masked_latents.shape, masks_latents.shape)
                latent_model_input = torch.cat([latent_model_input, masked_latents, masks_latents], dim=1)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input.half(),
                    timestep=timestep
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                progress_bar.update()

        latents = latents.half() / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        return WanPipelineOutput(frames=video)
