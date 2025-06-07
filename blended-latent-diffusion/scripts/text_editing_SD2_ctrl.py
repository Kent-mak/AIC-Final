import argparse
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel
import torch
# import cv2
import torchvision.transforms as T


class BlendedLatentDiffusion:
    def __init__(self):
        self.parse_args()
        self.load_models()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", type=str, required=True, help="The target text prompt")
        parser.add_argument("--init_image", type=str, required=True, help="The path to the input image")
        parser.add_argument("--mask", type=str, required=True, help="The path to the input mask")
        parser.add_argument("--control_image", type=str, required=True, help="The path to the control image for ControlNet")
        parser.add_argument("--controlnet_path", type=str, required=True, help="Path to ControlNet model")
        parser.add_argument(
            "--model_path",
            type=str,
            default="stabilityai/stable-diffusion-2-1-base",
            help="The path to the HuggingFace model",
        )
        parser.add_argument("--batch_size", type=int, default=4, help="The number of images to generate")
        parser.add_argument("--blending_start_percentage", type=float, default=0.25, help="The diffusion steps percentage to jump")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--output_path", type=str, default="outputs/res_controlnet.jpg", help="The destination output path")
        self.args = parser.parse_args()

    def load_models(self):
        controlnet = ControlNetModel.from_pretrained(self.args.controlnet_path, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.args.model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to(self.args.device)
        self.pipe.enable_xformers_memory_efficient_attention()
        self.controlnet = self.pipe.controlnet
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = self.pipe.text_encoder
        self.unet = self.pipe.unet
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )

    @torch.no_grad()
    def edit_image(
        self,
        image_path,
        mask_path,
        control_image_path,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        generator=torch.manual_seed(42),
        blending_percentage=0.25,
    ):
        batch_size = len(prompts)
        image = Image.open(image_path).convert("RGB").resize((height, width))
        source_latents = self._image2latent(np.array(image))
        latent_mask, org_mask = self._read_mask(mask_path)

        control_image = self._prepare_depth_map(control_image_path, height, width)
        control_image_tensor = T.ToTensor()(control_image).unsqueeze(0).to(self.args.device).half()

        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.args.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.args.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        latents = torch.randn(
            (batch_size, self.unet.in_channels, height // 8, width // 8),
            generator=generator,
        ).to(self.args.device).half()

        self.scheduler.set_timesteps(num_inference_steps, device=self.args.device)

        for t in self.scheduler.timesteps[int(len(self.scheduler.timesteps) * blending_percentage):]:
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            down_samples, mid_sample = self.controlnet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=control_image_tensor
            ).to_tuple()

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            # Blending
            noise_source_latents = self.scheduler.add_noise(
                source_latents, torch.randn_like(latents), t
            )
            latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        return images

    @torch.no_grad()
    def _image2latent(self, image):
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0).to(self.args.device).half()
        latents = self.vae.encode(image)["latent_dist"].mean * 0.18215
        return latents

    def _read_mask(self, mask_path: str, dest_size=(64, 64)):
        org_mask = Image.open(mask_path).convert("L")
        mask = org_mask.resize(dest_size, Image.NEAREST)
        mask = np.array(mask) / 255
        mask = (mask >= 0.5).astype(np.float32)
        mask = torch.from_numpy(mask[np.newaxis, np.newaxis, ...]).half().to(self.args.device)
        return mask, org_mask

    def _prepare_depth_map(self, path, height, width):
        # Assume the image is a colormapped RGB depth map
        image = Image.open(path).convert("RGB").resize((width, height))
        return image


if __name__ == "__main__":
    bld = BlendedLatentDiffusion()
    results = bld.edit_image(
        bld.args.init_image,
        bld.args.mask,
        bld.args.control_image,
        prompts=[bld.args.prompt] * bld.args.batch_size,
        blending_percentage=bld.args.blending_start_percentage,
    )
    results_flat = np.concatenate(results, axis=1)
    Image.fromarray(results_flat).save(bld.args.output_path)
