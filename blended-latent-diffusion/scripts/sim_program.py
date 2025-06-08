from text_editing_SD2_ctrl import BlendedLatentDiffusion
import numpy as np
from PIL import Image

bld = BlendedLatentDiffusion.from_parameters(
    prompt="A blonde caucasian woman looking forward",
    init_image="blended-latent-diffusion/inputs/test_image_512.jpg",
    mask="blended-latent-diffusion/inputs/test_mask.png",
    control_image="blended-latent-diffusion/inputs/test_depth.png",
    controlnet_path="ControlNet/model_VIDIT-FAID_1e-5_10_epoch",
    model_path="stabilityai/stable-diffusion-2-1-base",
    batch_size=1,
    blending_start_percentage=0.25,
    device="cuda",
    output_path="blended-latent-diffusion/outputs/generated_ablation_test.png"
)

results = bld.edit_image()
results_flat = np.concatenate(results, axis=1)
Image.fromarray(results_flat).save(bld.args.output_path)

# Save or process results
