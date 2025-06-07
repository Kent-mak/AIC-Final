# DATEGS

## Blended Diffusion
'''
python blended-latent-diffusion/scripts/text_editing_SD2_ctrl.py --prompt "a woman in red shirt" --init_image blended-latent-diffusion/inputs/test_image_512.jpg --mask blended-latent-diffusion/inputs/test_mask.png --control_image blended-latent-diffusion/inputs/test_depth.png --controlnet_path ControlNet/model_VIDIT-FAID_1e-5_10_epoch/checkpoint-10000/controlnet --batch_size 1 --output_path blended-latent-diffusion/outputs/generated_1.png --device cuda

'''

'''
bld = BlendedLatentDiffusion.from_parameters(
    prompt="A blonde caucasian woman looking forward",
    init_image="blended-latent-diffusion/inputs/test_image_512.jpg",
    mask="blended-latent-diffusion/inputs/test_mask.png",
    control_image="blended-latent-diffusion/inputs/test_depth.png",
    controlnet_path="ControlNet/model_VIDIT-FAID_1e-5_10_epoch/checkpoint-30000/controlnet",
    model_path="stabilityai/stable-diffusion-2-1-base",
    batch_size=1,
    blending_start_percentage=0.25,
    device="cuda",
    output_path="blended-latent-diffusion/outputs/generated_test.png"
)

results = bld.edit_image()
'''
