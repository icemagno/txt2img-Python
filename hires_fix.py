import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

def apply_hires_fix(
    pipe: StableDiffusionXLPipeline,
    prompt_embeds,
    negative_prompt_embeds,
    pooled_prompt_embeds,
    negative_pooled_prompt_embeds,
    generator: torch.Generator,
    image: Image.Image,
    upscale_factor: float = 1.5,
    denoising_strength: float = 0.3,  # Lowered default for better composition preservation
    cfg: float = 5.0,
    steps: int = 28,
):
    """
    Applies a high-resolution fix to a generated image using an Img2Img pass.

    Args:
        pipe: The base StableDiffusionXLPipeline object.
        prompt_embeds: Pre-computed positive prompt embeddings.
        negative_prompt_embeds: Pre-computed negative prompt embeddings.
        pooled_prompt_embeds: Pre-computed pooled positive prompt embeddings.
        negative_pooled_prompt_embeds: Pre-computed pooled negative prompt embeddings.
        generator: The torch generator for reproducible results.
        image: The initial low-resolution image to upscale.
        upscale_factor: The factor by which to increase the image resolution.
        denoising_strength: How much to modify the image (0.0-1.0). Lower values
                           preserve the original composition better.
        cfg: The guidance scale.
        steps: The number of inference steps.

    Returns:
        The upscaled and refined PIL Image.
    """
    print("\n--- Starting Hires. Fix ---")

    # --- 0. Create a dedicated Img2Img pipeline from the base pipeline's components ---
    # This ensures the correct pipeline is used, preserving the original aspect ratio.
    img2img_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
    
    # --- 1. Upscale the image ---
    original_width, original_height = image.size
    target_width = int(original_width * upscale_factor)
    target_height = int(original_height * upscale_factor)
    
    # Ensure dimensions are divisible by 8 for the model
    target_width = target_width - (target_width % 8)
    target_height = target_height - (target_height % 8)

    print(f"Upscaling image from {original_width}x{original_height} to {target_width}x{target_height}")
    upscaled_image = image.resize((target_width, target_height), Image.LANCZOS)

    # --- 2. Run Img2Img Pass ---
    print(f"Running Img2Img pass with denoising strength: {denoising_strength}...")
    hires_image = img2img_pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        generator=generator,
        image=upscaled_image,
        strength=denoising_strength,
        guidance_scale=cfg,
        num_inference_steps=steps,
    ).images[0]
    
    print("--- Hires. Fix Complete ---")
    return hires_image