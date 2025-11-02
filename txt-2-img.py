import torch
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from PIL import Image
import os
import random

# --- Environment Setup ---
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "./hf_cache"
os.environ["DIFFUSERS_CACHE"] = "./hf_cache"
os.environ["HF_HUB_OFFLINE"] = "0"

# --- Model and Configuration Paths ---
model_path = "/Magno/Projetos/train-model/digitalinfluencersv1.safetensors"
config_path = "/Magno/Projetos/train-model/sd_xl_base.yaml"

# --- Generation Parameters 
seed    = random.randint(0, 2**32 - 1)
cfg     = 5.0
steps   = 28
width   = 960
height  = 1280
# --- Upscale HiRes Fix
upscale_factor = 1.5
upscale_denoise = 0.4


long_prompt = (
    "score_9, score_8_up, score_7_up, source_pony, masterpiece, best quality, "
    "realistic, (full body), 1girl,"
    "masterpiece, best quality, newest, absurdres, highres, girl in armor, open stomach, armguards, "
    "exposed breasts, medium breasts, covered nipples , cybernetic, visible ribcage, beautiful face, long flowing hair"
    "action pose, horny, sexy, sensual"
)

negative_prompt = (
    "worst quality, low quality, normal quality, text, signature, watermark, ugly, deformed, "
    "simple background, plain background, white background, blurry background, out of frame, studio shot, "
    "cartoon, anime, drawing, painting"
)


# --- Load the Pipeline ---

# just_to_download_the_base_files = DiffusionPipeline.from_pretrained(
#    "stable-diffusion-v1-5/stable-diffusion-v1-5",
#    use_safetensors=True
# )

print("Loading model from checkpoint...")
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    original_config_file=config_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=False,
)
print("Model loaded successfully.")
pipe.to("cuda")

# --- ADVANCED: Function to handle long prompts ---
# This function correctly replicates ComfyUI's long prompt handling for SDXL.
def encode_prompt_long(pipe, prompt):
    """
    Encodes a long prompt by breaking it into 77-token chunks for the first
    text encoder and concatenating the results. It uses the second text
    encoder in the standard way for its pooled output.
    """
    device = pipe.device
    
    # Get tokenizers and text encoders
    tokenizer_1 = pipe.tokenizer
    encoder_1 = pipe.text_encoder
    tokenizer_2 = pipe.tokenizer_2
    encoder_2 = pipe.text_encoder_2

    # Tokenize the long prompt for the first encoder without padding or truncation
    tokens_1 = tokenizer_1(prompt, padding="do_not_pad", truncation=False, return_tensors="pt").input_ids.to(device)

    # Tokenize the prompt for the second encoder with standard padding and truncation
    tokens_2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length, truncation=True, return_tensors="pt").input_ids.to(device)

    # --- Get embeddings from the second text encoder ---
    with torch.no_grad():
        # The output object from this encoder is a CLIPTextModelOutput
        output_2 = encoder_2(tokens_2, output_hidden_states=True, return_dict=True)
        
        # The text embeddings are the second to last hidden state
        embeds_2 = output_2.hidden_states[-2]
        
        # *** THE FIX IS HERE ***
        # The pooled output for this specific encoder is in the `text_embeds` attribute, not `pooled_output`.
        pooled_embeds = output_2.text_embeds

    # --- Process the long prompt in chunks with the first text encoder ---
    max_len = tokenizer_1.model_max_length
    # Split the tokens into chunks of the maximum length
    token_chunks_1 = [tokens_1[:, i:i + max_len] for i in range(0, tokens_1.shape[1], max_len)]

    embeds_1_list = []
    for chunk in token_chunks_1:
        # Pad the last chunk if it's smaller than the max length
        if chunk.shape[1] < max_len:
            pad_size = max_len - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_size), "constant", value=tokenizer_1.pad_token_id)
        
        with torch.no_grad():
            output_1 = encoder_1(chunk, output_hidden_states=True)
            # Use the second to last hidden state for the first encoder (OpenCLIP standard)
            embeds_1 = output_1.hidden_states[-2]
            embeds_1_list.append(embeds_1)
    
    # Concatenate the embeddings from all chunks
    embeds_1 = torch.cat(embeds_1_list, dim=1)

    # --- Combine embeddings from both encoders ---
    # We need to pad the embeddings from the second encoder to match the length of the first.
    bs, seq_len_1, _ = embeds_1.shape
    _, seq_len_2, _ = embeds_2.shape
    
    padding_len = seq_len_1 - seq_len_2
    if padding_len > 0:
        padding = torch.zeros(bs, padding_len, embeds_2.shape[2], device=device, dtype=embeds_2.dtype)
        embeds_2 = torch.cat([embeds_2, padding], dim=1)

    # The final prompt embeddings are the concatenation of the two
    prompt_embeds = torch.cat([embeds_1, embeds_2], dim=-1)

    return prompt_embeds, pooled_embeds

# --- Manual Encoding ---
print("Encoding long prompts...")
# Encode the positive prompt
prompt_embeds, pooled_prompt_embeds = encode_prompt_long(pipe, long_prompt)
# Encode the negative prompt
negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_long(pipe, negative_prompt)
print("Encoding complete.")

# --- PADDING ---
# The diffusers pipeline requires the positive and negative prompt embeddings to have the same shape.
# Since our positive prompt is long and our negative prompt is short, we need to pad the negative one.
if prompt_embeds.shape != negative_prompt_embeds.shape:
    print("Padding negative prompt embeddings to match positive prompt embeddings shape.")
    padding_len = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
    
    # Create a padding tensor of zeros.
    padding = torch.zeros(
        negative_prompt_embeds.shape[0], 
        padding_len, 
        negative_prompt_embeds.shape[2], 
        device=pipe.device, 
        dtype=negative_prompt_embeds.dtype
    )
    
    # Concatenate the original negative embeddings with the padding.
    negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)
    print(f"Shape after padding: {negative_prompt_embeds.shape}")

generator = torch.Generator(device="cuda").manual_seed(seed)

# --- Generate the Image using Embeddings ---
print(f"Generating image with seed {seed}...")
# IMPORTANT: We now pass `prompt_embeds` and `pooled_prompt_embeds` instead of `prompt`.
image = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    width=width,
    height=height,
    num_inference_steps=steps,
    guidance_scale=cfg,
    generator=generator,
).images[0]
print("Image generated.")

# --- Save the Output ---
output_path = f"./basic-{seed}.png"
image.save(output_path)
print(f"Image saved successfully to {output_path}")


# --- (Optional) Apply Hires. Fix ---
# To use, simply uncomment the following lines.
from hires_fix import apply_hires_fix
hires_image = apply_hires_fix(
    pipe=pipe,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    generator=generator,
    image=image,
    upscale_factor=upscale_factor,
    denoising_strength=upscale_denoise,
    cfg=cfg,
    steps=steps,
)

hires_output_path = f"./hires-{seed}.png"
hires_image.save(hires_output_path)
print(f"Hires. Fix image saved successfully to {hires_output_path}")
