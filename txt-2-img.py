# --- Environment Setup ---
# python -m venv venv
# .\venv\Scripts\activate
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  
# python -m pip install transformers auto-gptq optimum datasets peft                                    
# python -m pip install numpy pandas
# python -m pip install tensorflow numpy tf-keras diffusers transformers accelerate
# python -m pip install xformers==0.0.23 
# python -m pip install hf_xet
# python -m pip install torchsde

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["HF_HOME"]                   = "./hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"]     = "./hf_cache"
os.environ["DIFFUSERS_CACHE"]           = "./hf_cache"
os.environ["HF_HUB_OFFLINE"]            = "0"

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, DPMSolverSDEScheduler,DPMSolverMultistepScheduler
from PIL import Image
import random


# ------------------------------------------------------------------------------
# --- Model and Configuration Paths ---
model_path              = "/Magno/Projetos/train-model/digitalinfluencersv1.safetensors"
hf_model_id             = "John6666/goddess-of-realism-gor-pony-v2art-sdxl" 
config_path             = "/Magno/Projetos/train-model/sd_xl_base.yaml"
# --- Generation Parameters 
seed                    = 820590516     # random.randint(0, 2**32 - 1)
batch_size              = 1             # Quantas imagens você quer gerar?
cfg                     = 7.0           # Classifier-Free Guidance Scale - Criatividade versus Prompt (abstração <-> fidelidade)
steps                   = 20            # Quantidade de "pinceladas" na difusão ( rascunho <-> refinamento)
width                   = 960           # Largura da imagem ( siga o padrão SDXL )
height                  = 1280          # Altura da imagem  ( siga o padrão SDXL )
clip_skip               = 2             # Quantas camadas finais pular do CLIP Text Encoder
# --- Upscale HiRes Fix
use_hires_fix           = False         # Usar HiRes Fix Upscaler? Vai ampliar a imagem e melhorar a resolução
upscale_factor          = 1.5           # Quer ampliar quantas vezes a imagem original
upscale_denoise         = 0.45          # O quanto quer modificar a imagem original no processo?
# --- Prompt File Paths ---
long_prompt_file        = "./prompt-2.txt"          # O prompt
negative_prompt_file    = "./negative_prompt.txt"   # O que quer evitar na imagem?

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------






# ----------------------------------------------------------------------
# ------------------- Ler os prompts do arquivo ------------------------
def load_prompt_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Prompt file not found at {filepath}")
        exit()
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

long_prompt = load_prompt_from_file(long_prompt_file)
negative_prompt = load_prompt_from_file(negative_prompt_file)


# ----------------------------------------------------------------------
print("Lendo / Baixando o modelo...")
# Se você baixou algum checkpoint do site CivitAI ...
# --- Load the Pipeline ---
# just_to_download_the_base_files = DiffusionPipeline.from_pretrained(
#    "stable-diffusion-v1-5/stable-diffusion-v1-5",
#    use_safetensors=True
# )

pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    original_config=config_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=False,
)
# ----------------------------------------------------------------------
# ... ou se quer que o script baixe o modelo para você do site Hugging Face.
#pipe = StableDiffusionXLPipeline.from_pretrained(
#    hf_model_id,
#    use_safetensors=True,
#    torch_dtype=torch.float16,
#)
# ----------------------------------------------------------------------

print("Modelo carregado.")
pipe.to("cuda")


# ----------------------------------------------------------------------
# Escolha UM Scheduler apenas:

# Scheduler Euler A 
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Scheduler Euler A + Karras
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
#   pipe.scheduler.config,
#   timestep_spacing="karras"
#)

# Scheduler DPM++ SDE Karras
pipe.scheduler = DPMSolverSDEScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True
)

# Scheduler DPM++ 1S
#pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
    # pipe.scheduler.config,
    # use_karras_sigmas=True
#)

# Scheduler DPM++ 2M
#pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#    pipe.scheduler.config,
#    use_karras_sigmas=True
#)
# ----------------------------------------------------------------------



# ----------------------------------------------------------------------
# --- Método para passar a limitação de 77 tokens ---
# --- Cortesia do Gemini CLI
def encode_prompt_long(pipe, prompt, clip_skip=0):
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
        
        # The text embeddings are the (clip_skip + 1)th to last hidden state
        embeds_2 = output_2.hidden_states[-(1 + clip_skip)]
        
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
            # Use the (clip_skip + 1)th to last hidden state for the first encoder (OpenCLIP standard)
            embeds_1 = output_1.hidden_states[-(1 + clip_skip)]
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
prompt_embeds, pooled_prompt_embeds = encode_prompt_long(pipe, long_prompt, clip_skip=clip_skip)
# Encode the negative prompt
negative_prompt_embeds, negative_pooled_prompt_embeds = encode_prompt_long(pipe, negative_prompt, clip_skip=clip_skip)
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
    

# ----------------------------------------------------------------------
# Geração da imagem

generator = torch.Generator(device="cuda").manual_seed(seed)
print(f"Gerando {batch_size} imagen(s) com semente {seed}...")
generated_images = pipe(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    width=width,
    height=height,
    num_inference_steps=steps,
    guidance_scale=cfg,
    generator=generator,
    num_images_per_prompt=batch_size,
).images
print("Imagen(s) geradas.")

# ----------------------------------------------------------------------
# --- Grava todas as imagens geradas no batch
for i, image in enumerate(generated_images):
    output_path = f"./basic-{seed}-{i+1}.png"
    image.save(output_path)
    print(f"Imagem {i+1} gravada como {output_path}")


# ----------------------------------------------------------------------
# --- Optou por usar HiRes Fix Upscaler?
if use_hires_fix:
    from hires_fix import apply_hires_fix
    for i, image in enumerate(generated_images):
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

        hires_output_path = f"./hires-{seed}-{i+1}.png"
        hires_image.save(hires_output_path)
        print(f"Imagem ampliada {i+1} gravada como {hires_output_path}")
