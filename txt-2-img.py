# ----------------------------------------------------------------------
# python -m venv venv       # Roda isso só uma vez!
# .\venv\Scripts\activate
# python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  
# python -m pip install transformers auto-gptq optimum datasets peft                                    
# python -m pip install numpy pandas
# python -m pip install tensorflow numpy tf-keras diffusers transformers accelerate
# python -m pip install xformers==0.0.23 
# python -m pip install hf_xet
# python -m pip install torchsde
# python -m pip install peft
# ----------------------------------------------------------------------

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["HF_HOME"]                   = "./hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"]     = "./hf_cache"
os.environ["DIFFUSERS_CACHE"]           = "./hf_cache"
os.environ["HF_HUB_OFFLINE"]            = "0"

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, LCMScheduler, DPMSolverSDEScheduler,DPMSolverMultistepScheduler
from PIL import Image
import random
from peft import LoraConfig, get_peft_model
from prompt_encoder import encode_prompt_long
from hires_fix import apply_hires_fix

# Checkpoints baixados do CivitAI:
# https://civitai.com/models/897413
# [ x ] bigLove_pony3.safetensors
# [ x ] bigLove_insta1.safetensors
# [ x ] bigLove_xl4.safetensors
# [ x ] bigLove_photo2.safetensors
# [ x ] realDream_sdxlPony14.safetensors
# ----------------------------------------------------------------------

# --- Modelos 
model_path              = "/Magno/Projetos/train-model/bigLove_xl4.safetensors"    # Está usando um checkpoint local baixado do CivitAI?
config_path             = "/Magno/Projetos/train-model/sd_xl_base.yaml"             # Usei os checkpoints Pony Diffusion então precisei disso
save_preview            = True                                                      # Grava imagens intermediarias do Sampler   
# --- Generation Parameters 
seed                    = 941758023     # random.randint(0, 2**32 - 1) se quiser gerar nova imagem a cada rodada
batch_size              = 1             # Quantas imagens você quer gerar?
cfg                     = 4             # Classifier-Free Guidance Scale - Criatividade versus Prompt (abstração <-> fidelidade)
steps                   = 20            # Quantidade de "pinceladas" na difusão ( rascunho <-> refinamento)
# 1024x1496 960x1280 832x1216
width                   = 832           # Largura da imagem ( siga o padrão SDXL ) 
height                  = 1216          # Altura da imagem  ( siga o padrão SDXL )
clip_skip               = 2             # Quantas camadas finais pular do CLIP Text Encoder
# --- Upscale HiRes Fix
use_hires_fix           = True         # Usar HiRes Fix Upscaler? Vai ampliar a imagem e melhorar a resolução
upscale_factor          = 1.5           # Quer ampliar quantas vezes a imagem original?
upscale_denoise         = 0.45          # O quanto quer modificar a imagem original no processo de upscaling?
# --- Prompt File Paths ---
long_prompt_file        = "./prompt-2.txt"          # O prompt
negative_prompt_file    = "./negative_prompt.txt"   # O que quer evitar na imagem?
# ----------------------------------------------------------------------
use_local               = True
# ----------------------------------------------------------------------


# os.mkdir("./sampler-images")
os.makedirs(f"./sampler-images/{seed}", exist_ok=True)
os.makedirs(f"./output-images/{seed}", exist_ok=True)

# ----------------------------------------------------------------------
# ----------------- Ler os prompts do arquivo --------------------------
def load_prompt_from_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: Prompt file not found at {filepath}")
        exit()
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

long_prompt = load_prompt_from_file(long_prompt_file)
negative_prompt = load_prompt_from_file(negative_prompt_file)


# ----------------------------------------------------------------------
print(f"Carregando Modelo {model_path} ")
pipe = StableDiffusionXLPipeline.from_single_file(
    model_path,
    original_config=config_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    local_files_only=False
)
# ----------------------------------------------------------------------

print("Modelo carregado.")
pipe.to("cuda")
pipe.enable_attention_slicing()
pipe.enable_vae_tiling()

# ----------------------------------------------------------------------
# Escolha UM Scheduler apenas:

# ----------------------------------------------------------------------
# Scheduler Euler A 
# ----------------------------------------------------------------------
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
  pipe.scheduler.config
)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Scheduler LCM Sampler
# ----------------------------------------------------------------------
#pipe.scheduler = LCMScheduler.from_config(
#    pipe.scheduler.config
#)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Scheduler Euler A Karras
# ----------------------------------------------------------------------
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
#   pipe.scheduler.config,
#   timestep_spacing="karras"
# )
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Scheduler DPM++ SDE Karras
# ----------------------------------------------------------------------
#pipe.scheduler = DPMSolverSDEScheduler.from_config(
#    pipe.scheduler.config,
#    use_karras_sigmas=True
#)
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Scheduler DPM++ Karras
# ----------------------------------------------------------------------
# pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
    # pipe.scheduler.config,
    # use_karras_sigmas=True
# )
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Scheduler DPM++ 2M Karras
# ----------------------------------------------------------------------
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(
#    pipe.scheduler.config,
#    use_karras_sigmas=True
# )
# ----------------------------------------------------------------------


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
    num_images_per_prompt=batch_size
).images
print("Imagen(s) geradas.")

# ----------------------------------------------------------------------
# --- Grava todas as imagens geradas no batch
for i, image in enumerate(generated_images):

    output_path = f"./output-images/{seed}/basic-{i+1}.png"
    image.save(output_path)
    print(f"Imagem {i+1} gravada como {output_path}")


# ----------------------------------------------------------------------
# --- Optou por usar HiRes Fix Upscaler?
if use_hires_fix:
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

        hires_output_path = f"./output-images/{seed}/hires-{i+1}.png"
        hires_image.save(hires_output_path)
        print(f"Imagem ampliada {i+1} gravada como {hires_output_path}")
