import torch

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
