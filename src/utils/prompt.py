"""
Prompt encoding utilities for DSM2DTM.

Consolidates the encode_prompt function that was duplicated across 5 scripts.
"""

import torch
from transformers import CLIPTokenizer, CLIPTextModel


@torch.no_grad()
def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    device: torch.device,
    prompt: str,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Encode a text prompt into CLIP embeddings.

    Args:
        tokenizer: CLIP tokenizer.
        text_encoder: CLIP text encoder.
        device: Target device.
        prompt: Text prompt string.
        batch_size: Number of copies to repeat.

    Returns:
        Tensor of shape [batch_size, seq_len, hidden_dim].
    """
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    emb = text_encoder(input_ids)[0]
    emb = emb.repeat(batch_size, 1, 1)
    return emb
