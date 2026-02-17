#!/usr/bin/env python
from pathlib import Path

import fire
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def main(
    model_dir=str(Path("~/llms/Qwen3-0.6B").expanduser()),
    prompt="Tell me a short fun fact about the moon.",
    max_new_tokens=128,
):
    model_dir = str(Path(model_dir).expanduser())

    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference required but CUDA is not available.")
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, attn_implementation="eager")
    model.to(device)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    eos_token_id = tokenizer.eos_token_id
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    with torch.inference_mode():
        # --- Prefill: one forward over the full prompt ---
        seq_len = input_ids.shape[1]
        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "cache_position": torch.arange(seq_len, device=device, dtype=torch.long),
        }
        model_inputs = model.prepare_inputs_for_generation(input_ids, is_first_iteration=True, **model_kwargs)
        prefill_outputs = model(**model_inputs, return_dict=True)

        # Logits at last position -> first new token
        next_token_logits = prefill_outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        model_kwargs = model._update_model_kwargs_for_generation(
            prefill_outputs, model_kwargs, is_encoder_decoder=False
        )

        # Stream first generated token
        streamer.put(next_token.cpu())

        # --- Decode: repeated single-token forward with KV cache ---
        for _ in range(max_new_tokens - 1):
            if next_token.item() == eos_token_id:
                break
            model_inputs = model.prepare_inputs_for_generation(next_token, **model_kwargs)
            outputs = model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False
            )
            streamer.put(next_token.cpu())
            if next_token.item() == eos_token_id:
                break

    streamer.end()


if __name__ == "__main__":
    fire.Fire(main)
