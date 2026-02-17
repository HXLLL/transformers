#!/usr/bin/env python
import time
from types import SimpleNamespace

import fire
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def select_device(args):
    if args.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return args.device


def benchmark_prefill_decode(model, tokenizer, prompt, device, decode_tokens, prefill_tokens, vocab_size):
    if prefill_tokens > 0:
        start_id = 100
        seq = torch.arange(start_id, start_id + prefill_tokens, device=device) % vocab_size
        input_ids = seq.unsqueeze(0).long()
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0
    prefill_tokens = int(input_ids.shape[-1])

    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(decode_tokens):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)],
                dim=-1,
            )
            out = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if device == "cuda":
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1

    print("\n=== benchmark ===")
    print(f"prefill_tokens={prefill_tokens}")
    print(f"prefill_latency_s={prefill_s:.4f}")
    print(f"prefill_tokens_per_s={prefill_tokens / prefill_s:.2f}")
    print(f"decode_tokens={decode_tokens}")
    print(f"decode_latency_s={decode_s:.4f}")
    print(f"decode_tokens_per_s={decode_tokens / decode_s:.2f}")


def patch_bnb_qwen3_retry():
    # Local workaround for a Qwen3 + bitsandbytes module replacement path issue.
    import bitsandbytes as bnb
    from torch import nn

    import transformers.integrations as integrations_pkg
    from transformers.integrations import bitsandbytes as bnb_integration
    from transformers.pytorch_utils import Conv1D

    should_convert_module = bnb_integration.should_convert_module

    def get_parent_and_key(root_module, module_name):
        parts = module_name.split(".")
        parent = root_module
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        return parent, parts[-1]

    def safe_replace_with_bnb_linear(model, modules_to_not_convert=None, quantization_config=None, pre_quantized=False):
        modules_to_not_convert = modules_to_not_convert or []
        has_been_replaced = False
        # Use a static snapshot since we mutate the module tree during traversal.
        for module_name, module in list(model.named_modules()):
            if not should_convert_module(module_name, modules_to_not_convert):
                continue
            if not isinstance(module, (nn.Linear, Conv1D)):
                continue

            if isinstance(module, Conv1D):
                in_features, out_features = module.weight.shape
            else:
                in_features = module.in_features
                out_features = module.out_features

            with torch.device("meta"):
                if quantization_config.quantization_method() == "llm_int8":
                    new_module = bnb.nn.Linear8bitLt(
                        in_features,
                        out_features,
                        module.bias is not None,
                        has_fp16_weights=quantization_config.llm_int8_has_fp16_weight,
                        threshold=quantization_config.llm_int8_threshold,
                    )
                    if pre_quantized:
                        new_module.weight.data = new_module.weight.data.to(dtype=torch.int8)
                else:
                    new_module = bnb.nn.Linear4bit(
                        in_features,
                        out_features,
                        module.bias is not None,
                        quantization_config.bnb_4bit_compute_dtype,
                        compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                        quant_type=quantization_config.bnb_4bit_quant_type,
                        quant_storage=quantization_config.bnb_4bit_quant_storage,
                    )
                    if pre_quantized:
                        new_module.weight.data = new_module.weight.data.to(dtype=quantization_config.bnb_4bit_quant_storage)

            new_module.source_cls = type(module)
            new_module.requires_grad_(False)
            parent, key = get_parent_and_key(model, module_name)
            if key.isdigit():
                parent[int(key)] = new_module
            else:
                setattr(parent, key, new_module)
            has_been_replaced = True

        if not has_been_replaced:
            raise RuntimeError("No linear modules were replaced by bitsandbytes in the workaround path.")
        return model

    bnb_integration.replace_with_bnb_linear = safe_replace_with_bnb_linear
    integrations_pkg.replace_with_bnb_linear = safe_replace_with_bnb_linear


def main(
    model_dir="/home/hxl/llms/Qwen3-0.6B",
    prompt="你好，请用一句话介绍你自己。",
    max_new_tokens=64,
    attn_impl="eager",
    device="cuda",
    require_cuda=True,
    allow_cpu=False,
    benchmark=False,
    decode_tokens=64,
    prefill_tokens=0,
    quant="none",
):
    valid_attn_impl = {"eager", "sdpa", "flash_attention_2", "flash_attention_3"}
    if attn_impl not in valid_attn_impl:
        raise ValueError(f"Invalid attn_impl={attn_impl!r}. Choose one of: {sorted(valid_attn_impl)}")

    valid_device = {"auto", "cpu", "cuda"}
    if device not in valid_device:
        raise ValueError(f"Invalid device={device!r}. Choose one of: {sorted(valid_device)}")

    valid_quant = {"none", "int8", "int4"}
    if quant not in valid_quant:
        raise ValueError(f"Invalid quant={quant!r}. Choose one of: {sorted(valid_quant)}")

    args = SimpleNamespace(
        model_dir=model_dir,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        attn_impl=attn_impl,
        device=device,
        require_cuda=require_cuda,
        allow_cpu=allow_cpu,
        benchmark=benchmark,
        decode_tokens=decode_tokens,
        prefill_tokens=prefill_tokens,
        quant=quant,
    )

    device = select_device(args)
    require_cuda = args.require_cuda and not args.allow_cpu
    if require_cuda and (device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError(
            "CUDA is required but unavailable. Check `nvidia-smi`, CUDA driver, and PyTorch CUDA build."
        )
    if args.allow_cpu and device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable, falling back to CPU because --allow-cpu is set.")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device}, dtype={dtype}, attn_implementation={args.attn_impl}, quant={args.quant}")

    config = AutoConfig.from_pretrained(args.model_dir)
    # Some local checkpoints include both lm_head and embed_tokens; disable tying to avoid noisy warning.
    config.tie_word_embeddings = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    load_kwargs = {}
    if args.quant == "none":
        load_kwargs["dtype"] = dtype
    else:
        if device != "cuda":
            raise RuntimeError("bitsandbytes quantization requires CUDA device.")
        try:
            from transformers import BitsAndBytesConfig
        except Exception as error:
            raise RuntimeError(
                "Failed to import BitsAndBytesConfig. Install bitsandbytes first in this environment."
            ) from error

        if args.quant == "int8":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        load_kwargs["device_map"] = {"": 0}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            config=config,
            attn_implementation=args.attn_impl,
            **load_kwargs,
        )
    except AttributeError as error:
        if args.quant != "none" and "`model` is not an nn.Module" in str(error):
            print("Applying local bitsandbytes patch for Qwen3 and retrying model load...")
            patch_bnb_qwen3_retry()
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir,
                config=config,
                attn_implementation=args.attn_impl,
                **load_kwargs,
            )
        else:
            raise
    if args.quant == "none":
        model = model.to(device)
    model.eval()

    # New generation stack warns if these sampling fields are set while do_sample=False.
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\n=== output ===")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    if args.benchmark:
        benchmark_prefill_decode(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            decode_tokens=args.decode_tokens,
            prefill_tokens=args.prefill_tokens,
            vocab_size=int(config.vocab_size),
        )


if __name__ == "__main__":
    fire.Fire(main)
