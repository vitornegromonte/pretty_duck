"""Load a causal language model and its tokenizer the same way in every experiment."""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_model(path: str):
    """Return ``(model, tokenizer)`` in eval mode, on the GPU when one exists.

    The dtype is fp16 on a GPU and bf16 on the CPU. Every compared configuration is evaluated in the same dtype.
    A multimodal checkpoint (for example Qwen3.5) loses its vision tower, because the pipeline only uses text.
    """
    import transformers

    # transformers renamed torch_dtype to dtype in v5; support both.
    kwarg = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.bfloat16
    config = AutoConfig.from_pretrained(path)
    if getattr(config, "vision_config", None) is not None:
        # AutoModelForCausalLM would silently skip the text weights of a multimodal checkpoint.
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(path, **{kwarg: dtype})
        if hasattr(model.model, "visual"):
            del model.model.visual  # the vision tower costs 1 to 2 GB of memory and never runs on text
    else:
        model = AutoModelForCausalLM.from_pretrained(path, **{kwarg: dtype})
    if cuda:
        model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model.eval(), tokenizer
