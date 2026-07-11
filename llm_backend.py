"""
llm_backend.py
~~~~~~~~~~~~~~
HuggingFace Transformers backend for synthetic log generation.
Supports 4-bit NF4 quantization via bitsandbytes, enabling large models
to run within the 15 GB VRAM of a Google Colab T4 GPU.

Supported model families (all free, open-source):
  ┌──────────────┬─────────────────────────────────────────────┬───────────┐
  │  Alias       │  HuggingFace model ID                       │  VRAM*    │
  ├──────────────┼─────────────────────────────────────────────┼───────────┤
  │  qwen        │  Qwen/Qwen2.5-3B-Instruct                   │  ~2 GB    │
  │  qwen7b      │  Qwen/Qwen2.5-7B-Instruct                   │  ~5 GB    │
  │  deepseek1   │  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  │  ~1.5 GB  │
  │  deepseek    │  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B    │  ~5 GB    │
  │  gemma       │  google/gemma-2-2b-it  †                    │  ~2 GB    │
  │  gemma9b     │  google/gemma-2-9b-it  †                    │  ~6 GB    │
  │  phi         │  microsoft/Phi-3.5-mini-instruct            │  ~2 GB    │
  │  mistral     │  mistralai/Mistral-7B-Instruct-v0.3         │  ~5 GB    │
  └──────────────┴─────────────────────────────────────────────┴───────────┘
  * Approximate VRAM with 4-bit NF4 quantization on Colab T4.
  † Gemma requires accepting the licence at huggingface.co and passing
    hf_token=<your_token> to build_model().

Install in Colab (run once in a code cell before importing this module):
  !pip install transformers bitsandbytes accelerate huggingface_hub -q
"""

from __future__ import annotations

import collections
import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ─── Model alias table (mirrors config_loader.MODEL_ALIASES) ─────────────────

SUPPORTED_MODELS: dict[str, str] = {
    "qwen":       "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b":     "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b":    "Qwen/Qwen2.5-14B-Instruct",
    "deepseek":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek1":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek14": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "gemma":      "google/gemma-2-2b-it",
    "gemma9b":    "google/gemma-2-9b-it",
    "phi":        "microsoft/Phi-3.5-mini-instruct",
    "phi4":       "microsoft/phi-4",
    "mistral":    "mistralai/Mistral-7B-Instruct-v0.3",
}


# ─── Build model + tokenizer ─────────────────────────────────────────────────

def build_model(
    model_id:     str,
    load_in_4bit: bool       = True,
    device_map:   str        = "auto",
    hf_token:     str | None = None,
) -> tuple:
    """
    Load a HuggingFace causal language model with optional 4-bit quantization.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID or a shorthand alias (e.g. ``'qwen'``, ``'deepseek'``).
        Aliases are resolved via :data:`SUPPORTED_MODELS`.
    load_in_4bit : bool
        Enable NF4 4-bit quantization via ``bitsandbytes``.
        Strongly recommended on Colab T4.  Falls back to fp16 if
        ``bitsandbytes`` is not installed.
    device_map : str
        Device placement strategy passed to ``from_pretrained``.
        ``"auto"`` distributes across available GPU(s); use ``"cpu"`` to force
        CPU-only mode (slow).
    hf_token : str | None
        HuggingFace API token.  Required only for gated/licensed models
        (e.g. Gemma).  Set ``HF_TOKEN`` env-var or pass explicitly.

    Returns
    -------
    tuple[AutoModelForCausalLM, AutoTokenizer]
        The loaded model (in eval mode) and its matching tokenizer.
    """
    # Resolve shorthand alias → full model ID
    resolved = SUPPORTED_MODELS.get(model_id.lower(), model_id)
    if resolved != model_id:
        print(f"[Model] Alias '{model_id}'  →  {resolved}")

    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        print("[Model] ⚠️  No GPU detected — switching to CPU mode (slow).")
        load_in_4bit = False
        device_map   = "cpu"

    print(f"[Model] Loading  : {resolved}")
    print(f"        4-bit QNT: {'✓ NF4 via bitsandbytes' if load_in_4bit else '✗ fp16 / fp32'}")
    print(f"        Device   : {device_map}")

    # ── Quantization config ───────────────────────────────────────────────────
    quant_config = None
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401 — just checking it's installed
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,   # double quant saves ~0.4 GB extra
                bnb_4bit_quant_type="nf4",
            )
        except ImportError:
            print("[Model] ⚠️  bitsandbytes not installed — falling back to fp16.")

    # Phi-3/3.5 models from the Hub can have outdated modeling code that errors on
    # newer transformers versions with: AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'.
    # Disabling trust_remote_code forces transformers to use the correct native implementation.
    use_remote_code = True
    if "phi-3" in resolved.lower() or "phi3" in resolved.lower() or "phi-4" in resolved.lower():
        use_remote_code = False

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        resolved,
        token=hf_token,
        trust_remote_code=use_remote_code,
    )
    # Some models don't have a pad token — use EOS as fallback
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Model ─────────────────────────────────────────────────────────────────
    model_kwargs: dict = {
        "device_map":        device_map,
        "trust_remote_code": use_remote_code,
        "token":             hf_token,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = (
            torch.float16 if gpu_available else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(resolved, **model_kwargs)
    model.eval()

    # ── Memory report ─────────────────────────────────────────────────────────
    if gpu_available:
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(
            f"[Model] ✅ Ready — GPU memory: {mem_used:.2f} GB used "
            f"/ {mem_total:.1f} GB total"
        )
    else:
        print("[Model] ✅ Ready (CPU mode)")

    return model, tokenizer


# ─── Text generation ─────────────────────────────────────────────────────────

def generate_text(
    model,
    tokenizer,
    messages:       list[dict],
    max_new_tokens: int   = 1024,
    temperature:    float = 0.85,
    do_sample:      bool  = True,
) -> str:
    """
    Apply the model's native chat template and generate text.

    Parameters
    ----------
    model : AutoModelForCausalLM
        Loaded model (from :func:`build_model`).
    tokenizer : AutoTokenizer
        Matching tokenizer.
    messages : list[dict]
        Chat messages in ``[{"role": ..., "content": ...}]`` format.
    max_new_tokens : int
        Maximum number of *new* tokens to generate (does not include prompt).
    temperature : float
        Sampling temperature.  Higher values → more varied output.
    do_sample : bool
        ``True`` for temperature-based sampling; ``False`` for greedy decoding.

    Returns
    -------
    str
        The newly generated text (prompt tokens are stripped from the output).
        DeepSeek-R1 ``<think>...</think>`` reasoning blocks are removed.
    """
    # Encode with the model's native chat template
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device if hasattr(model, "parameters") else (model.device or "cuda")
    # If tokenizer.apply_chat_template returns a dict-like BatchEncoding structure
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids.to(device)
    elif isinstance(encoded, (dict, collections.abc.Mapping)) or (isinstance(encoded, object) and "input_ids" in dir(encoded)):
        # Fallback for dict-like structures
        input_ids = encoded["input_ids"].to(device)
    else:
        try:
            input_ids = encoded.to(device)
        except AttributeError:
            # If it's a dict-like/BatchEncoding but missed other checks
            input_ids = encoded["input_ids"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # discourage verbatim repetition of sample lines
        )

    # Decode only the newly generated tokens (slice off the prompt)
    new_ids  = output_ids[0][input_ids.shape[-1]:]
    raw_text = tokenizer.decode(new_ids, skip_special_tokens=True)

    # Remove DeepSeek-R1 chain-of-thought reasoning blocks
    raw_text = _strip_thinking_tokens(raw_text)

    return raw_text


# ─── Internal helpers ────────────────────────────────────────────────────────

def _strip_thinking_tokens(text: str) -> str:
    """Remove ``<think>…</think>`` blocks produced by DeepSeek-R1 models."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


# ─── Colab install snippet ───────────────────────────────────────────────────

def print_install_instructions() -> None:
    """Print the Colab cell commands needed to install all dependencies."""
    print("=" * 68)
    print("# ── Colab setup (paste into a code cell and run) ──────────────")
    print()
    print("!pip install transformers bitsandbytes accelerate huggingface_hub -q")
    print()
    print("# ── For Gemma (requires licence acceptance at huggingface.co): ─")
    print("# import os; os.environ['HF_TOKEN'] = 'hf_YOUR_TOKEN_HERE'")
    print()
    print("# ── Then run the pipeline: ─────────────────────────────────────")
    print("# !python generate_logs.py \\")
    print("#     --config  config_healthapp.json \\")
    print("#     --input   my_sample_logs.txt    \\")
    print("#     --target  2000")
    print("=" * 68)


if __name__ == "__main__":
    print_install_instructions()
