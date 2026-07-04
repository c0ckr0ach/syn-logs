"""
config_loader.py
~~~~~~~~~~~~~~~~
Load and validate a simplified dataset configuration JSON file.
Returns a SimpleNamespace object for dot-access to all config fields
(e.g. cfg.generation.batch_size, cfg.model_id).

Required top-level keys: dataset_name, model_id, generation
All other sections (log_format, etc.) are optional.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

# ─── Required keys ────────────────────────────────────────────────────────────

_REQUIRED_KEYS = ["dataset_name", "model_id", "generation"]

# ─── Model alias table ────────────────────────────────────────────────────────
# Maps short human-friendly names → full HuggingFace model IDs.
# All listed models are free and open-source (no HF token required,
# except gemma/* which requires accepting the licence at huggingface.co).

MODEL_ALIASES: dict[str, str] = {
    # Qwen 2.5 family (Alibaba) — recommended default
    "qwen":       "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b":     "Qwen/Qwen2.5-7B-Instruct",
    "qwen14b":    "Qwen/Qwen2.5-14B-Instruct",

    # DeepSeek-R1 distilled family — free, strong reasoning
    "deepseek":   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek1":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "deepseek14": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",

    # Gemma 2 family (Google) — requires HF token after licence acceptance
    "gemma":      "google/gemma-2-2b-it",
    "gemma9b":    "google/gemma-2-9b-it",

    # Phi-3.5 (Microsoft) — very fast small model
    "phi":        "microsoft/Phi-3.5-mini-instruct",
    "phi4":       "microsoft/phi-4",

    # Mistral (free, no token needed)
    "mistral":    "mistralai/Mistral-7B-Instruct-v0.3",
}

# ─── Default generation parameters ───────────────────────────────────────────
# These are merged in when the config does not explicitly set a value.
_GEN_DEFAULTS: dict = {
    "target_logs":        1000,
    "batch_size":         20,
    "temperature":        0.85,
    "max_new_tokens":     1024,
    "output_file":        "synthetic_logs.txt",
    "checkpoint_file":    "checkpoint.txt",
    "checkpoint_every_n": 200,
    "anomaly_rate":       0.15,
    "sample_window_size": 30,
    "write_labels":       True,   # also produce a _labeled.jsonl file
}


# ─── Public API ───────────────────────────────────────────────────────────────

def load_config(config_path: str) -> SimpleNamespace:
    """
    Load a dataset config JSON and return it as a SimpleNamespace.

    Parameters
    ----------
    config_path : str
        Path to a config file, e.g. ``'config_healthapp.json'``.

    Returns
    -------
    SimpleNamespace
        All config keys accessible as attributes.
        ``cfg.model_id`` is resolved from MODEL_ALIASES if a shorthand is used.
        ``cfg.generation`` is guaranteed to have every key in _GEN_DEFAULTS.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If any of the required top-level keys are missing.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict = json.load(f)

    # Validate
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Config '{config_path}' is missing required keys: {missing}")

    # Resolve model alias (case-insensitive)
    raw["model_id"] = MODEL_ALIASES.get(raw["model_id"].lower(), raw["model_id"])

    # Merge generation defaults (don't overwrite values already in config)
    gen_raw = raw.get("generation", {})
    for k, v in _GEN_DEFAULTS.items():
        gen_raw.setdefault(k, v)
    raw["generation"] = gen_raw

    cfg = _to_ns(raw)
    print(
        f"✅ Config loaded  →  dataset : '{cfg.dataset_name}'\n"
        f"                     model   : '{cfg.model_id}'\n"
        f"                     target  : {cfg.generation.target_logs} logs"
    )
    return cfg


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_ns(obj):
    """Recursively convert dict → SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(i) for i in obj]
    return obj


# ─── CLI self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config_healthapp.json"
    cfg = load_config(cfg_path)
    g = cfg.generation
    print(f"\n   Batch size       : {g.batch_size}")
    print(f"   Temperature      : {g.temperature}")
    print(f"   Anomaly rate     : {g.anomaly_rate}")
    print(f"   Output file      : {g.output_file}")

    if hasattr(cfg, "log_format"):
        lf = cfg.log_format
        auto = getattr(lf, "auto_detect", False)
        print(f"\n   Log format       : {'auto-detect' if auto else getattr(lf, 'log_type', 'custom')}")
