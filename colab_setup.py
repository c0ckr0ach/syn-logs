"""
colab_setup.py
~~~~~~~~~~~~~~
One-shot setup script for Google Colab.
Run this file ONCE at the start of a Colab session to install all
required packages.  After it finishes, run the pipeline with generate_logs.py.

Usage (in a Colab code cell):
    !python colab_setup.py

Or import and call from a notebook cell:
    from colab_setup import install, print_usage
    install()
    print_usage()
"""

from __future__ import annotations

import subprocess
import sys


# ─── Packages to install ─────────────────────────────────────────────────────

_PACKAGES = [
    "torch>=2.1.0",
    "transformers>=4.44.0",
    "accelerate>=0.30.0",
    "bitsandbytes>=0.43.0",
    "huggingface_hub>=0.23.0",
]


# ─── Install ─────────────────────────────────────────────────────────────────

def install(quiet: bool = True) -> None:
    """
    Install all required packages via pip.

    Parameters
    ----------
    quiet : bool
        Suppress pip output (``-q`` flag).  Set False for verbose output.
    """
    print("=" * 60)
    print("  Synthetic Log Generator — Colab Setup")
    print("=" * 60)
    print(f"\n[Setup] Installing {len(_PACKAGES)} packages...\n")

    cmd = [sys.executable, "-m", "pip", "install"] + _PACKAGES
    if quiet:
        cmd.append("-q")

    result = subprocess.run(cmd, capture_output=quiet)

    if result.returncode != 0:
        print("[Setup] ⚠️  pip install returned non-zero exit code.")
        if quiet:
            print("        Re-run with install(quiet=False) to see full output.")
    else:
        print("[Setup] ✅ All packages installed successfully.")

    print()


# ─── Usage instructions ───────────────────────────────────────────────────────

def print_usage() -> None:
    """Print a quick-start guide for running the pipeline in Colab."""
    msg = """
╔══════════════════════════════════════════════════════════════╗
║        Synthetic Log Generator — Quick Start (Colab)        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Upload your sample log file                              ║
║     from google.colab import files                           ║
║     uploaded = files.upload()   # pick your .txt log file   ║
║                                                              ║
║  2. Choose a config  (or use config_generic.json for any     ║
║     unknown log type):                                       ║
║     config_healthapp.json  — Android HealthApp logs          ║
║     config_windows.json    — Windows Event logs              ║
║     config_linux.json      — Linux syslog / journald         ║
║     config_generic.json    — auto-detect any format          ║
║                                                              ║
║  3. Run the pipeline:                                        ║
║     !python generate_logs.py \\                              ║
║         --config config_generic.json \\                      ║
║         --input  my_sample_logs.txt  \\                      ║
║         --target 2000                                        ║
║                                                              ║
║  4. Override the model at runtime:                           ║
║     --model qwen        Qwen/Qwen2.5-3B-Instruct (~2 GB)    ║
║     --model qwen7b      Qwen/Qwen2.5-7B-Instruct (~5 GB)    ║
║     --model deepseek1   DeepSeek-R1-Distill 1.5B (~2 GB)    ║
║     --model deepseek    DeepSeek-R1-Distill 7B   (~5 GB)    ║
║     --model phi         Phi-3.5-mini-instruct     (~2 GB)   ║
║     --model gemma       gemma-2-2b-it             (~2 GB)   ║
║                  (Gemma needs --hf-token YOUR_HF_TOKEN)      ║
║                                                              ║
║  5. Test prompt without loading the model:                   ║
║     !python generate_logs.py \\                              ║
║         --config config_generic.json \\                      ║
║         --input  my_logs.txt --dry-run                       ║
║                                                              ║
║  Output files:                                               ║
║    synthetic_*.txt           raw log lines                   ║
║    synthetic_*_labeled.jsonl log + anomaly_type labels       ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(msg)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    install()
    print_usage()
