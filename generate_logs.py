"""
generate_logs.py
~~~~~~~~~~~~~~~~
Main pipeline orchestrator for synthetic log generation.

Architecture
------------
  sample_logs.txt
       │
       ▼
  config_loader.py    ← load JSON config + resolve model alias
       │
       ▼
  log_assembler.py    ← auto-detect or read log format
       │
       ▼
  llm_backend.py      ← load HF model (4-bit) + tokenizer
       │
  ┌────┴────────────────────────────────────────────┐
  │  Generation loop (until target_logs reached)    │
  │                                                 │
  │  noise.py          ← pick anomaly type          │
  │  prompt_builder.py ← build chat messages        │
  │  llm_backend.py    ← generate_text()            │
  │  log_assembler.py  ← parse_generated_output()   │
  │                                                 │
  │  Checkpoint every N lines                       │
  └────────────────────────────────────────────────┘
       │
       ▼
  synthetic_logs.txt          (plain log lines)
  synthetic_logs_labeled.jsonl  (log + anomaly_type, optional)

Usage (Colab shell / terminal)
------------------------------
  python generate_logs.py \\
      --config  config_healthapp.json \\
      --input   my_sample_logs.txt    \\
      --target  2000

  # Override model at runtime:
  python generate_logs.py --config config_generic.json \\
      --input logs.txt --model deepseek --target 500

  # Dry-run: print prompt without loading the model:
  python generate_logs.py --config config.json --input logs.txt --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ─── Project modules ─────────────────────────────────────────────────────────
from config_loader  import load_config
from log_assembler  import (
    detect_log_format,
    load_sample_logs,
    parse_generated_output,
    load_checkpoint,
    save_checkpoint,
    save_logs,
)
from prompt_builder import build_format_description, build_messages
from noise          import pick_anomaly_type, get_label


# ─── Pipeline ────────────────────────────────────────────────────────────────

def run_pipeline(
    config_path:  str,
    input_file:   str,
    model_id:     str | None = None,
    output_file:  str | None = None,
    target:       int | None = None,
    batch_size:   int | None = None,
    hf_token:     str | None = None,
    dry_run:      bool       = False,
    load_in_4bit: bool       = True,
) -> list[str]:
    """
    Run the full synthetic log generation pipeline.

    Parameters
    ----------
    config_path : str
        Path to a dataset config JSON.
    input_file : str
        Path to the sample log file (500–2000 lines).
    model_id : str | None
        Override the model from config (alias or full HF model ID).
    output_file : str | None
        Override the output file path from config.
    target : int | None
        Override the target log count from config.
    batch_size : int | None
        Override the batch size from config.
    hf_token : str | None
        HuggingFace API token (required for gated models like Gemma).
    dry_run : bool
        If True, print the first batch prompt and exit without loading the model.
    load_in_4bit : bool
        Enable 4-bit NF4 quantization. Disable on CPU-only environments.

    Returns
    -------
    list[str]
        The generated log lines (up to *target* entries).
    """
    # ── 1. Load config ────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    gen = cfg.generation

    # CLI overrides take priority over config values
    resolved_model  = model_id     or cfg.model_id
    resolved_output = output_file  or gen.output_file
    target_logs     = target       or gen.target_logs
    b_size          = batch_size   or gen.batch_size
    anomaly_rate    = gen.anomaly_rate
    window_size     = gen.sample_window_size
    chk_file        = gen.checkpoint_file
    chk_every       = gen.checkpoint_every_n
    write_labels    = getattr(gen, "write_labels", True)

    _print_banner(cfg.dataset_name, resolved_model, input_file, target_logs, b_size, anomaly_rate)

    # ── 2. Load sample logs + detect format ───────────────────────────────────
    sample_lines = load_sample_logs(input_file)

    log_fmt = None
    if hasattr(cfg, "log_format"):
        lf = cfg.log_format
        if getattr(lf, "auto_detect", False):
            log_fmt = detect_log_format(sample_lines)
        else:
            # Use config-specified format but enrich with component samples
            log_fmt = lf
            print(f"[Format] Using config-specified format: {getattr(lf, 'log_type', 'custom')}")
    else:
        log_fmt = detect_log_format(sample_lines)

    fmt_description = build_format_description(log_fmt)
    expected_sep    = getattr(log_fmt, "separator", "") if log_fmt else ""
    # Don't filter on space separator (syslog / generic) — too restrictive
    filter_sep = expected_sep if expected_sep not in ("", " ") else ""

    # ── 3. Dry-run: print prompt and exit ─────────────────────────────────────
    if dry_run:
        _dry_run_preview(sample_lines, fmt_description, b_size, window_size)
        return []

    # ── 4. Load model ─────────────────────────────────────────────────────────
    from llm_backend import build_model, generate_text  # deferred import

    model, tokenizer = build_model(
        model_id=resolved_model,
        load_in_4bit=load_in_4bit,
        hf_token=hf_token,
    )

    # ── 5. Resume from checkpoint ─────────────────────────────────────────────
    all_logs:     list[str]  = load_checkpoint(chk_file)
    label_data:   list[dict] = []   # for _labeled.jsonl

    # ── 6. Generation loop ────────────────────────────────────────────────────
    consecutive_failures = 0
    batch_idx            = 0

    print(f"\n[Pipeline] Starting — target: {target_logs}  |  done so far: {len(all_logs)}\n")

    while len(all_logs) < target_logs:
        remaining    = target_logs - len(all_logs)
        this_batch   = min(b_size, remaining)
        anomaly_type = pick_anomaly_type(anomaly_rate)

        messages = build_messages(
            sample_lines       = sample_lines,
            format_description = fmt_description,
            n_to_generate      = this_batch,
            anomaly_type       = anomaly_type,
            window_size        = window_size,
            batch_idx          = batch_idx,
        )

        try:
            raw_output = generate_text(
                model          = model,
                tokenizer      = tokenizer,
                messages       = messages,
                max_new_tokens = gen.max_new_tokens,
                temperature    = gen.temperature,
            )
        except Exception as exc:
            consecutive_failures += 1
            print(f"[Warning] Generation failed ({consecutive_failures}/3): {exc}")
            if consecutive_failures >= 3:
                print("[Error] 3 consecutive failures — stopping early.")
                break
            time.sleep(2)
            batch_idx += 1
            continue

        parsed = parse_generated_output(raw_output, expected_sep=filter_sep)

        if not parsed:
            consecutive_failures += 1
            print(f"[Warning] No valid lines parsed (failure {consecutive_failures}/3). "
                  "Skipping batch.")
            if consecutive_failures >= 3:
                print("[Error] 3 consecutive empty batches — stopping early.")
                break
            batch_idx += 1
            continue

        consecutive_failures = 0
        added = 0

        for line in parsed:
            if len(all_logs) >= target_logs:
                break
            all_logs.append(line)
            if write_labels:
                label_data.append({
                    "log":          line,
                    "anomaly_type": anomaly_type,
                    "anomaly_label": get_label(anomaly_type),
                    "batch_idx":    batch_idx,
                })
            added += 1

        # Progress bar
        pct    = min(100, int(len(all_logs) / target_logs * 100))
        a_flag = f"[{anomaly_type[:12]:<12}]" if anomaly_type != "none" else "[normal      ]"
        print(f"[Progress] {a_flag}  {len(all_logs):>5}/{target_logs}  ({pct:>3}%)  "
              f"+{added} lines  (batch {batch_idx})")

        # Checkpoint
        if len(all_logs) % chk_every < b_size:
            save_checkpoint(all_logs, chk_file)

        batch_idx += 1

    # ── 7. Save final output ─────────────────────────────────────────────────
    final = all_logs[:target_logs]
    save_logs(final, resolved_output)

    if write_labels and label_data:
        label_path = resolved_output.replace(".txt", "_labeled.jsonl")
        Path(label_path).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in label_data[:target_logs]),
            encoding="utf-8",
        )
        print(f"[Save]   Labels  → {label_path}")

    # Clean up checkpoint after successful completion
    if Path(chk_file).exists():
        Path(chk_file).unlink()
        print("[Checkpoint] Removed (run complete).")

    print(f"\n{'='*60}")
    print(f"  ✅  Done!  {len(final)} synthetic log lines generated.")
    print(f"      Output : {resolved_output}")
    print(f"{'='*60}\n")

    return final


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _print_banner(
    dataset:      str,
    model:        str,
    input_file:   str,
    target_logs:  int,
    batch_size:   int,
    anomaly_rate: float,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Synthetic Log Generator")
    print(f"{'─'*60}")
    print(f"  Dataset      : {dataset}")
    print(f"  Model        : {model}")
    print(f"  Input file   : {input_file}")
    print(f"  Target logs  : {target_logs}  |  Batch size: {batch_size}")
    print(f"  Anomaly rate : {anomaly_rate:.0%}")
    print(f"{'='*60}\n")


def _dry_run_preview(
    sample_lines:    list[str],
    fmt_description: str,
    batch_size:      int,
    window_size:     int,
) -> None:
    """Print the first batch prompt and exit."""
    from prompt_builder import build_messages
    messages = build_messages(
        sample_lines       = sample_lines,
        format_description = fmt_description,
        n_to_generate      = batch_size,
        anomaly_type       = "none",
        window_size        = window_size,
        batch_idx          = 0,
    )
    print("\n" + "=" * 60)
    print("DRY-RUN — First batch prompt")
    print("=" * 60)
    for msg in messages:
        print(f"\n[{msg['role'].upper()}]\n{msg['content']}")
    print("\n" + "=" * 60)
    print("(Model not loaded. Remove --dry-run to run the full pipeline.)")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Synthetic log generator using open-source HuggingFace LLMs.\n"
            "Generates realistic log data from a provided sample log file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to dataset config JSON  (e.g. config_healthapp.json)",
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Path to sample log file  (500–2000 lines recommended)",
    )
    p.add_argument(
        "--model", "-m", default=None,
        help=(
            "Model alias or full HF model ID.  Overrides the model in config.\n"
            "Aliases: qwen | qwen7b | deepseek | deepseek1 | gemma | phi | mistral"
        ),
    )
    p.add_argument(
        "--target", "-n", type=int, default=None,
        help="Number of synthetic log lines to generate  (overrides config)",
    )
    p.add_argument(
        "--batch-size", "-b", type=int, default=None,
        help="Lines per LLM call  (overrides config)",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output file path  (overrides config)",
    )
    p.add_argument(
        "--hf-token", default=None,
        help="HuggingFace API token (needed for gated models such as Gemma)",
    )
    p.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization (use fp16; needs more VRAM)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the first batch prompt and exit without loading the model",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        config_path  = args.config,
        input_file   = args.input,
        model_id     = args.model,
        output_file  = args.output,
        target       = args.target,
        batch_size   = args.batch_size,
        hf_token     = args.hf_token,
        dry_run      = args.dry_run,
        load_in_4bit = not args.no_4bit,
    )
