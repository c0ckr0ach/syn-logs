"""
prompt_builder.py
~~~~~~~~~~~~~~~~~
Constructs the chat-format messages sent to the LLM for each generation batch.

Prompt anatomy
--------------
  [system]  Role definition + strict output rules
  [user]    ① LOG FORMAT section  — human-readable format description
            ② EXAMPLE LINES      — rotating window of sample log lines
            ③ ANOMALY DIRECTIVE  — injected only when anomaly_type != "none"
            ④ TASK instruction   — "generate exactly N lines now"

Separating prompt construction into this module makes it trivial to tweak
wording, add few-shot examples, or experiment with different instruction
styles without touching the pipeline orchestrator.
"""

from __future__ import annotations

from types import SimpleNamespace

from noise import ANOMALY_DIRECTIVES


# ─── Format description ───────────────────────────────────────────────────────

def build_format_description(fmt: SimpleNamespace | None) -> str:
    """
    Convert a log_format config object (from JSON or auto-detected) into a
    concise, human-readable description for inclusion in the LLM prompt.

    Parameters
    ----------
    fmt : SimpleNamespace | None
        The ``cfg.log_format`` object (or None for fully unstructured logs).

    Returns
    -------
    str
        Bullet-list string describing the log format.
    """
    if fmt is None:
        return "- Match the format of the example lines exactly — same fields, same delimiter, same timestamp style."

    parts: list[str] = []

    # Log type / family
    if getattr(fmt, "log_type", ""):
        parts.append(f"Log type: {fmt.log_type}")

    # Field delimiter
    sep = getattr(fmt, "separator", None)
    if sep:
        sep_names = {"|": "pipe (|)", "\t": "tab (\\t)", ",": "comma (,)", ";": "semicolon (;)"}
        parts.append(f"Field delimiter: {sep_names.get(sep, repr(sep))}")

    # Field names in order
    fields = getattr(fmt, "fields", None)
    if fields:
        parts.append(f"Fields in order: {' | '.join(fields)}")

    # Timestamp format
    ts = getattr(fmt, "timestamp_pattern", None)
    if ts:
        parts.append(f"Timestamp format: {ts}")

    # Known components / sources
    components = getattr(fmt, "components", [])
    if components:
        shown = components[:8]
        extra = f"  (+{len(components) - 8} more)" if len(components) > 8 else ""
        parts.append(f"Known sources/components: {', '.join(shown)}{extra}")

    # Free-text notes from the config
    notes = getattr(fmt, "notes", "")
    if notes:
        parts.append(f"Notes: {notes}")

    # Example line
    ex = getattr(fmt, "example_line", "")
    if ex:
        parts.append(f"Example line:\n    {ex}")

    if not parts:
        return "- Match the format of the example lines exactly."

    return "\n".join(f"- {p}" for p in parts)


# ─── Message builder ──────────────────────────────────────────────────────────

def build_messages(
    sample_lines:        list[str],
    format_description:  str,
    n_to_generate:       int,
    anomaly_type:        str = "none",
    window_size:         int = 30,
    batch_idx:           int = 0,
) -> list[dict]:
    """
    Build the ``messages`` list (OpenAI-style chat format) for one generation call.

    Parameters
    ----------
    sample_lines : list[str]
        All non-empty lines from the input log file.
    format_description : str
        Human-readable format description (from :func:`build_format_description`).
    n_to_generate : int
        Exact number of log lines to request from the LLM.
    anomaly_type : str
        Key from ``noise.ANOMALY_DIRECTIVES``.  ``"none"`` → normal batch.
    window_size : int
        Number of sample lines shown to the LLM per prompt.
    batch_idx : int
        Monotonically increasing batch counter; used to rotate the sample
        window so successive batches see different reference lines.

    Returns
    -------
    list[dict]
        ``[{"role": "system", "content": ...}, {"role": "user", "content": ...}]``
    """
    # ── Rotate sample window ──────────────────────────────────────────────────
    n      = len(sample_lines)
    stride = max(1, window_size // 2)
    start  = (batch_idx * stride) % max(1, n - window_size)
    window = sample_lines[start : start + window_size]
    # Guard: if we're near the end of the file, wrap around
    if len(window) < min(window_size, n):
        window = sample_lines[-min(window_size, n):]
    sample_text = "\n".join(window)

    # ── System message ────────────────────────────────────────────────────────
    # Role + context framing (Anthropic / OpenAI best practice: tell the model
    # *who* it is, *what* it produces, and *why* precision matters — not just
    # a list of prohibitions).
    system_content = (
        "You are a log-data generation engine embedded in an ML training pipeline.\n"
        "Your output is consumed directly by a parser — no human reads it first.\n\n"
        "Produce raw log lines that are byte-for-byte compatible with the format "
        "shown in the examples: same fields, same delimiter, same timestamp style, "
        "same vocabulary. Every line must be independently parseable.\n\n"
        "Output rules:\n"
        "  • One log line per output line, nothing else.\n"
        "  • No numbering, bullets, markdown, code fences, or commentary.\n"
        "  • Vary timestamps, hostnames, PIDs, and messages — do not repeat values.\n"
        "  • Generate the exact count requested — stopping early breaks the pipeline."
    )

    # ── Anomaly directive (empty string for normal batches) ───────────────────
    directive = ANOMALY_DIRECTIVES.get(anomaly_type, "")
    anomaly_block = (
        f"\n<anomaly_directive>\n{directive}\n"
        "Weave the anomalous lines naturally into the batch at varied positions. "
        "They must remain syntactically valid log lines — the anomaly is semantic, "
        "not structural.\n</anomaly_directive>\n"
        if directive
        else ""
    )

    # ── User message ──────────────────────────────────────────────────────────
    # Structure: XML delimiters separate data from instructions so the model
    # can't confuse reference lines with its own output (OpenAI tactic §6).
    # The final line primes the model directly into log-output mode.
    user_content = (
        "<format>\n"
        f"{format_description}\n"
        "</format>\n\n"
        "<examples>\n"
        f"{sample_text}\n"
        "</examples>\n"
        f"{anomaly_block}\n"
        f"Write exactly {n_to_generate} new log lines. "
        f"Follow the format above precisely.\n"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
