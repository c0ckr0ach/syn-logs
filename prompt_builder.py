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
    system_content = (
        "You are a synthetic log generation assistant.\n"
        "Your SOLE task is to output raw log lines that are indistinguishable "
        "in format and style from the provided examples.\n\n"
        "STRICT OUTPUT RULES — follow these without exception:\n"
        "  • Output ONLY raw log lines, one per line.\n"
        "  • Do NOT number the lines.\n"
        "  • Do NOT add headers, bullet points, markdown, or code fences.\n"
        "  • Do NOT explain, summarise, or add any commentary.\n"
        "  • Do NOT repeat these instructions in your response.\n"
        "  • Generate the EXACT number of lines requested — no more, no fewer."
    )

    # ── Anomaly directive (empty string for normal batches) ───────────────────
    directive = ANOMALY_DIRECTIVES.get(anomaly_type, "")
    anomaly_block = (
        f"\n{directive}\n"
        if directive
        else ""
    )

    # ── User message ──────────────────────────────────────────────────────────
    user_content = (
        f"LOG FORMAT DESCRIPTION:\n"
        f"{format_description}\n\n"
        f"EXAMPLE LOG LINES (reproduce this exact format and style):\n"
        f"{sample_text}\n"
        f"{anomaly_block}\n"
        f"Generate exactly {n_to_generate} new log lines now. "
        f"Output ONLY the log lines, one per line — no other text:"
    )

    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
    ]
