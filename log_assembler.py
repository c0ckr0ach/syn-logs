"""
log_assembler.py
~~~~~~~~~~~~~~~~
Two responsibilities:
  1. Auto-detect log format from a sample of raw log lines.
  2. Parse and validate the raw text output from the LLM into clean log lines.

Detection supports:
  - Pipe-separated (Android HealthApp, custom apps)
  - Tab-separated
  - Comma-separated (CSV logs)
  - RFC-3164 Syslog  (e.g. Linux /var/log/syslog)
  - Windows Event Log (space/pipe hybrid)
  - Apache / Nginx access & error logs
  - Generic (best-effort, space-delimited)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

# ─── Known timestamp patterns ─────────────────────────────────────────────────
# Each entry: (compiled_regex, human-readable description)

_TS_PATTERNS: list[tuple] = [
    (
        re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?"),
        "ISO 8601  (YYYY-MM-DD HH:MM:SS[.fff])",
    ),
    (
        re.compile(r"\d{8}-\d{2}:\d{2}:\d{2}:\d+"),
        "HealthApp (YYYYMMDD-HH:MM:SS:ms)",
    ),
    (
        re.compile(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"),
        "Syslog RFC 3164  (Mon DD HH:MM:SS)",
    ),
    (
        re.compile(r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}\s[+-]\d{4}"),
        "Apache combined  (DD/Mon/YYYY:HH:MM:SS ±zone)",
    ),
    (
        re.compile(r"\d{2}/\d{2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*(?:AM|PM)?"),
        "Windows  (MM/DD/YYYY HH:MM:SS AM/PM)",
    ),
    (
        re.compile(r"\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}"),
        "Apache error log  (YYYY/MM/DD HH:MM:SS)",
    ),
    (
        re.compile(r"\b\d{13}\b"),
        "Unix epoch milliseconds",
    ),
    (
        re.compile(r"\b\d{10}\b"),
        "Unix epoch seconds",
    ),
]

# Lines that start with these patterns are LLM meta-commentary, not log lines
_META_RE = re.compile(
    r"^(?:"
    r"here are|sure[,!]|certainly|below|these are|generated|"
    r"log lines?:|output:|```|#{1,6}\s|^\*{1,2}[^*]|^-{3,}|"
    r"\d+\.\s|note:|warning:|the following"
    r")",
    re.IGNORECASE,
)


# ─── Public: detect format ───────────────────────────────────────────────────

def detect_log_format(sample_lines: list[str]) -> SimpleNamespace:
    """
    Infer log format characteristics from raw sample log lines.

    Parameters
    ----------
    sample_lines : list[str]
        Lines read from the input file (500–2000 expected).
        Uses only the first 200 non-empty lines for speed.

    Returns
    -------
    SimpleNamespace
        Attributes:
          separator         (str)       best-guess field delimiter
          fields            (list[str]) inferred field names
          field_count       (int)       expected number of fields per line
          timestamp_pattern (str)       human-readable timestamp description
          log_type          (str)       inferred log family name
          components        (list[str]) up to 15 sampled component/source names
          example_line      (str)       a representative raw sample line
          auto_detect       (bool)      always True when returned from here
    """
    clean = [l for l in sample_lines if l.strip()][:200]
    if not clean:
        return _fallback_format()

    sep, field_count  = _detect_separator(clean)
    ts_desc, log_type = _detect_timestamp(clean, sep)
    fields            = _infer_field_names(clean, sep, field_count, ts_desc)
    components        = _sample_components(clean, sep, fields)

    fmt = SimpleNamespace(
        separator         = sep,
        fields            = fields,
        field_count       = field_count,
        timestamp_pattern = ts_desc,
        log_type          = log_type,
        components        = components,
        example_line      = clean[0],
        auto_detect       = True,
    )

    print(f"[Format] Detected log type  : {log_type}")
    print(f"         Separator          : {repr(sep)}")
    print(f"         Fields ({field_count})         : {' | '.join(fields)}")
    print(f"         Timestamp pattern  : {ts_desc}")
    if components:
        print(f"         Sample components  : {', '.join(components[:5])}" +
              (f"  (+{len(components)-5} more)" if len(components) > 5 else ""))
    return fmt


# ─── Public: parse LLM output ────────────────────────────────────────────────

def parse_generated_output(
    raw_text:     str,
    expected_sep: str = "",
    min_length:   int = 5,
) -> list[str]:
    """
    Parse raw LLM-generated text into a list of clean log lines.

    Discards:
    - Empty / whitespace-only lines
    - Lines shorter than *min_length*
    - Obvious LLM meta-commentary (headers, bullet points, ``` fences, etc.)
    - Lines that don't contain the expected separator (if *expected_sep* is set)

    Parameters
    ----------
    raw_text : str
        Full text returned by the LLM.
    expected_sep : str
        If non-empty, lines without this separator are dropped.
        Set to ``""`` to skip that check (e.g. for space-delimited logs).
    min_length : int
        Minimum character length for a valid log line.

    Returns
    -------
    list[str]
        Validated, cleaned log lines (order preserved).
    """
    cleaned: list[str] = []

    for line in raw_text.splitlines():
        line = line.strip()

        if not line or len(line) < min_length:
            continue
        if _META_RE.match(line):
            continue
        if expected_sep and expected_sep not in line:
            continue

        cleaned.append(line)

    return cleaned


# ─── Public: file I/O ────────────────────────────────────────────────────────

def load_sample_logs(filepath: str) -> list[str]:
    """Read and return all non-empty lines from *filepath*."""
    text  = Path(filepath).read_text(encoding="utf-8", errors="replace")
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    print(f"[Loader] {len(lines)} sample lines ← {filepath}")
    return lines


def save_logs(logs: list[str], filepath: str) -> None:
    """Write *logs* to *filepath*, one line per entry."""
    Path(filepath).write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(f"[Save]   {len(logs)} lines → {filepath}")


# ─── Checkpoint helpers ──────────────────────────────────────────────────────

def load_checkpoint(path: str) -> list[str]:
    """Return previously generated lines from a checkpoint file (or [])."""
    p = Path(path)
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"[Checkpoint] Resumed: {len(lines)} lines from {path}")
        return lines
    return []


def save_checkpoint(logs: list[str], path: str) -> None:
    """Write current logs to the checkpoint file."""
    Path(path).write_text("\n".join(logs), encoding="utf-8")
    print(f"[Checkpoint] Saved {len(logs)} lines → {path}")


# ─── Internal helpers ────────────────────────────────────────────────────────

def _detect_separator(lines: list[str]) -> tuple[str, int]:
    """
    Return (best_separator, expected_field_count).
    Candidates: ``|``, ``\\t``, ``,``, ``;``.
    Falls back to space-delimited with field_count=1 if nothing fits.
    """
    candidates = ["|", "\t", ",", ";"]
    best_sep   = " "
    best_fc    = 1
    best_score = 0.0

    for sep in candidates:
        counts = [line.count(sep) for line in lines if line.strip()]
        if not counts or max(counts) == 0:
            continue
        mode_sep_count = Counter(counts).most_common(1)[0][0]
        if mode_sep_count == 0:
            continue
        consistency = counts.count(mode_sep_count) / len(counts)
        # Score: high consistency AND multiple fields
        score = consistency * min(mode_sep_count + 1, 8)
        if score > best_score:
            best_score = score
            best_sep   = sep
            best_fc    = mode_sep_count + 1

    return best_sep, best_fc


def _detect_timestamp(lines: list[str], sep: str) -> tuple[str, str]:
    """Return (timestamp_description, log_type_label)."""
    # Check first field (before separator) for timestamp
    first_fields = []
    for line in lines[:60]:
        if sep and sep != " ":
            first_fields.append(line.split(sep, 1)[0].strip())
        else:
            first_fields.append(line[:30].strip())

    for pat, label in _TS_PATTERNS:
        hit_rate = sum(1 for f in first_fields if pat.search(f)) / max(1, len(first_fields))
        if hit_rate > 0.55:
            return label, _infer_log_type(label, lines)

    return "Unknown timestamp format", "Generic"


def _infer_log_type(ts_label: str, lines: list[str]) -> str:
    sample = " ".join(lines[:30]).lower()
    tsl    = ts_label.lower()

    if "healthapp" in tsl:
        return "Android HealthApp"
    if "rfc 3164" in tsl or "syslog" in tsl:
        if any(k in sample for k in ["kernel", "systemd", "sshd", "sudo", "cron"]):
            return "Linux Syslog"
        return "Syslog"
    if "windows" in tsl:
        return "Windows Event Log"
    if "apache combined" in tsl:
        return "Nginx Log" if "nginx" in sample else "Apache Access Log"
    if "apache error" in tsl:
        return "Apache Error Log"
    if "iso 8601" in tsl:
        if any(k in sample for k in ["error", "warn", "info", "debug", "trace", "fatal"]):
            return "Application Log (structured)"
        return "Generic ISO8601 Log"
    return "Generic"


def _infer_field_names(
    lines:       list[str],
    sep:         str,
    field_count: int,
    ts_desc:     str,
) -> list[str]:
    """Heuristically name each field based on content analysis."""
    if field_count <= 1:
        return ["message"]

    sample = []
    for line in lines[:30]:
        parts = line.split(sep, field_count - 1) if sep != " " else line.split(None, field_count - 1)
        if len(parts) == field_count:
            sample.append(parts)

    if not sample:
        return [f"field{i}" for i in range(field_count)]

    fields: list[str] = []
    used: set[str] = set()

    for i in range(field_count):
        col = [row[i].strip() for row in sample if i < len(row)]
        name = _name_column(i, col, field_count, used)
        fields.append(name)
        used.add(name)

    return fields


def _name_column(idx: int, col: list[str], total: int, used: set) -> str:
    """Name a single column by position + content heuristics."""
    if not col:
        return f"field{idx}"

    # First field is almost always timestamp
    if idx == 0:
        return "timestamp"

    # Last field is almost always the message body
    if idx == total - 1:
        return "message"

    non_empty = [c for c in col if c]

    # All digits → PID / numeric ID
    if all(c.isdigit() for c in non_empty[:10]) and "pid" not in used:
        return "pid"

    # Short tokens with no spaces → component / logger / process
    if all(len(c) < 45 and " " not in c for c in non_empty[:10]):
        if "component" not in used:
            return "component"
        if "source" not in used:
            return "source"

    # Level-like values
    level_values = {"error", "warn", "warning", "info", "debug", "trace", "fatal", "critical"}
    if any(c.lower() in level_values for c in non_empty[:10]) and "level" not in used:
        return "level"

    return f"field{idx}"


def _sample_components(
    lines:  list[str],
    sep:    str,
    fields: list[str],
) -> list[str]:
    """Extract unique component/source names (up to 15)."""
    target_indices = [
        i for i, f in enumerate(fields)
        if f in ("component", "source", "logger")
    ]
    if not target_indices:
        return []

    idx = target_indices[0]
    seen: set[str] = set()

    for line in lines[:150]:
        parts = line.split(sep) if sep != " " else line.split()
        if idx < len(parts):
            comp = parts[idx].strip()
            if comp and len(comp) < 60 and comp not in seen:
                seen.add(comp)
                if len(seen) >= 15:
                    break

    return sorted(seen)


def _fallback_format() -> SimpleNamespace:
    return SimpleNamespace(
        separator         = " ",
        fields            = ["message"],
        field_count       = 1,
        timestamp_pattern = "Unknown",
        log_type          = "Generic",
        components        = [],
        example_line      = "",
        auto_detect       = True,
    )
