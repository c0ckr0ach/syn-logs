"""
noise.py
~~~~~~~~
Anomaly type definitions and selector for synthetic log generation.

Phase 1 strategy: anomalies are injected through LLM prompt directives
rather than programmatic text manipulation. The LLM is asked to generate
one or more anomalous lines within an otherwise normal batch, making the
anomalies semantically realistic rather than superficially garbled.

Anomaly types are designed to be useful for training anomaly-detection ML
models (e.g., classifiers, autoencoders, or isolation forests).
"""

from __future__ import annotations

import random

# ─── Anomaly type registry ────────────────────────────────────────────────────

# Maps an anomaly key → short prompt directive that is injected into the LLM
# user message. The LLM is instructed to include these anomalous lines
# naturally within the batch so they blend with the surrounding normal logs.

ANOMALY_DIRECTIVES: dict[str, str] = {
    # ── Default: no anomaly ──────────────────────────────────────────────────
    "none": "",

    # ── Structural / operational anomalies ───────────────────────────────────
    "error_cascade": (
        "ANOMALY DIRECTIVE: Include 2–3 consecutive lines depicting a "
        "service error or crash sequence (e.g., initialisation failure → "
        "exception thrown → service stopped / restarted)."
    ),
    "repeated_event": (
        "ANOMALY DIRECTIVE: Include 3–5 nearly identical repeated log lines "
        "suggesting an infinite loop, retry storm, or runaway log flood."
    ),
    "unexpected_shutdown": (
        "ANOMALY DIRECTIVE: Include 1–2 lines indicating an unexpected "
        "process termination, panic, or unplanned service restart."
    ),
    "high_latency": (
        "ANOMALY DIRECTIVE: Include 1–2 lines indicating an unusually slow "
        "operation, response-time SLA breach, or deadline-exceeded warning."
    ),

    # ── Resource anomalies ───────────────────────────────────────────────────
    "resource_exhaustion": (
        "ANOMALY DIRECTIVE: Include 1–2 lines indicating a resource limit "
        "being hit — e.g., out of memory, disk full, thread pool exhausted, "
        "or file-descriptor limit reached."
    ),
    "network_anomaly": (
        "ANOMALY DIRECTIVE: Include 1–2 lines indicating a network-level "
        "problem: connection refused, TCP timeout, DNS resolution failure, "
        "or packet loss warning."
    ),

    # ── Security / auth anomalies ────────────────────────────────────────────
    "auth_failure": (
        "ANOMALY DIRECTIVE: Include 1–2 lines indicating repeated "
        "authentication failures, access-denied responses, or a "
        "privilege-escalation attempt."
    ),
    "suspicious_access": (
        "ANOMALY DIRECTIVE: Include 1–2 lines showing access to an "
        "unexpected resource path, an unusual source IP, or an "
        "off-hours login event."
    ),

    # ── Data / config anomalies ──────────────────────────────────────────────
    "config_error": (
        "ANOMALY DIRECTIVE: Include 1–2 lines indicating a configuration "
        "parse error, a missing required setting, or an invalid parameter "
        "value being used."
    ),
    "data_corruption": (
        "ANOMALY DIRECTIVE: Include 1–2 lines indicating data integrity "
        "issues — checksum mismatch, corrupted record, or unexpected null "
        "value in a critical field."
    ),

    # ── Source / format anomalies ────────────────────────────────────────────
    "wrong_component": (
        "ANOMALY DIRECTIVE: Include 1–2 lines where the source component "
        "or process name is unexpected or mismatched with the operation "
        "being logged (e.g., a database query logged under a UI component)."
    ),
}

# Human-readable labels (for JSONL output and reporting)
ANOMALY_LABELS: dict[str, str] = {
    "none":                "Normal",
    "error_cascade":       "Error Cascade",
    "repeated_event":      "Repeated Event",
    "unexpected_shutdown": "Unexpected Shutdown",
    "high_latency":        "High Latency",
    "resource_exhaustion": "Resource Exhaustion",
    "network_anomaly":     "Network Anomaly",
    "auth_failure":        "Auth Failure",
    "suspicious_access":   "Suspicious Access",
    "config_error":        "Config Error",
    "data_corruption":     "Data Corruption",
    "wrong_component":     "Wrong Component",
}

# All available anomaly keys (excluding "none")
ANOMALY_TYPES: list[str] = [k for k in ANOMALY_DIRECTIVES if k != "none"]


# ─── Selector ─────────────────────────────────────────────────────────────────

def pick_anomaly_type(anomaly_rate: float) -> str:
    """
    Randomly pick an anomaly type for the current batch.

    Parameters
    ----------
    anomaly_rate : float
        Probability (0–1) that this batch will contain an anomaly.
        E.g. 0.15 → ~15 % of batches will be anomalous.

    Returns
    -------
    str
        An anomaly key from ANOMALY_DIRECTIVES.
        Returns ``"none"`` when no anomaly is injected.
    """
    if random.random() < anomaly_rate:
        return random.choice(ANOMALY_TYPES)
    return "none"


def get_directive(anomaly_type: str) -> str:
    """Return the prompt directive string for *anomaly_type*."""
    return ANOMALY_DIRECTIVES.get(anomaly_type, "")


def get_label(anomaly_type: str) -> str:
    """Return the human-readable label for *anomaly_type*."""
    return ANOMALY_LABELS.get(anomaly_type, anomaly_type)
