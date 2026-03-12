from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def extract_chief_complaint(note_text: str) -> List[str]:
    """
    Best-effort symptom/chief-complaint extraction from note text.
    Works across variable note formats; returns a short list of snippets.
    """
    if not note_text:
        return []

    t = note_text
    patterns: List[Tuple[str, int]] = [
        (r"chief\s+complaint\s*:\s*(.+)", re.IGNORECASE),
        (r"cc\s*:\s*(.+)", re.IGNORECASE),
        (r"presenting\s+complaint\s*:\s*(.+)", re.IGNORECASE),
        (r"history\s+of\s+present\s+illness\s*:\s*(.+)", re.IGNORECASE),
    ]

    hits: List[str] = []
    for pat, flags in patterns:
        m = re.search(pat, t, flags)
        if not m:
            continue
        snippet = m.group(1)
        snippet = snippet.split("\n", 1)[0]
        snippet = normalize_ws(snippet)
        if snippet and snippet.lower() not in {h.lower() for h in hits}:
            hits.append(snippet[:240])
        if len(hits) >= 5:
            break

    return hits


def simple_note_summary(note_texts: Iterable[str], max_chars: int = 2000) -> str:
    parts: List[str] = []
    total = 0
    for t in note_texts:
        nt = normalize_ws(t or "")
        if not nt:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        if len(nt) > remaining:
            nt = nt[:remaining]
        parts.append(nt)
        total += len(nt) + 2
    return "\n\n".join(parts)


def render_case_text(
    *,
    age: Optional[float],
    sex: Optional[str],
    symptoms: List[str],
    vitals: Dict[str, str],
    labs: Dict[str, str],
    note_summary: str,
) -> str:
    age_str = "unknown" if age is None else f"{age:.1f}"
    sex_str = (sex or "unknown").strip()
    symptoms_str = ", ".join(symptoms) if symptoms else "unknown"
    vitals_str = "; ".join([f"{k}: {v}" for k, v in vitals.items()]) if vitals else "none"
    labs_str = "; ".join([f"{k}: {v}" for k, v in labs.items()]) if labs else "none"
    note_str = note_summary.strip() if note_summary else "none"

    return (
        "Patient:\n"
        f"Age: {age_str}\n"
        f"Sex: {sex_str}\n\n"
        "Symptoms:\n"
        f"{symptoms_str}\n\n"
        "Vital Signs:\n"
        f"{vitals_str}\n\n"
        "Lab Findings:\n"
        f"{labs_str}\n\n"
        "Clinical Notes:\n"
        f"{note_str}\n"
    )


