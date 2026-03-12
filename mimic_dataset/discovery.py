from __future__ import annotations

import os
from typing import Dict, List, Tuple

from .io_utils import FileRef, iter_table_file_refs, read_csv_schema_sample
from .types import DiscoveredTable


def _make_table_id(root_dir: str, fr: FileRef) -> str:
    root = os.path.abspath(root_dir)
    src = os.path.abspath(fr.source_path)
    rel = os.path.relpath(src, root)
    if fr.member_path:
        return f"{rel}::{fr.member_path}"
    return rel


def discover_tables(
    mimic_root: str,
    *,
    schema_sample_rows: int = 200,
) -> List[DiscoveredTable]:
    """Recursively discover table-like resources and sample schemas."""
    tables: List[DiscoveredTable] = []
    for fr in iter_table_file_refs(mimic_root):
        table_id = _make_table_id(mimic_root, fr)
        try:
            cols, dtypes = read_csv_schema_sample(fr, nrows=schema_sample_rows)
        except Exception as e:  # pragma: no cover (data-dependent)
            tables.append(
                DiscoveredTable(
                    table_id=table_id,
                    source_path=os.path.abspath(fr.source_path),
                    member_path=fr.member_path,
                    format="unknown",
                    columns=[],
                    dtypes={},
                    extra={"error": repr(e)},
                )
            )
            continue

        fmt = "zip_csv" if fr.member_path else "csv"
        tables.append(
            DiscoveredTable(
                table_id=table_id,
                source_path=os.path.abspath(fr.source_path),
                member_path=fr.member_path,
                format=fmt,
                columns=cols,
                dtypes=dtypes,
            )
        )
    return tables


def score_table_type(table: DiscoveredTable) -> Dict[str, float]:
    """Heuristic scoring of likely table types from schema only (no filenames)."""
    cols = set(table.lower_columns)

    def has(*names: str) -> bool:
        return any(n.lower() in cols for n in names)

    scores: Dict[str, float] = {}

    # Core entities
    scores["patients"] = 0.0
    if has("subject_id") and (has("gender", "sex") or has("anchor_age", "dob")):
        scores["patients"] += 2.0
    if has("anchor_age") or has("anchor_year"):
        scores["patients"] += 1.0

    scores["admissions"] = 0.0
    if has("hadm_id") and has("subject_id"):
        scores["admissions"] += 2.0
    if has("admittime", "admittime") or has("admission_type", "admission_location"):
        scores["admissions"] += 1.0
    if has("dischtime", "deathtime", "admission_type"):
        scores["admissions"] += 1.0

    # Diagnoses
    scores["diagnoses"] = 0.0
    if has("hadm_id") and has("icd_code", "diag_code"):
        scores["diagnoses"] += 2.0
    if has("icd_version") or has("seq_num"):
        scores["diagnoses"] += 1.0

    # Notes
    scores["notes"] = 0.0
    if has("text") and (has("hadm_id") or has("subject_id")):
        scores["notes"] += 2.0
    if has("category", "description"):
        scores["notes"] += 1.0
    if has("charttime", "storetime"):
        scores["notes"] += 0.5

    # Labs
    scores["labs"] = 0.0
    if (has("itemid") or has("labitemid")) and (has("valuenum") or has("value")):
        scores["labs"] += 1.5
    if has("flag", "ref_range_lower", "ref_range_upper"):
        scores["labs"] += 1.0
    if has("hadm_id") and has("charttime"):
        scores["labs"] += 1.0

    # Vitals / chart events (ICU)
    scores["vitals"] = 0.0
    if has("stay_id") and (has("itemid") or has("charttime")) and (has("valuenum") or has("value")):
        scores["vitals"] += 2.0
    if has("hadm_id") and (has("heart rate") or has("sysbp")):
        scores["vitals"] += 0.2  # unlikely; structured vitals use itemids

    # Dictionaries / dimension tables
    scores["item_dictionary"] = 0.0
    if has("itemid") and has("label"):
        scores["item_dictionary"] += 2.0
    if has("category") or has("unitname"):
        scores["item_dictionary"] += 0.5

    return scores


def pick_best_table(tables: List[DiscoveredTable], kind: str) -> Tuple[DiscoveredTable | None, List[Tuple[str, float]]]:
    scored: List[Tuple[DiscoveredTable, float]] = []
    for t in tables:
        if not t.columns:
            continue
        s = score_table_type(t).get(kind, 0.0)
        if s > 0:
            scored.append((t, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return None, []
    return scored[0][0], [(t.table_id, s) for t, s in scored[:10]]


