from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd
from tqdm import tqdm

from .discovery import pick_best_table
from .io_utils import FileRef, iter_csv_chunks
from .text_utils import extract_chief_complaint, render_case_text, simple_note_summary
from .types import DiscoveredTable, RelationshipGraph


@dataclass(frozen=True)
class BuildConfig:
    mimic_root: str
    out_dir: str
    cache_dir: Optional[str] = None
    max_cases: Optional[int] = None
    seed: int = 7
    chunksize: int = 200_000
    min_note_chars: int = 40


def _table_to_fileref(t: DiscoveredTable) -> FileRef:
    return FileRef(source_path=t.source_path, member_path=t.member_path)


def _safe_parse_datetime(x: Any) -> Optional[datetime]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return pd.to_datetime(x, errors="coerce").to_pydatetime()
    except Exception:
        return None


def _infer_age(
    patient_row: Dict[str, Any],
    admission_row: Dict[str, Any],
) -> Optional[float]:
    """
    MIMIC-IV provides de-identified age via anchor_age/anchor_year.
    If those exist, estimate age at admission year.
    """
    keys = {k.lower(): k for k in patient_row.keys()}
    akeys = {k.lower(): k for k in admission_row.keys()}

    # anchor-based (preferred)
    if "anchor_age" in keys and "anchor_year" in keys:
        anchor_age = patient_row.get(keys["anchor_age"])
        anchor_year = patient_row.get(keys["anchor_year"])
        admit_time = admission_row.get(akeys.get("admittime", "admittime"))
        dt = _safe_parse_datetime(admit_time)
        if dt is None:
            return float(anchor_age) if anchor_age is not None and not pd.isna(anchor_age) else None
        try:
            return float(anchor_age) + (dt.year - int(anchor_year))
        except Exception:
            return None

    # fallback: explicit age column
    if "age" in keys:
        try:
            return float(patient_row.get(keys["age"]))
        except Exception:
            return None

    return None


def _infer_sex(patient_row: Dict[str, Any]) -> Optional[str]:
    for k in ("gender", "sex"):
        if k in {c.lower() for c in patient_row.keys()}:
            # find original key
            for orig in patient_row.keys():
                if orig.lower() == k:
                    v = patient_row.get(orig)
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        return None
                    return str(v)
    return None


def _load_patients_map(
    patients_table: DiscoveredTable,
    *,
    chunksize: int,
) -> Dict[int, Dict[str, Any]]:
    fr = _table_to_fileref(patients_table)
    needed = [c for c in patients_table.columns if c.lower() in {"subject_id", "gender", "sex", "anchor_age", "anchor_year", "age"}]
    subj_col = next((c for c in needed if c.lower() == "subject_id"), None)
    if subj_col is None:
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    for chunk in iter_csv_chunks(fr, usecols=needed, chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            sid = row.get(subj_col)
            if sid is None or (isinstance(sid, float) and pd.isna(sid)):
                continue
            try:
                out[int(sid)] = row
            except Exception:
                continue
    return out


def _load_admissions(
    admissions_table: DiscoveredTable,
    *,
    chunksize: int,
    max_cases: Optional[int],
) -> List[Dict[str, Any]]:
    fr = _table_to_fileref(admissions_table)
    needed = [c for c in admissions_table.columns if c.lower() in {"hadm_id", "subject_id", "admittime", "dischtime", "deathtime"}]
    hadm_col = next((c for c in needed if c.lower() == "hadm_id"), None)
    if hadm_col is None:
        return []

    rows: List[Dict[str, Any]] = []
    for chunk in iter_csv_chunks(fr, usecols=needed, chunksize=chunksize):
        rows.extend(chunk.to_dict(orient="records"))
        if max_cases is not None and len(rows) >= max_cases:
            rows = rows[:max_cases]
            break
    # ensure stable order by hadm_id if possible
    try:
        rows.sort(key=lambda r: int(r.get(hadm_col)))
    except Exception:
        pass
    return rows


def _build_code_map(
    dict_table: Optional[DiscoveredTable],
    *,
    code_col_candidates: Sequence[str],
    title_col_candidates: Sequence[str],
    chunksize: int,
) -> Dict[Tuple[str, str], str]:
    """
    Return map keyed by (code, version?) or (code, "") to a title.
    Designed to work for ICD dictionaries without assuming filenames.
    """
    if dict_table is None or not dict_table.columns:
        return {}

    cols_lower = {c.lower(): c for c in dict_table.columns}
    code_col = next((cols_lower.get(c.lower()) for c in code_col_candidates if c.lower() in cols_lower), None)
    title_col = next((cols_lower.get(c.lower()) for c in title_col_candidates if c.lower() in cols_lower), None)
    version_col = cols_lower.get("icd_version")
    if code_col is None or title_col is None:
        return {}

    fr = _table_to_fileref(dict_table)
    usecols = [code_col, title_col] + ([version_col] if version_col else [])
    out: Dict[Tuple[str, str], str] = {}
    for chunk in iter_csv_chunks(fr, usecols=usecols, chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            code = row.get(code_col)
            title = row.get(title_col)
            if code is None or title is None:
                continue
            code_s = str(code).strip()
            title_s = str(title).strip()
            if not code_s or not title_s:
                continue
            ver_s = ""
            if version_col:
                v = row.get(version_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    ver_s = str(int(v)) if str(v).isdigit() else str(v)
            out[(code_s, ver_s)] = title_s
            if ver_s == "":
                out[(code_s, "")] = title_s
    return out


def _collect_diagnoses(
    diagnoses_table: Optional[DiscoveredTable],
    *,
    hadm_ids: Set[int],
    code_map: Dict[Tuple[str, str], str],
    chunksize: int,
) -> Dict[int, List[str]]:
    if diagnoses_table is None or not diagnoses_table.columns:
        return {}
    cols_lower = {c.lower(): c for c in diagnoses_table.columns}
    hadm_col = cols_lower.get("hadm_id")
    code_col = cols_lower.get("icd_code") or cols_lower.get("diag_code") or cols_lower.get("code")
    ver_col = cols_lower.get("icd_version")
    seq_col = cols_lower.get("seq_num")
    if hadm_col is None or code_col is None:
        return {}

    usecols = [hadm_col, code_col]
    if ver_col:
        usecols.append(ver_col)
    if seq_col:
        usecols.append(seq_col)
    fr = _table_to_fileref(diagnoses_table)

    temp: Dict[int, List[Tuple[int, str]]] = defaultdict(list)  # hadm -> (seq, dx)
    for chunk in iter_csv_chunks(fr, usecols=usecols, chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            hid = row.get(hadm_col)
            if hid is None or (isinstance(hid, float) and pd.isna(hid)):
                continue
            try:
                hid_i = int(hid)
            except Exception:
                continue
            if hid_i not in hadm_ids:
                continue
            code = row.get(code_col)
            if code is None or (isinstance(code, float) and pd.isna(code)):
                continue
            code_s = str(code).strip()
            ver_s = ""
            if ver_col:
                v = row.get(ver_col)
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    ver_s = str(int(v)) if str(v).isdigit() else str(v)
            name = code_map.get((code_s, ver_s)) or code_map.get((code_s, "")) or code_s
            seq = 10**9
            if seq_col:
                s = row.get(seq_col)
                if s is not None and not (isinstance(s, float) and pd.isna(s)):
                    try:
                        seq = int(s)
                    except Exception:
                        seq = 10**9
            temp[hid_i].append((seq, name))

    out: Dict[int, List[str]] = {}
    for hid, items in temp.items():
        items.sort(key=lambda x: x[0])
        out[hid] = [dx for _seq, dx in items]
    return out


def _build_item_label_map(
    dict_table: Optional[DiscoveredTable],
    *,
    id_col: str = "itemid",
    label_col: str = "label",
    chunksize: int,
) -> Dict[int, str]:
    if dict_table is None or not dict_table.columns:
        return {}
    cols_lower = {c.lower(): c for c in dict_table.columns}
    idc = cols_lower.get(id_col)
    labc = cols_lower.get(label_col)
    if idc is None or labc is None:
        return {}
    fr = _table_to_fileref(dict_table)
    out: Dict[int, str] = {}
    for chunk in iter_csv_chunks(fr, usecols=[idc, labc], chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            iid = row.get(idc)
            lab = row.get(labc)
            if iid is None or lab is None:
                continue
            try:
                out[int(iid)] = str(lab)
            except Exception:
                continue
    return out


def _select_itemids_by_label(label_map: Dict[int, str], keywords: Sequence[str]) -> Set[int]:
    kws = [k.lower() for k in keywords]
    out: Set[int] = set()
    for iid, lab in label_map.items():
        ll = (lab or "").lower()
        if any(k in ll for k in kws):
            out.add(iid)
    return out


def _collect_events_by_hadm(
    events_table: Optional[DiscoveredTable],
    *,
    hadm_ids: Set[int],
    item_label_map: Dict[int, str],
    target_itemids: Set[int],
    value_cols: Sequence[str] = ("valuenum", "value", "valueuom"),
    time_cols: Sequence[str] = ("charttime", "chartdate", "storetime"),
    chunksize: int,
    max_per_hadm: int = 200,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Generic collector for labs/vitals-like tables keyed by hadm_id.
    Stores a small bounded list of event dicts per admission.
    """
    if events_table is None or not events_table.columns:
        return {}
    cols_lower = {c.lower(): c for c in events_table.columns}
    hadm_col = cols_lower.get("hadm_id")
    item_col = cols_lower.get("itemid")
    if hadm_col is None or item_col is None:
        return {}

    usecols = [hadm_col, item_col]
    for c in value_cols:
        if c in cols_lower:
            usecols.append(cols_lower[c])
    for c in time_cols:
        if c in cols_lower:
            usecols.append(cols_lower[c])
    if "flag" in cols_lower:
        usecols.append(cols_lower["flag"])
    if "ref_range_lower" in cols_lower:
        usecols.append(cols_lower["ref_range_lower"])
    if "ref_range_upper" in cols_lower:
        usecols.append(cols_lower["ref_range_upper"])

    fr = _table_to_fileref(events_table)
    out: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in iter_csv_chunks(fr, usecols=usecols, chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            hid = row.get(hadm_col)
            iid = row.get(item_col)
            if hid is None or iid is None:
                continue
            try:
                hid_i = int(hid)
                iid_i = int(iid)
            except Exception:
                continue
            if hid_i not in hadm_ids or iid_i not in target_itemids:
                continue
            if len(out[hid_i]) >= max_per_hadm:
                continue
            label = item_label_map.get(iid_i, f"ITEMID_{iid_i}")
            evt: Dict[str, Any] = {"itemid": iid_i, "label": label}
            for c in ("valuenum", "value", "valueuom", "charttime", "chartdate", "storetime", "flag", "ref_range_lower", "ref_range_upper"):
                if c in cols_lower:
                    evt[c] = row.get(cols_lower[c])
            out[hid_i].append(evt)
    return dict(out)


def _render_abnormal_labs(events: List[Dict[str, Any]], max_items: int = 12) -> Dict[str, str]:
    """
    Heuristic: keep flagged abnormal labs if present; else keep out-of-range numeric.
    """
    kept: Dict[str, str] = {}
    for e in events:
        label = str(e.get("label") or "").strip()
        if not label:
            continue
        flag = str(e.get("flag") or "").strip().lower()
        vnum = e.get("valuenum")
        val = e.get("value")
        uom = str(e.get("valueuom") or "").strip()
        lo = e.get("ref_range_lower")
        hi = e.get("ref_range_upper")

        abnormal = False
        if flag and flag not in {"nan", "none"}:
            abnormal = True
        else:
            try:
                x = float(vnum)
                lo_f = float(lo) if lo is not None and not pd.isna(lo) else None
                hi_f = float(hi) if hi is not None and not pd.isna(hi) else None
                if lo_f is not None and x < lo_f:
                    abnormal = True
                if hi_f is not None and x > hi_f:
                    abnormal = True
            except Exception:
                abnormal = False

        if not abnormal:
            continue

        val_str = None
        if vnum is not None and not (isinstance(vnum, float) and pd.isna(vnum)):
            try:
                val_str = f"{float(vnum):g}"
            except Exception:
                val_str = str(vnum)
        elif val is not None and not (isinstance(val, float) and pd.isna(val)):
            val_str = str(val)
        if val_str is None:
            continue
        if uom:
            val_str = f"{val_str} {uom}"
        if label not in kept:
            kept[label] = val_str
        if len(kept) >= max_items:
            break
    return kept


def _render_vitals(events: List[Dict[str, Any]], max_items: int = 10) -> Dict[str, str]:
    """
    Keep a small set of vital sign labels with latest-ish values (no strict time sorting).
    """
    out: Dict[str, str] = {}
    for e in events:
        label = str(e.get("label") or "").strip()
        if not label:
            continue
        vnum = e.get("valuenum")
        val = e.get("value")
        uom = str(e.get("valueuom") or "").strip()
        val_str = None
        if vnum is not None and not (isinstance(vnum, float) and pd.isna(vnum)):
            try:
                val_str = f"{float(vnum):g}"
            except Exception:
                val_str = str(vnum)
        elif val is not None and not (isinstance(val, float) and pd.isna(val)):
            val_str = str(val)
        if val_str is None:
            continue
        if uom:
            val_str = f"{val_str} {uom}"
        out[label] = val_str
        if len(out) >= max_items:
            break
    return out


def _maybe_find_notes_table(tables: List[DiscoveredTable]) -> Optional[DiscoveredTable]:
    # Notes may not be present; schema-based pick.
    t, _ = pick_best_table(tables, "notes")
    return t


def _collect_notes_by_hadm(
    notes_table: Optional[DiscoveredTable],
    *,
    hadm_ids: Set[int],
    chunksize: int,
    max_notes_per_hadm: int = 5,
) -> Dict[int, List[str]]:
    if notes_table is None or not notes_table.columns:
        return {}
    cols_lower = {c.lower(): c for c in notes_table.columns}
    hadm_col = cols_lower.get("hadm_id")
    text_col = cols_lower.get("text")
    if hadm_col is None or text_col is None:
        return {}

    usecols = [hadm_col, text_col]
    # optional note categorization fields
    for c in ("category", "description"):
        if c in cols_lower:
            usecols.append(cols_lower[c])

    fr = _table_to_fileref(notes_table)
    out: Dict[int, List[str]] = defaultdict(list)
    for chunk in iter_csv_chunks(fr, usecols=usecols, chunksize=chunksize):
        for row in chunk.to_dict(orient="records"):
            hid = row.get(hadm_col)
            if hid is None or (isinstance(hid, float) and pd.isna(hid)):
                continue
            try:
                hid_i = int(hid)
            except Exception:
                continue
            if hid_i not in hadm_ids:
                continue
            if len(out[hid_i]) >= max_notes_per_hadm:
                continue
            txt = row.get(text_col)
            if txt is None or (isinstance(txt, float) and pd.isna(txt)):
                continue
            out[hid_i].append(str(txt))
    return dict(out)


def _build_evidence(symptoms: List[str], vitals: Dict[str, str], labs: Dict[str, str]) -> List[str]:
    ev: List[str] = []
    for s in symptoms[:5]:
        if s:
            ev.append(s)
    for k, v in list(vitals.items())[:5]:
        ev.append(f"{k}: {v}")
    for k, v in list(labs.items())[:6]:
        ev.append(f"{k}: {v}")
    # de-duplicate while preserving order
    seen = set()
    out: List[str] = []
    for x in ev:
        xl = x.lower()
        if xl in seen:
            continue
        seen.add(xl)
        out.append(x)
    return out


def _make_incomplete_variant(
    sample: Dict[str, Any],
    rng: random.Random,
) -> Dict[str, Any]:
    s = json.loads(json.dumps(sample))  # deep copy w/o external deps
    s["labels"]["uncertainty_level"] = "high_uncertainty"

    structured = s.get("structured_input", {})
    # randomly remove some evidence sources
    drops = rng.sample(["labs", "vitals", "symptoms", "note"], k=2)
    if "labs" in drops:
        structured["labs"] = {}
    if "vitals" in drops:
        structured["vitals"] = {}
    if "symptoms" in drops:
        structured["symptoms"] = []
    if "note" in drops:
        # strip the notes portion from input_text by removing everything after "Clinical Notes:"
        it = s.get("input_text", "")
        if "Clinical Notes:" in it:
            it = it.split("Clinical Notes:", 1)[0] + "Clinical Notes:\nnone\n"
        s["input_text"] = it

    # rebuild evidence from remaining fields
    s["labels"]["evidence"] = _build_evidence(
        structured.get("symptoms", []) or [],
        structured.get("vitals", {}) or {},
        structured.get("labs", {}) or {},
    )
    s["structured_input"] = structured
    return s


def build_dataset(
    graph: RelationshipGraph,
    config: BuildConfig,
) -> Dict[str, Any]:
    os.makedirs(config.out_dir, exist_ok=True)
    rng = random.Random(config.seed)

    tables = [t for t in graph.tables if t.columns]

    admissions_table, admissions_rank = pick_best_table(tables, "admissions")
    patients_table, patients_rank = pick_best_table(tables, "patients")
    diagnoses_table, diagnoses_rank = pick_best_table(tables, "diagnoses")
    notes_table = _maybe_find_notes_table(tables)
    labs_table, labs_rank = pick_best_table(tables, "labs")
    vitals_table, vitals_rank = pick_best_table(tables, "vitals")
    item_dict_table, _ = pick_best_table(tables, "item_dictionary")

    # Also try to find separate dictionaries for ICD and labs if present
    icd_dict_table = None
    for t in tables:
        cols = set(t.lower_columns)
        if "icd_code" in cols and ("long_title" in cols or "short_title" in cols):
            icd_dict_table = t
            break
    labitem_dict_table = None
    for t in tables:
        cols = set(t.lower_columns)
        if "itemid" in cols and ("fluid" in cols or "loinc_code" in cols or "label" in cols) and ("hadm_id" not in cols and "stay_id" not in cols):
            # heuristic: dictionary-like and not event-like
            labitem_dict_table = t
            # don't break; prefer one with 'label'
            if "label" in cols:
                break

    if admissions_table is None or patients_table is None:
        raise RuntimeError(
            "Could not infer admissions/patients tables from schemas. "
            "Ensure MIMIC-IV CSVs are present under mimic_root."
        )

    admissions = _load_admissions(admissions_table, chunksize=config.chunksize, max_cases=config.max_cases)
    # Identify hadm_id / subject_id columns
    a_cols = {c.lower(): c for c in admissions_table.columns}
    hadm_col = a_cols.get("hadm_id")
    subj_col = a_cols.get("subject_id")
    if hadm_col is None or subj_col is None:
        raise RuntimeError("Admissions table missing hadm_id/subject_id columns.")

    hadm_ids: Set[int] = set()

    for r in admissions:
        try:
            hadm_ids.add(int(r.get(hadm_col)))
        except Exception:
            continue

    patients_map = _load_patients_map(patients_table, chunksize=config.chunksize)

    # ICD code -> title
    code_map = _build_code_map(
        icd_dict_table,
        code_col_candidates=("icd_code",),
        title_col_candidates=("long_title", "short_title", "title", "description"),
        chunksize=config.chunksize,
    )

    dx_by_hadm = _collect_diagnoses(
        diagnoses_table,
        hadm_ids=hadm_ids,
        code_map=code_map,
        chunksize=config.chunksize,
    )

    notes_by_hadm = _collect_notes_by_hadm(notes_table, hadm_ids=hadm_ids, chunksize=config.chunksize)

    # item label maps
    # for vitals, d_items is typical; for labs, d_labitems is typical
    label_map_vitals = _build_item_label_map(item_dict_table, chunksize=config.chunksize)
    if not label_map_vitals and item_dict_table is not None:
        # sometimes label column differs
        label_map_vitals = _build_item_label_map(item_dict_table, label_col="name", chunksize=config.chunksize)

    label_map_labs = _build_item_label_map(labitem_dict_table, chunksize=config.chunksize)
    if not label_map_labs:
        # fallback: use same item dict if it exists
        label_map_labs = dict(label_map_vitals)

    vital_itemids = _select_itemids_by_label(
        label_map_vitals,
        keywords=(
            "heart rate",
            "hr",
            "respiratory rate",
            "rr",
            "temperature",
            "temp",
            "spo2",
            "o2 saturation",
            "systolic blood pressure",
            "diastolic blood pressure",
            "mean blood pressure",
            "sbp",
            "dbp",
            "map",
            "non invasive blood pressure",
        ),
    )
    lab_itemids = _select_itemids_by_label(
        label_map_labs,
        keywords=("wbc", "white blood", "hemoglobin", "hgb", "platelet", "plt", "creatinine", "crp", "c-reactive"),
    )

    labs_events = _collect_events_by_hadm(
        labs_table,
        hadm_ids=hadm_ids,
        item_label_map=label_map_labs,
        target_itemids=lab_itemids,
        chunksize=config.chunksize,
        max_per_hadm=300,
    )
    vitals_events = _collect_events_by_hadm(
        vitals_table,
        hadm_ids=hadm_ids,
        item_label_map=label_map_vitals,
        target_itemids=vital_itemids,
        chunksize=config.chunksize,
        max_per_hadm=200,
    )

    out_path = os.path.join(config.out_dir, "data.jsonl")
    stats_dx = Counter()
    total_written = 0
    total_len = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for adm in tqdm(admissions, desc="Building cases"):
            hid = adm.get(hadm_col)
            sid = adm.get(subj_col)
            if hid is None or sid is None:
                continue
            try:
                hid_i = int(hid)
                sid_i = int(sid)
            except Exception:
                continue

            dxs = dx_by_hadm.get(hid_i, [])
            if not dxs:
                continue
            primary = dxs[0]
            diffs = dxs[1:10]

            p = patients_map.get(sid_i, {})
            age = _infer_age(p, adm)
            sex = _infer_sex(p)

            note_texts = notes_by_hadm.get(hid_i, [])
            note_summary = simple_note_summary(note_texts)
            symptoms = []
            if note_texts:
                symptoms = extract_chief_complaint(note_texts[0])

            labs = _render_abnormal_labs(labs_events.get(hid_i, []))
            vitals = _render_vitals(vitals_events.get(hid_i, []))

            # Filtering: require some clinical text
            if len(note_summary) < config.min_note_chars and not (labs or vitals or symptoms):
                continue

            input_text = render_case_text(
                age=age,
                sex=sex,
                symptoms=symptoms,
                vitals=vitals,
                labs=labs,
                note_summary=note_summary,
            )

            structured_input = {
                "age": age,
                "sex": sex,
                "symptoms": symptoms,
                "vitals": vitals,
                "labs": labs,
            }
            evidence = _build_evidence(symptoms, vitals, labs)

            complete = {
                "case_id": str(hid_i),
                "input_text": input_text,
                "structured_input": structured_input,
                "labels": {
                    "primary_diagnosis": primary,
                    "differential_diagnoses": diffs,
                    "evidence": evidence,
                    "uncertainty_level": "low_uncertainty",
                },
            }

            incomplete = _make_incomplete_variant(complete, rng)

            for sample in (complete, incomplete):
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                total_written += 1
                total_len += len(sample.get("input_text", ""))

            stats_dx[primary] += 1

    meta = {
        "total_cases": total_written,
        "unique_admissions": len({r.get(hadm_col) for r in admissions if r.get(hadm_col) is not None}),
        "diagnosis_distribution": dict(stats_dx.most_common(200)),
        "average_case_length": (total_len / total_written) if total_written else 0.0,
        "selected_tables": {
            "admissions": getattr(admissions_table, "table_id", None),
            "patients": getattr(patients_table, "table_id", None),
            "diagnoses": getattr(diagnoses_table, "table_id", None),
            "notes": getattr(notes_table, "table_id", None) if notes_table else None,
            "labs": getattr(labs_table, "table_id", None),
            "vitals": getattr(vitals_table, "table_id", None),
            "item_dictionary": getattr(item_dict_table, "table_id", None),
        },
        "table_rank_debug": {
            "admissions_top": admissions_rank,
            "patients_top": patients_rank,
            "diagnoses_top": diagnoses_rank,
            "labs_top": labs_rank,
            "vitals_top": vitals_rank,
        },
    }
    with open(os.path.join(config.out_dir, "metadata.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    return meta


