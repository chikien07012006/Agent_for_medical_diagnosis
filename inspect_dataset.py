import json
import os
import random
from collections import Counter
from typing import Any, Dict, List

import pandas as pd


def load_jsonl(path: str, max_rows: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def print_random_cases(rows: List[Dict[str, Any]], n: int = 5) -> None:
    print("\n=== RANDOM SAMPLE CASES ===")
    if not rows:
        print("No rows loaded.")
        return
    rng = random.Random(0)
    for i, row in enumerate(rng.sample(rows, k=min(n, len(rows)))):
        labels = row.get("labels", {})
        primary = labels.get("primary_diagnosis")
        diffs = labels.get("differential_diagnoses", [])
        u = labels.get("uncertainty_level")
        sinput = row.get("structured_input", {})
        age = sinput.get("age")
        sex = sinput.get("sex")
        print(f"\n--- CASE {i+1} ---")
        print("case_id:", row.get("case_id"))
        print("age / sex:", age, "/", sex)
        print("primary_diagnosis:", primary)
        print("num_differentials:", len(diffs))
        print("uncertainty_level:", u)
        print("symptoms:", sinput.get("symptoms"))
        print("num_labs:", len(sinput.get("labs", {})))
        print("num_vitals:", len(sinput.get("vitals", {})))
        text = (row.get("input_text") or "").strip()
        if len(text) > 600:
            text = text[:600] + " ... [TRUNCATED]"
        print("\ninput_text preview:\n", text)


def summarize_diagnoses(rows: List[Dict[str, Any]], top_k: int = 20) -> None:
    print("\n=== DIAGNOSIS DISTRIBUTION (PRIMARY, TOP) ===")
    c = Counter()
    for r in rows:
        dx = (r.get("labels") or {}).get("primary_diagnosis")
        if dx:
            c[dx] += 1
    for dx, n in c.most_common(top_k):
        print(f"{dx}: {n}")


def summarize_age_sex(rows: List[Dict[str, Any]]) -> None:
    print("\n=== AGE / SEX SUMMARY ===")
    records: List[Dict[str, Any]] = []
    for r in rows:
        sinput = r.get("structured_input", {})
        age = sinput.get("age")
        sex = sinput.get("sex")
        if age is None:
            continue
        records.append({"age": age, "sex": sex})
    if not records:
        print("No age/sex info available.")
        return
    df = pd.DataFrame.from_records(records)
    print("\nCount by sex:")
    print(df["sex"].value_counts(dropna=False))
    print("\nAge stats overall:")
    print(df["age"].describe())
    print("\nAge stats by sex:")
    print(df.groupby("sex")["age"].describe())


def main() -> None:
    dataset_path = os.path.join("dataset", "data.jsonl")
    if not os.path.exists(dataset_path):
        raise SystemExit(f"Dataset not found at {dataset_path}. Run the builder CLI first.")

    # Load a manageable subset for quick inspection
    rows = load_jsonl(dataset_path, max_rows=5000)
    print(f"Loaded {len(rows)} rows from {dataset_path}")

    print_random_cases(rows, n=5)
    summarize_diagnoses(rows, top_k=20)
    summarize_age_sex(rows)


if __name__ == "__main__":
    main()


