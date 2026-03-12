from __future__ import annotations

import argparse
import json
import os

from .build_dataset import BuildConfig, build_dataset
from .discovery import discover_tables
from .relations import infer_relationship_graph


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build admission-level diagnostic dataset from local MIMIC-IV tables.")
    p.add_argument("--mimic_root", required=True, help="Root directory containing MIMIC-IV modules (scanned recursively).")
    p.add_argument("--out_dir", required=True, help="Output directory for dataset/ (data.jsonl, metadata.json).")
    p.add_argument("--cache_dir", default=None, help="Optional cache directory (reserved for future use).")
    p.add_argument("--max_cases", type=int, default=None, help="Optional cap on number of admissions to process.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for incomplete-variant generation.")
    p.add_argument("--chunksize", type=int, default=200_000, help="Chunk size for streaming CSV reads.")
    p.add_argument("--schema_sample_rows", type=int, default=200, help="Rows to sample per table to infer schema.")
    p.add_argument("--min_note_chars", type=int, default=40, help="Minimum note chars (or other evidence) to keep a case.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    mimic_root = os.path.abspath(args.mimic_root)
    out_dir = os.path.abspath(args.out_dir)

    tables = discover_tables(mimic_root, schema_sample_rows=args.schema_sample_rows)
    graph = infer_relationship_graph(tables)

    # Save relationship graph for transparency/debugging
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "relationship_graph.json"), "w", encoding="utf-8") as f:
        json.dump(graph.to_dict(), f, ensure_ascii=False, indent=2)

    meta = build_dataset(
        graph,
        BuildConfig(
            mimic_root=mimic_root,
            out_dir=out_dir,
            cache_dir=args.cache_dir,
            max_cases=args.max_cases,
            seed=args.seed,
            chunksize=args.chunksize,
            min_note_chars=args.min_note_chars,
        ),
    )

    print("Wrote dataset to:", out_dir)
    print("Total samples (including uncertainty variants):", meta.get("total_cases"))
    print("Average case length:", meta.get("average_case_length"))
    print("Top diagnoses:")
    for dx, n in list((meta.get("diagnosis_distribution") or {}).items())[:10]:
        print(f"  {dx}: {n}")


if __name__ == "__main__":
    main()


