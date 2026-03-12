from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


KEY_CANDIDATES = ("subject_id", "hadm_id", "stay_id")


@dataclass(frozen=True)
class DiscoveredTable:
    """Metadata about a discovered table-like resource (CSV, CSV inside ZIP, etc.)."""

    # A stable identifier for the table resource (human readable).
    table_id: str

    # Absolute path to the source file on disk.
    source_path: str

    # Optional member path when the source is an archive (e.g., zip member).
    member_path: Optional[str]

    # Detected "format": 'csv' | 'zip_csv' | 'parquet' (extendable).
    format: str

    # Column names discovered from the header/sample.
    columns: List[str]

    # Best-effort pandas/pyarrow-like dtype strings (from a small sample).
    dtypes: Dict[str, str] = field(default_factory=dict)

    # Row count is optional; often unknown without scanning.
    approx_rows: Optional[int] = None

    # Extra notes for debugging / provenance.
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def lower_columns(self) -> List[str]:
        return [c.lower() for c in self.columns]

    def has_any(self, *names: str) -> bool:
        cols = set(self.lower_columns)
        return any(n.lower() in cols for n in names)

    def has_all(self, *names: str) -> bool:
        cols = set(self.lower_columns)
        return all(n.lower() in cols for n in names)

    def key_columns(self) -> List[str]:
        cols = set(self.lower_columns)
        return [k for k in KEY_CANDIDATES if k in cols]


@dataclass(frozen=True)
class RelationEdge:
    left_table_id: str
    right_table_id: str
    shared_keys: List[str]


@dataclass(frozen=True)
class RelationshipGraph:
    """Simple key-based relationship graph."""

    tables: List[DiscoveredTable]
    edges: List[RelationEdge]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tables": [
                {
                    "table_id": t.table_id,
                    "source_path": t.source_path,
                    "member_path": t.member_path,
                    "format": t.format,
                    "columns": t.columns,
                    "dtypes": t.dtypes,
                    "approx_rows": t.approx_rows,
                }
                for t in self.tables
            ],
            "edges": [
                {
                    "left_table_id": e.left_table_id,
                    "right_table_id": e.right_table_id,
                    "shared_keys": e.shared_keys,
                }
                for e in self.edges
            ],
        }


