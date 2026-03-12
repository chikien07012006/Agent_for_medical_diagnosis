from __future__ import annotations

from itertools import combinations
from typing import List

from .types import DiscoveredTable, RelationEdge, RelationshipGraph


KEYS = ("subject_id", "hadm_id", "stay_id")


def infer_relationship_graph(tables: List[DiscoveredTable]) -> RelationshipGraph:
    """Infer a simple undirected graph based on shared key columns."""
    edges: List[RelationEdge] = []
    for a, b in combinations(tables, 2):
        if not a.columns or not b.columns:
            continue
        a_cols = set(a.lower_columns)
        b_cols = set(b.lower_columns)
        shared = [k for k in KEYS if (k in a_cols and k in b_cols)]
        if shared:
            edges.append(
                RelationEdge(
                    left_table_id=a.table_id,
                    right_table_id=b.table_id,
                    shared_keys=shared,
                )
            )
    return RelationshipGraph(tables=tables, edges=edges)


