"""Aggregate node opportunities over bounded reachable networks."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from accessibility.reachability import (
    bounded_reachable_nodes,
)


OPPORTUNITY_COLUMNS = (
    "jobs",
    "schools",
    "transit_stations",
    "bus_stops",
    "greenspace",
    "healthcare",
    "stores",
)


@dataclass(frozen=True)
class OpportunityIndex:
    """Validated node-to-opportunity lookup."""

    values_by_node: dict[str, tuple[float, ...]]
    columns: tuple[str, ...] = OPPORTUNITY_COLUMNS

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        node_column: str = "node_id",
    ) -> "OpportunityIndex":
        """Build a validated lookup from a node opportunity table."""
        required = {
            node_column,
            *OPPORTUNITY_COLUMNS,
        }

        missing_columns = required.difference(frame.columns)

        if missing_columns:
            raise ValueError(
                "Opportunity table is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        normalized = frame[
            [node_column, *OPPORTUNITY_COLUMNS]
        ].copy()

        node_ids = normalized[node_column].astype("string")

        if node_ids.isna().any():
            raise ValueError("Opportunity node IDs may not be missing.")

        normalized[node_column] = (
            node_ids.str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        if normalized[node_column].eq("").any():
            raise ValueError("Opportunity node IDs may not be blank.")

        if normalized[node_column].duplicated().any():
            duplicates = (
                normalized.loc[
                    normalized[node_column].duplicated(
                        keep=False
                    ),
                    node_column,
                ]
                .drop_duplicates()
                .head(10)
                .tolist()
            )

            raise ValueError(
                "Opportunity node IDs must be unique. "
                f"Examples: {duplicates}"
            )

        for column in OPPORTUNITY_COLUMNS:
            numeric = pd.to_numeric(
                normalized[column],
                errors="coerce",
            )

            if numeric.isna().any():
                raise ValueError(
                    f"Opportunity column {column!r} "
                    "contains missing or non-numeric values."
                )

            values = numeric.to_numpy(dtype=float)

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Opportunity column {column!r} "
                    "contains non-finite values."
                )

            if (values < 0).any():
                raise ValueError(
                    f"Opportunity column {column!r} "
                    "contains negative values."
                )

            normalized[column] = values

        values_by_node = {
            str(row[node_column]): tuple(
                float(row[column])
                for column in OPPORTUNITY_COLUMNS
            )
            for _, row in normalized.iterrows()
        }

        return cls(values_by_node=values_by_node)

    @classmethod
    def from_csv(
        cls,
        path: Path,
        *,
        node_column: str = "node_id",
    ) -> "OpportunityIndex":
        """Load and validate a node opportunity CSV."""
        frame = pd.read_csv(
            path,
            dtype={node_column: "string"},
        )

        return cls.from_frame(
            frame,
            node_column=node_column,
        )

    def totals_for_nodes(
        self,
        nodes: Iterable[Hashable],
    ) -> dict[str, float]:
        """Sum every opportunity category across supplied nodes."""
        totals = [0.0] * len(self.columns)
        missing_nodes = []

        for node in nodes:
            node_id = str(node)
            values = self.values_by_node.get(node_id)

            if values is None:
                missing_nodes.append(node_id)
                continue

            for index, value in enumerate(values):
                totals[index] += value

        if missing_nodes:
            examples = sorted(set(missing_nodes))[:10]

            raise KeyError(
                "Reachable nodes are absent from the "
                f"opportunity table. Examples: {examples}"
            )

        return {
            column: totals[index]
            for index, column in enumerate(self.columns)
        }


@dataclass(frozen=True)
class OpportunityAccessibilityResult:
    """Reachable opportunity totals for one origin and profile."""

    origin_node: Hashable
    weight_attribute: str
    budget: float
    max_lts: float | None
    reachable_node_count: int
    processed_node_count: int
    opportunity_totals: dict[str, float]

    def to_record(self) -> dict:
        """Return a flat serializable result record."""
        record = {
            "origin_node": self.origin_node,
            "weight_attribute": self.weight_attribute,
            "budget": self.budget,
            "max_lts": self.max_lts,
            "reachable_node_count": (
                self.reachable_node_count
            ),
            "processed_node_count": (
                self.processed_node_count
            ),
        }

        record.update(self.opportunity_totals)
        return record


def calculate_opportunity_accessibility(
    graph: nx.Graph,
    opportunities: OpportunityIndex,
    origin_node: Hashable,
    budget: float,
    weight_attribute: str,
    *,
    max_lts: float | None = None,
    lts_attribute: str = "max_lts",
) -> OpportunityAccessibilityResult:
    """Calculate reachable opportunity totals for one origin."""
    reachability = bounded_reachable_nodes(
        graph=graph,
        origin_node=origin_node,
        budget=budget,
        weight_attribute=weight_attribute,
        max_lts=max_lts,
        lts_attribute=lts_attribute,
    )

    totals = opportunities.totals_for_nodes(
        reachability.reachable_nodes
    )

    return OpportunityAccessibilityResult(
        origin_node=origin_node,
        weight_attribute=weight_attribute,
        budget=reachability.budget,
        max_lts=reachability.max_lts,
        reachable_node_count=(
            reachability.reachable_node_count
        ),
        processed_node_count=(
            reachability.processed_node_count
        ),
        opportunity_totals=totals,
    )
