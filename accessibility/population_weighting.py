"""Attach Census-assigned population to node accessibility results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


NODE_POPULATION_COLUMNS = (
    "assigned_total_population",
    "contributing_tract_count",
)


@dataclass(frozen=True)
class PopulationJoinSummary:
    """Summary of one node-population join."""

    accessibility_node_count: int
    allocated_node_count: int
    zero_allocation_node_count: int
    allocation_row_count: int
    tract_count: int
    multiple_tract_node_count: int
    assigned_population_total: float
    output_path: Path
    diagnostics_path: Path | None


def _normalize_ids(
    values: pd.Series,
    *,
    label: str,
) -> pd.Series:
    """Normalize and validate identifier values."""
    normalized = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    if normalized.isna().any():
        raise ValueError(f"{label} may not contain missing values.")

    if normalized.eq("").any():
        raise ValueError(f"{label} may not contain blank values.")

    return normalized


def load_population_allocation(
    path: Path,
) -> pd.DataFrame:
    """Load and validate a tract-to-node population allocation."""
    allocation = pd.read_csv(
        path,
        dtype={
            "node_id": "string",
            "GEOID": "string",
        },
    )

    required = {
        "node_id",
        "GEOID",
        "assigned_population",
    }

    missing = required.difference(allocation.columns)

    if missing:
        raise ValueError(
            "Population allocation is missing columns: "
            f"{sorted(missing)}"
        )

    allocation = allocation[
        [
            "node_id",
            "GEOID",
            "assigned_population",
        ]
    ].copy()

    allocation["node_id"] = _normalize_ids(
        allocation["node_id"],
        label="Allocation node IDs",
    )

    allocation["GEOID"] = _normalize_ids(
        allocation["GEOID"],
        label="Allocation tract IDs",
    )

    population = pd.to_numeric(
        allocation["assigned_population"],
        errors="coerce",
    )

    if population.isna().any():
        raise ValueError(
            "assigned_population contains missing or "
            "non-numeric values."
        )

    values = population.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "assigned_population contains non-finite values."
        )

    if (values < 0).any():
        raise ValueError(
            "assigned_population contains negative values."
        )

    allocation["assigned_population"] = values

    if allocation.duplicated(
        ["node_id", "GEOID"]
    ).any():
        raise ValueError(
            "Population allocation contains duplicate "
            "node-and-tract pairs."
        )

    return allocation.sort_values(
        ["GEOID", "node_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def aggregate_node_population(
    allocation: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate tract allocation rows to one row per node."""
    result = (
        allocation.groupby(
            "node_id",
            as_index=False,
            sort=True,
        )
        .agg(
            assigned_total_population=(
                "assigned_population",
                "sum",
            ),
            contributing_tract_count=(
                "GEOID",
                "nunique",
            ),
        )
    )

    result["contributing_tract_count"] = (
        result["contributing_tract_count"]
        .astype(int)
    )

    return result


def attach_population_to_accessibility(
    accessibility: pd.DataFrame,
    allocation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Attach aggregated population to every accessibility node."""
    if "origin_node" not in accessibility.columns:
        raise ValueError(
            "Accessibility data is missing 'origin_node'."
        )

    conflicting = set(
        NODE_POPULATION_COLUMNS
    ).intersection(accessibility.columns)

    if conflicting:
        raise ValueError(
            "Accessibility data already contains population "
            f"columns: {sorted(conflicting)}"
        )

    accessibility = accessibility.copy()

    accessibility["origin_node"] = _normalize_ids(
        accessibility["origin_node"],
        label="Accessibility origin IDs",
    )

    if accessibility["origin_node"].duplicated().any():
        raise ValueError(
            "Accessibility origin IDs must be unique."
        )

    node_population = aggregate_node_population(
        allocation
    )

    accessibility_nodes = set(
        accessibility["origin_node"].astype(str)
    )

    allocation_nodes = set(
        node_population["node_id"].astype(str)
    )

    unknown_allocation_nodes = (
        allocation_nodes.difference(
            accessibility_nodes
        )
    )

    if unknown_allocation_nodes:
        examples = sorted(
            unknown_allocation_nodes
        )[:10]

        raise ValueError(
            "Population allocation contains nodes absent "
            "from the accessibility data. "
            f"Examples: {examples}"
        )

    joined = accessibility.merge(
        node_population,
        left_on="origin_node",
        right_on="node_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    joined.drop(
        columns=["node_id"],
        inplace=True,
    )

    no_allocation = joined[
        "assigned_total_population"
    ].isna()

    joined["assigned_total_population"] = (
        joined["assigned_total_population"]
        .fillna(0.0)
        .astype(float)
    )

    joined["contributing_tract_count"] = (
        joined["contributing_tract_count"]
        .fillna(0)
        .astype(int)
    )

    original_columns = [
        column
        for column in accessibility.columns
        if column != "origin_node"
    ]

    joined = joined[
        [
            "origin_node",
            "assigned_total_population",
            "contributing_tract_count",
            *original_columns,
        ]
    ]

    multiple_tract_nodes = int(
        (
            node_population[
                "contributing_tract_count"
            ]
            > 1
        ).sum()
    )

    diagnostics = {
        "accessibility_node_count": len(
            accessibility
        ),
        "allocation_row_count": len(allocation),
        "allocated_node_count": len(
            node_population
        ),
        "zero_allocation_node_count": int(
            no_allocation.sum()
        ),
        "tract_count": int(
            allocation["GEOID"].nunique()
        ),
        "multiple_tract_node_count": (
            multiple_tract_nodes
        ),
        "assigned_population_total": float(
            allocation[
                "assigned_population"
            ].sum()
        ),
    }

    return joined, diagnostics


def build_population_attached_nodes(
    accessibility_path: Path,
    allocation_path: Path,
    output_path: Path,
    *,
    diagnostics_path: Path | None = None,
) -> PopulationJoinSummary:
    """Build and atomically save node accessibility with population."""
    accessibility = pd.read_csv(
        accessibility_path,
        dtype={"origin_node": "string"},
    )

    allocation = load_population_allocation(
        allocation_path
    )

    joined, diagnostics = (
        attach_population_to_accessibility(
            accessibility,
            allocation,
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = output_path.with_name(
        output_path.name + ".tmp"
    )

    if temporary_output.exists():
        temporary_output.unlink()

    joined.to_csv(
        temporary_output,
        index=False,
    )

    os.replace(
        temporary_output,
        output_path,
    )

    diagnostics.update(
        {
            "accessibility_path": str(
                accessibility_path
            ),
            "allocation_path": str(
                allocation_path
            ),
            "output_path": str(output_path),
            "output_row_count": len(joined),
            "output_column_count": len(
                joined.columns
            ),
        }
    )

    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_diagnostics = (
            diagnostics_path.with_name(
                diagnostics_path.name + ".tmp"
            )
        )

        temporary_diagnostics.write_text(
            json.dumps(
                diagnostics,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        os.replace(
            temporary_diagnostics,
            diagnostics_path,
        )

    return PopulationJoinSummary(
        accessibility_node_count=diagnostics[
            "accessibility_node_count"
        ],
        allocated_node_count=diagnostics[
            "allocated_node_count"
        ],
        zero_allocation_node_count=diagnostics[
            "zero_allocation_node_count"
        ],
        allocation_row_count=diagnostics[
            "allocation_row_count"
        ],
        tract_count=diagnostics["tract_count"],
        multiple_tract_node_count=diagnostics[
            "multiple_tract_node_count"
        ],
        assigned_population_total=diagnostics[
            "assigned_population_total"
        ],
        output_path=output_path,
        diagnostics_path=diagnostics_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach Census-assigned population to node "
            "accessibility results."
        )
    )

    parser.add_argument(
        "--accessibility",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--allocation",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--diagnostics",
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = build_population_attached_nodes(
        accessibility_path=args.accessibility,
        allocation_path=args.allocation,
        output_path=args.output,
        diagnostics_path=args.diagnostics,
    )

    print("POPULATION JOIN VALIDATION PASSED")
    print(
        "Accessibility nodes:",
        f"{summary.accessibility_node_count:,}",
    )
    print(
        "Allocated nodes:",
        f"{summary.allocated_node_count:,}",
    )
    print(
        "Nodes assigned zero population:",
        f"{summary.zero_allocation_node_count:,}",
    )
    print(
        "Allocation rows:",
        f"{summary.allocation_row_count:,}",
    )
    print(
        "Census tracts:",
        f"{summary.tract_count:,}",
    )
    print(
        "Multiple-tract nodes:",
        f"{summary.multiple_tract_node_count:,}",
    )
    print(
        "Assigned population:",
        f"{summary.assigned_population_total:,.6f}",
    )
    print(f"Output: {summary.output_path}")

    if summary.diagnostics_path is not None:
        print(
            "Diagnostics:",
            summary.diagnostics_path,
        )


if __name__ == "__main__":
    main()
