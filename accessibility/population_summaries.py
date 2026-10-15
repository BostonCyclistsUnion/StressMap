"""Build regional and tract population-weighted accessibility summaries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from accessibility.opportunity_access import OPPORTUNITY_COLUMNS
from accessibility.population_weighting import (
    aggregate_node_population,
    load_population_allocation,
)
from accessibility.profile_access import DEFAULT_PROFILE_SPECS


PROFILE_NAMES = tuple(
    profile.name
    for profile in DEFAULT_PROFILE_SPECS
)

ACCESS_COLUMNS = tuple(
    f"{profile}_{opportunity}"
    for profile in PROFILE_NAMES
    for opportunity in OPPORTUNITY_COLUMNS
)


@dataclass(frozen=True)
class SummaryBuildResult:
    """Metadata for generated population-weighted summaries."""

    node_count: int
    allocation_row_count: int
    tract_count: int
    total_population: float
    regional_row_count: int
    tract_row_count: int
    zero_population_tract_count: int
    regional_output_path: Path
    tract_output_path: Path
    diagnostics_path: Path | None


def _normalize_ids(
    values: pd.Series,
    *,
    label: str,
) -> pd.Series:
    normalized = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    if normalized.isna().any():
        raise ValueError(
            f"{label} may not contain missing values."
        )

    if normalized.eq("").any():
        raise ValueError(
            f"{label} may not contain blank values."
        )

    return normalized


def load_population_nodes(
    path: Path,
) -> pd.DataFrame:
    """Load and validate node accessibility with population."""
    frame = pd.read_csv(
        path,
        dtype={"origin_node": "string"},
    )

    required = {
        "origin_node",
        "assigned_total_population",
        *ACCESS_COLUMNS,
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            "Population-node input is missing columns: "
            f"{sorted(missing)}"
        )

    frame["origin_node"] = _normalize_ids(
        frame["origin_node"],
        label="Accessibility origin IDs",
    )

    if frame["origin_node"].duplicated().any():
        raise ValueError(
            "Accessibility origin IDs must be unique."
        )

    numeric_columns = (
        "assigned_total_population",
        *ACCESS_COLUMNS,
    )

    for column in numeric_columns:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

        if values.isna().any():
            raise ValueError(
                f"{column!r} contains missing or "
                "non-numeric values."
            )

        array = values.to_numpy(dtype=float)

        if not np.isfinite(array).all():
            raise ValueError(
                f"{column!r} contains non-finite values."
            )

        if (array < 0).any():
            raise ValueError(
                f"{column!r} contains negative values."
            )

        frame[column] = array

    for profile in PROFILE_NAMES:
        if profile == "distance":
            continue

        for opportunity in OPPORTUNITY_COLUMNS:
            profile_values = frame[
                f"{profile}_{opportunity}"
            ].to_numpy(dtype=float)

            distance_values = frame[
                f"distance_{opportunity}"
            ].to_numpy(dtype=float)

            invalid = (
                profile_values > distance_values
            ) & ~np.isclose(
                profile_values,
                distance_values,
                rtol=1e-12,
                atol=1e-9,
            )

            if invalid.any():
                raise ValueError(
                    f"{profile}_{opportunity} exceeds "
                    f"distance_{opportunity}."
                )

    return frame


def _weighted_metrics(
    access: np.ndarray,
    distance_access: np.ndarray,
    population: np.ndarray,
) -> dict[str, float]:
    """Calculate absolute and relative weighted metrics."""
    total_population = float(
        population.sum()
    )

    weighted_access_sum = float(
        np.dot(population, access)
    )

    weighted_distance_sum = float(
        np.dot(population, distance_access)
    )

    if total_population > 0:
        weighted_mean_access = (
            weighted_access_sum
            / total_population
        )

        weighted_mean_distance = (
            weighted_distance_sum
            / total_population
        )
    else:
        weighted_mean_access = np.nan
        weighted_mean_distance = np.nan

    if weighted_distance_sum > 0:
        ratio_of_weighted_means = (
            weighted_access_sum
            / weighted_distance_sum
        )
    else:
        ratio_of_weighted_means = np.nan

    valid_relative = distance_access > 0

    valid_relative_population = float(
        population[valid_relative].sum()
    )

    zero_distance_access_population = float(
        population[~valid_relative].sum()
    )

    if valid_relative_population > 0:
        origin_relative = (
            access[valid_relative]
            / distance_access[valid_relative]
        )

        weighted_mean_origin_relative = float(
            np.dot(
                population[valid_relative],
                origin_relative,
            )
            / valid_relative_population
        )
    else:
        weighted_mean_origin_relative = np.nan

    if total_population > 0:
        valid_relative_population_share = (
            valid_relative_population
            / total_population
        )
    else:
        valid_relative_population_share = np.nan

    return {
        "population_weighted_access_sum": (
            weighted_access_sum
        ),
        "population_weighted_mean_access": (
            weighted_mean_access
        ),
        "population_weighted_distance_access_sum": (
            weighted_distance_sum
        ),
        "population_weighted_mean_distance_access": (
            weighted_mean_distance
        ),
        "ratio_of_weighted_means": (
            ratio_of_weighted_means
        ),
        "valid_relative_population": (
            valid_relative_population
        ),
        "zero_distance_access_population": (
            zero_distance_access_population
        ),
        "valid_relative_population_share": (
            valid_relative_population_share
        ),
        "population_weighted_mean_origin_relative": (
            weighted_mean_origin_relative
        ),
    }


def summarize_region(
    nodes: pd.DataFrame,
) -> pd.DataFrame:
    """Build one regional row per profile and opportunity."""
    population = nodes[
        "assigned_total_population"
    ].to_numpy(dtype=float)

    total_population = float(
        population.sum()
    )

    positive_population_nodes = int(
        (population > 0).sum()
    )

    rows = []

    for profile in PROFILE_NAMES:
        for opportunity in OPPORTUNITY_COLUMNS:
            access = nodes[
                f"{profile}_{opportunity}"
            ].to_numpy(dtype=float)

            distance_access = nodes[
                f"distance_{opportunity}"
            ].to_numpy(dtype=float)

            metrics = _weighted_metrics(
                access,
                distance_access,
                population,
            )

            rows.append(
                {
                    "profile": profile,
                    "opportunity": opportunity,
                    "accessibility_node_count": (
                        len(nodes)
                    ),
                    "positive_population_node_count": (
                        positive_population_nodes
                    ),
                    "total_population": (
                        total_population
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def summarize_tracts(
    nodes: pd.DataFrame,
    allocation: pd.DataFrame,
) -> pd.DataFrame:
    """Build population-weighted summaries by Census tract."""
    node_ids = set(
        nodes["origin_node"].astype(str)
    )

    allocation_ids = set(
        allocation["node_id"].astype(str)
    )

    unknown_nodes = allocation_ids.difference(
        node_ids
    )

    if unknown_nodes:
        examples = sorted(unknown_nodes)[:10]

        raise ValueError(
            "Population allocation contains nodes absent "
            "from the accessibility data. "
            f"Examples: {examples}"
        )

    metric_source = nodes[
        [
            "origin_node",
            *ACCESS_COLUMNS,
        ]
    ]

    joined = allocation.merge(
        metric_source,
        left_on="node_id",
        right_on="origin_node",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    if joined["origin_node"].isna().any():
        raise ValueError(
            "Some allocation rows could not be joined "
            "to accessibility origins."
        )

    rows = []

    for geoid, tract in joined.groupby(
        "GEOID",
        sort=True,
    ):
        population = tract[
            "assigned_population"
        ].to_numpy(dtype=float)

        tract_population = float(
            population.sum()
        )

        allocation_row_count = len(tract)

        allocated_node_count = int(
            tract["node_id"].nunique()
        )

        for profile in PROFILE_NAMES:
            for opportunity in OPPORTUNITY_COLUMNS:
                access = tract[
                    f"{profile}_{opportunity}"
                ].to_numpy(dtype=float)

                distance_access = tract[
                    f"distance_{opportunity}"
                ].to_numpy(dtype=float)

                metrics = _weighted_metrics(
                    access,
                    distance_access,
                    population,
                )

                rows.append(
                    {
                        "GEOID": str(geoid),
                        "profile": profile,
                        "opportunity": opportunity,
                        "allocation_row_count": (
                            allocation_row_count
                        ),
                        "allocated_node_count": (
                            allocated_node_count
                        ),
                        "tract_population": (
                            tract_population
                        ),
                        **metrics,
                    }
                )

    return pd.DataFrame(rows)


def _validate_population_consistency(
    nodes: pd.DataFrame,
    allocation: pd.DataFrame,
) -> None:
    """Require node population to equal raw allocation totals."""
    node_population = aggregate_node_population(
        allocation
    )

    accessibility_nodes = set(
        nodes["origin_node"].astype(str)
    )

    allocation_nodes = set(
        node_population["node_id"].astype(str)
    )

    unknown_nodes = allocation_nodes.difference(
        accessibility_nodes
    )

    if unknown_nodes:
        examples = sorted(unknown_nodes)[:10]

        raise ValueError(
            "Population allocation contains unknown nodes. "
            f"Examples: {examples}"
        )

    comparison = nodes[
        [
            "origin_node",
            "assigned_total_population",
        ]
    ].merge(
        node_population[
            [
                "node_id",
                "assigned_total_population",
            ]
        ],
        left_on="origin_node",
        right_on="node_id",
        how="left",
        suffixes=("_nodes", "_allocation"),
        validate="one_to_one",
        sort=False,
    )

    comparison[
        "assigned_total_population_allocation"
    ] = comparison[
        "assigned_total_population_allocation"
    ].fillna(0.0)

    if not np.allclose(
        comparison[
            "assigned_total_population_nodes"
        ],
        comparison[
            "assigned_total_population_allocation"
        ],
        rtol=1e-10,
        atol=1e-10,
    ):
        raise ValueError(
            "Node population does not match the raw "
            "tract-to-node allocation."
        )


def _validate_summary_outputs(
    regional: pd.DataFrame,
    tract: pd.DataFrame,
    tract_count: int,
) -> None:
    """Validate row counts, keys, ranges, and aggregation."""
    expected_regional_rows = (
        len(PROFILE_NAMES)
        * len(OPPORTUNITY_COLUMNS)
    )

    expected_tract_rows = (
        tract_count
        * expected_regional_rows
    )

    if len(regional) != expected_regional_rows:
        raise ValueError(
            "Unexpected regional summary row count."
        )

    if len(tract) != expected_tract_rows:
        raise ValueError(
            "Unexpected tract summary row count."
        )

    if regional.duplicated(
        ["profile", "opportunity"]
    ).any():
        raise ValueError(
            "Regional summary contains duplicate keys."
        )

    if tract.duplicated(
        ["GEOID", "profile", "opportunity"]
    ).any():
        raise ValueError(
            "Tract summary contains duplicate keys."
        )

    bounded_columns = (
        "ratio_of_weighted_means",
        "valid_relative_population_share",
        "population_weighted_mean_origin_relative",
    )

    for frame_name, frame in (
        ("regional", regional),
        ("tract", tract),
    ):
        for column in bounded_columns:
            present = frame[column].dropna()

            invalid = (
                (present < -1e-12)
                | (present > 1 + 1e-12)
            )

            if invalid.any():
                raise ValueError(
                    f"{frame_name} {column} is outside "
                    "[0, 1]."
                )

    for _, regional_row in regional.iterrows():
        matching = tract[
            (
                tract["profile"]
                == regional_row["profile"]
            )
            & (
                tract["opportunity"]
                == regional_row["opportunity"]
            )
        ]

        comparisons = (
            "population_weighted_access_sum",
            "population_weighted_distance_access_sum",
            "valid_relative_population",
            "zero_distance_access_population",
        )

        for column in comparisons:
            tract_total = float(
                matching[column].sum()
            )

            regional_total = float(
                regional_row[column]
            )

            if not np.isclose(
                tract_total,
                regional_total,
                rtol=1e-10,
                atol=1e-8,
            ):
                raise ValueError(
                    "Regional and tract summaries disagree "
                    f"for {regional_row['profile']}, "
                    f"{regional_row['opportunity']}, "
                    f"{column}."
                )


def _write_csv_atomically(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    if temporary.exists():
        temporary.unlink()

    frame.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        path,
    )


def build_population_weighted_summaries(
    nodes_path: Path,
    allocation_path: Path,
    regional_output_path: Path,
    tract_output_path: Path,
    *,
    diagnostics_path: Path | None = None,
) -> SummaryBuildResult:
    """Build, validate, and save regional and tract summaries."""
    nodes = load_population_nodes(
        nodes_path
    )

    allocation = load_population_allocation(
        allocation_path
    )

    _validate_population_consistency(
        nodes,
        allocation,
    )

    regional = summarize_region(nodes)

    tract = summarize_tracts(
        nodes,
        allocation,
    )

    tract_count = int(
        allocation["GEOID"].nunique()
    )

    _validate_summary_outputs(
        regional,
        tract,
        tract_count,
    )

    _write_csv_atomically(
        regional,
        regional_output_path,
    )

    _write_csv_atomically(
        tract,
        tract_output_path,
    )

    tract_populations = (
        tract[
            [
                "GEOID",
                "tract_population",
            ]
        ]
        .drop_duplicates("GEOID")
    )

    total_population = float(
        nodes[
            "assigned_total_population"
        ].sum()
    )

    diagnostics = {
        "nodes_path": str(nodes_path),
        "allocation_path": str(
            allocation_path
        ),
        "regional_output_path": str(
            regional_output_path
        ),
        "tract_output_path": str(
            tract_output_path
        ),
        "node_count": len(nodes),
        "positive_population_node_count": int(
            (
                nodes[
                    "assigned_total_population"
                ]
                > 0
            ).sum()
        ),
        "allocation_row_count": len(
            allocation
        ),
        "tract_count": tract_count,
        "zero_population_tract_count": int(
            (
                tract_populations[
                    "tract_population"
                ]
                == 0
            ).sum()
        ),
        "total_population": total_population,
        "profile_count": len(PROFILE_NAMES),
        "opportunity_count": len(
            OPPORTUNITY_COLUMNS
        ),
        "regional_row_count": len(
            regional
        ),
        "tract_row_count": len(tract),
    }

    if diagnostics_path is not None:
        diagnostics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = diagnostics_path.with_name(
            diagnostics_path.name + ".tmp"
        )

        temporary.write_text(
            json.dumps(
                diagnostics,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        os.replace(
            temporary,
            diagnostics_path,
        )

    return SummaryBuildResult(
        node_count=len(nodes),
        allocation_row_count=len(
            allocation
        ),
        tract_count=tract_count,
        total_population=total_population,
        regional_row_count=len(regional),
        tract_row_count=len(tract),
        zero_population_tract_count=(
            diagnostics[
                "zero_population_tract_count"
            ]
        ),
        regional_output_path=(
            regional_output_path
        ),
        tract_output_path=tract_output_path,
        diagnostics_path=diagnostics_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build regional and tract population-weighted "
            "accessibility summaries."
        )
    )

    parser.add_argument(
        "--nodes",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--allocation",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--regional-output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--tract-output",
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

    result = (
        build_population_weighted_summaries(
            nodes_path=args.nodes,
            allocation_path=args.allocation,
            regional_output_path=(
                args.regional_output
            ),
            tract_output_path=(
                args.tract_output
            ),
            diagnostics_path=args.diagnostics,
        )
    )

    print(
        "POPULATION-WEIGHTED SUMMARY "
        "VALIDATION PASSED"
    )

    print(
        "Accessibility nodes:",
        f"{result.node_count:,}",
    )

    print(
        "Allocation rows:",
        f"{result.allocation_row_count:,}",
    )

    print(
        "Census tracts:",
        f"{result.tract_count:,}",
    )

    print(
        "Zero-population tracts:",
        f"{result.zero_population_tract_count:,}",
    )

    print(
        "Modeled population:",
        f"{result.total_population:,.6f}",
    )

    print(
        "Regional rows:",
        f"{result.regional_row_count:,}",
    )

    print(
        "Tract rows:",
        f"{result.tract_row_count:,}",
    )

    print(
        "Regional output:",
        result.regional_output_path,
    )

    print(
        "Tract output:",
        result.tract_output_path,
    )

    if result.diagnostics_path is not None:
        print(
            "Diagnostics:",
            result.diagnostics_path,
        )


if __name__ == "__main__":
    main()
