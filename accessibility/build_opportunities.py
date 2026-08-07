"""Build node-level job and amenity opportunity counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree


CATEGORY_MAP = {
    "Schools": "schools",
    "Transit stations": "transit_stations",
    "Bus stops": "bus_stops",
    "Greenspace": "greenspace",
    "Healthcare": "healthcare",
    "Stores": "stores",
}

OPPORTUNITY_COLUMNS = [
    "jobs",
    "schools",
    "transit_stations",
    "bus_stops",
    "greenspace",
    "healthcare",
    "stores",
]


def clean_node_ids(series: pd.Series) -> pd.Series:
    """Normalize graph node identifiers read from tabular files."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    """Raise an informative error when required columns are absent."""
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing)}"
        )


def load_graph_node_table(
    graph_path: Path,
) -> tuple[nx.Graph, pd.DataFrame]:
    """Load a GraphML network and return its node-coordinate table."""
    graph = nx.read_graphml(graph_path)

    rows = []

    for node_id, attributes in graph.nodes(data=True):
        if "x" not in attributes or "y" not in attributes:
            raise ValueError(
                f"Graph node {node_id!r} is missing x or y coordinates."
            )

        rows.append(
            {
                "node_id": str(node_id),
                "longitude": float(attributes["x"]),
                "latitude": float(attributes["y"]),
            }
        )

    nodes = pd.DataFrame(rows)

    if nodes.empty:
        raise ValueError("The graph contains no nodes.")

    if nodes["node_id"].duplicated().any():
        raise ValueError("The graph contains duplicate node identifiers.")

    return graph, nodes


def build_amenity_counts(
    destinations_path: Path,
    graph_node_ids: set[str],
    max_snap_distance_m: float,
) -> tuple[pd.DataFrame, dict]:
    """Aggregate already-snapped amenity destinations by graph node."""
    destinations = pd.read_csv(
        destinations_path,
        dtype={
            "destination_id": "string",
            "nearest_node_id": "string",
        },
    )

    require_columns(
        destinations,
        {
            "destination_id",
            "display_category",
            "nearest_node_id",
            "snap_distance_meters",
        },
        "Destination file",
    )

    if destinations["destination_id"].isna().any():
        raise ValueError("Destination IDs may not be missing.")

    if destinations["destination_id"].duplicated().any():
        raise ValueError("Destination IDs must be unique.")

    destinations["nearest_node_id"] = clean_node_ids(
        destinations["nearest_node_id"]
    )

    destinations["snap_distance_meters"] = pd.to_numeric(
        destinations["snap_distance_meters"],
        errors="coerce",
    )

    if destinations["snap_distance_meters"].isna().any():
        raise ValueError(
            "Destination snap distances must all be numeric."
        )

    unknown_categories = sorted(
        set(
            destinations["display_category"]
            .dropna()
            .astype(str)
            .unique()
        )
        - set(CATEGORY_MAP)
    )

    if unknown_categories:
        raise ValueError(
            "Unexpected destination categories: "
            f"{unknown_categories}"
        )

    destinations["opportunity_category"] = (
        destinations["display_category"].map(CATEGORY_MAP)
    )

    destinations["node_in_graph"] = (
        destinations["nearest_node_id"].isin(graph_node_ids)
    )

    destinations["within_snap_limit"] = (
        destinations["snap_distance_meters"]
        <= float(max_snap_distance_m)
    )

    accepted = destinations[
        destinations["node_in_graph"]
        & destinations["within_snap_limit"]
    ].copy()

    accepted["opportunity_count"] = 1

    amenity_counts = (
        accepted.pivot_table(
            index="nearest_node_id",
            columns="opportunity_category",
            values="opportunity_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"nearest_node_id": "node_id"})
    )

    for column in CATEGORY_MAP.values():
        if column not in amenity_counts.columns:
            amenity_counts[column] = 0

    amenity_counts = amenity_counts[
        ["node_id", *CATEGORY_MAP.values()]
    ]

    category_totals = {
        category: int(
            accepted["opportunity_category"]
            .eq(category)
            .sum()
        )
        for category in CATEGORY_MAP.values()
    }

    diagnostics = {
        "source_rows": int(len(destinations)),
        "accepted_rows": int(len(accepted)),
        "missing_graph_node_rows": int(
            (~destinations["node_in_graph"]).sum()
        ),
        "beyond_snap_limit_rows": int(
            (
                destinations["node_in_graph"]
                & ~destinations["within_snap_limit"]
            ).sum()
        ),
        "maximum_allowed_snap_distance_m": float(
            max_snap_distance_m
        ),
        "maximum_source_snap_distance_m": float(
            destinations["snap_distance_meters"].max()
        ),
        "category_totals": category_totals,
    }

    return amenity_counts, diagnostics


def build_job_counts(
    lodes_path: Path,
    source_graph_path: Path,
    graph_nodes: pd.DataFrame,
    projected_crs: str,
    max_reassignment_distance_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate LODES jobs and reassign nearby removed graph nodes."""
    lodes = pd.read_csv(
        lodes_path,
        dtype={"destination_node": "string"},
    )

    require_columns(
        lodes,
        {
            "destination_node",
            "employees",
        },
        "LODES file",
    )

    lodes["destination_node"] = clean_node_ids(
        lodes["destination_node"]
    )

    lodes["employees"] = pd.to_numeric(
        lodes["employees"],
        errors="coerce",
    )

    if lodes["employees"].isna().any():
        raise ValueError("LODES employee values must be numeric.")

    if (lodes["employees"] < 0).any():
        raise ValueError("LODES employee values may not be negative.")

    jobs_by_source_node = (
        lodes.groupby(
            "destination_node",
            as_index=False,
        )["employees"]
        .sum()
        .rename(
            columns={
                "destination_node": "source_node_id",
                "employees": "jobs",
            }
        )
    )

    graph_node_ids = set(graph_nodes["node_id"])

    direct = jobs_by_source_node[
        jobs_by_source_node["source_node_id"].isin(
            graph_node_ids
        )
    ].copy()

    direct = direct.rename(
        columns={"source_node_id": "node_id"}
    )

    missing = jobs_by_source_node[
        ~jobs_by_source_node["source_node_id"].isin(
            graph_node_ids
        )
    ].copy()

    reassignment_columns = [
        "source_node_id",
        "jobs",
        "longitude",
        "latitude",
        "nearest_node_id",
        "reassignment_distance_m",
        "accepted",
    ]

    if missing.empty:
        reassignment = pd.DataFrame(
            columns=reassignment_columns
        )

        final_jobs = direct[["node_id", "jobs"]].copy()

        diagnostics = {
            "source_jobs": float(
                jobs_by_source_node["jobs"].sum()
            ),
            "direct_jobs": float(direct["jobs"].sum()),
            "missing_workplace_nodes": 0,
            "reassigned_workplace_nodes": 0,
            "excluded_workplace_nodes": 0,
            "reassigned_jobs": 0.0,
            "excluded_jobs": 0.0,
            "maximum_reassignment_distance_m": float(
                max_reassignment_distance_m
            ),
        }

        return final_jobs, reassignment, diagnostics

    source_graph = nx.read_graphml(source_graph_path)

    source_coordinates = []

    for row in missing.itertuples(index=False):
        node_id = str(row.source_node_id)

        if node_id not in source_graph:
            source_coordinates.append(
                {
                    "source_node_id": node_id,
                    "jobs": float(row.jobs),
                    "longitude": np.nan,
                    "latitude": np.nan,
                }
            )
            continue

        attributes = source_graph.nodes[node_id]

        source_coordinates.append(
            {
                "source_node_id": node_id,
                "jobs": float(row.jobs),
                "longitude": float(attributes["x"]),
                "latitude": float(attributes["y"]),
            }
        )

    missing_locations = pd.DataFrame(source_coordinates)

    valid_locations = missing_locations.dropna(
        subset=["longitude", "latitude"]
    ).copy()

    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )

    graph_xy = np.asarray(
        list(
            transformer.itransform(
                zip(
                    graph_nodes["longitude"],
                    graph_nodes["latitude"],
                )
            )
        ),
        dtype=float,
    )

    missing_xy = np.asarray(
        list(
            transformer.itransform(
                zip(
                    valid_locations["longitude"],
                    valid_locations["latitude"],
                )
            )
        ),
        dtype=float,
    )

    tree = cKDTree(graph_xy)

    distances, positions = tree.query(
        missing_xy,
        k=1,
    )

    valid_locations["nearest_node_id"] = [
        graph_nodes.iloc[int(position)]["node_id"]
        for position in positions
    ]

    valid_locations["reassignment_distance_m"] = distances

    valid_locations["accepted"] = (
        valid_locations["reassignment_distance_m"]
        <= float(max_reassignment_distance_m)
    )

    invalid_locations = missing_locations[
        missing_locations["longitude"].isna()
        | missing_locations["latitude"].isna()
    ].copy()

    if not invalid_locations.empty:
        invalid_locations["nearest_node_id"] = pd.NA
        invalid_locations["reassignment_distance_m"] = np.nan
        invalid_locations["accepted"] = False

    reassignment = pd.concat(
        [
            valid_locations,
            invalid_locations,
        ],
        ignore_index=True,
    )[reassignment_columns]

    reassignment["accepted"] = (
        reassignment["accepted"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    accepted_reassignment = reassignment[
        reassignment["accepted"]
    ].copy()

    reassigned_jobs = (
        accepted_reassignment.groupby(
            "nearest_node_id",
            as_index=False,
        )["jobs"]
        .sum()
        .rename(
            columns={"nearest_node_id": "node_id"}
        )
    )

    final_jobs = (
        pd.concat(
            [
                direct[["node_id", "jobs"]],
                reassigned_jobs[["node_id", "jobs"]],
            ],
            ignore_index=True,
        )
        .groupby("node_id", as_index=False)["jobs"]
        .sum()
    )

    excluded = reassignment[
        ~reassignment["accepted"]
    ]

    diagnostics = {
        "source_jobs": float(
            jobs_by_source_node["jobs"].sum()
        ),
        "direct_jobs": float(direct["jobs"].sum()),
        "missing_workplace_nodes": int(len(missing)),
        "reassigned_workplace_nodes": int(
            accepted_reassignment["source_node_id"].nunique()
        ),
        "excluded_workplace_nodes": int(
            excluded["source_node_id"].nunique()
        ),
        "reassigned_jobs": float(
            accepted_reassignment["jobs"].sum()
        ),
        "excluded_jobs": float(excluded["jobs"].sum()),
        "maximum_reassignment_distance_m": float(
            max_reassignment_distance_m
        ),
        "maximum_observed_reassignment_distance_m": float(
            valid_locations["reassignment_distance_m"].max()
        ),
    }

    return final_jobs, reassignment, diagnostics


def build_opportunity_table(
    graph_path: Path,
    source_graph_path: Path,
    destinations_path: Path,
    lodes_path: Path,
    output_path: Path,
    diagnostics_path: Path,
    reassignment_path: Path,
    projected_crs: str = "EPSG:26986",
    max_job_reassignment_distance_m: float = 100.0,
    max_amenity_snap_distance_m: float = 300.0,
) -> pd.DataFrame:
    """Create one opportunity-count row for every graph node."""
    graph, graph_nodes = load_graph_node_table(graph_path)
    graph_node_ids = set(graph_nodes["node_id"])

    amenity_counts, amenity_diagnostics = (
        build_amenity_counts(
            destinations_path=destinations_path,
            graph_node_ids=graph_node_ids,
            max_snap_distance_m=max_amenity_snap_distance_m,
        )
    )

    job_counts, reassignment, job_diagnostics = (
        build_job_counts(
            lodes_path=lodes_path,
            source_graph_path=source_graph_path,
            graph_nodes=graph_nodes,
            projected_crs=projected_crs,
            max_reassignment_distance_m=(
                max_job_reassignment_distance_m
            ),
        )
    )

    output = graph_nodes.merge(
        job_counts,
        on="node_id",
        how="left",
    )

    output = output.merge(
        amenity_counts,
        on="node_id",
        how="left",
    )

    for column in OPPORTUNITY_COLUMNS:
        if column not in output.columns:
            output[column] = 0

        output[column] = (
            pd.to_numeric(
                output[column],
                errors="coerce",
            )
            .fillna(0)
        )

    output = output[
        [
            "node_id",
            "longitude",
            "latitude",
            *OPPORTUNITY_COLUMNS,
        ]
    ].sort_values("node_id")

    if output["node_id"].duplicated().any():
        raise RuntimeError(
            "Output contains duplicate graph nodes."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reassignment_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(output_path, index=False)

    reassignment.sort_values(
        [
            "accepted",
            "jobs",
            "source_node_id",
        ],
        ascending=[False, False, True],
    ).to_csv(
        reassignment_path,
        index=False,
    )

    output_totals = {
        column: float(output[column].sum())
        for column in OPPORTUNITY_COLUMNS
    }

    diagnostics = {
        "graph": {
            "nodes": int(graph.number_of_nodes()),
            "edges": int(graph.number_of_edges()),
        },
        "amenities": amenity_diagnostics,
        "jobs": job_diagnostics,
        "output_totals": output_totals,
    }

    diagnostics_path.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine LODES jobs and snapped amenities into "
            "a graph-node opportunity table."
        )
    )

    parser.add_argument(
        "--graph",
        required=True,
        type=Path,
        help="Pruned analysis GraphML file.",
    )

    parser.add_argument(
        "--source-graph",
        required=True,
        type=Path,
        help=(
            "Unpruned GraphML file used to locate removed "
            "workplace nodes."
        ),
    )

    parser.add_argument(
        "--destinations",
        required=True,
        type=Path,
        help="Standardized and snapped destination CSV.",
    )

    parser.add_argument(
        "--lodes",
        required=True,
        type=Path,
        help="LODES origin-destination job-pair CSV.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output opportunity-by-node CSV.",
    )

    parser.add_argument(
        "--diagnostics",
        required=True,
        type=Path,
        help="Output JSON diagnostics file.",
    )

    parser.add_argument(
        "--job-reassignments",
        required=True,
        type=Path,
        help="Output audit CSV for removed workplace nodes.",
    )

    parser.add_argument(
        "--projected-crs",
        default="EPSG:26986",
    )

    parser.add_argument(
        "--max-job-reassignment-distance-m",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--max-amenity-snap-distance-m",
        type=float,
        default=300.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output = build_opportunity_table(
        graph_path=args.graph,
        source_graph_path=args.source_graph,
        destinations_path=args.destinations,
        lodes_path=args.lodes,
        output_path=args.output,
        diagnostics_path=args.diagnostics,
        reassignment_path=args.job_reassignments,
        projected_crs=args.projected_crs,
        max_job_reassignment_distance_m=(
            args.max_job_reassignment_distance_m
        ),
        max_amenity_snap_distance_m=(
            args.max_amenity_snap_distance_m
        ),
    )

    print(f"Saved: {args.output}")
    print(f"Graph nodes written: {len(output):,}")

    for column in OPPORTUNITY_COLUMNS:
        print(
            f"{column}: "
            f"{output[column].sum():,.0f}"
        )

    print(f"Diagnostics: {args.diagnostics}")
    print(
        "Job reassignment audit: "
        f"{args.job_reassignments}"
    )


if __name__ == "__main__":
    main()
