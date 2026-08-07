"""Generate reproducible population-weighted origin-destination trips."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.stats import lognorm

TRIP_TYPES = {
    "elementary_school": {
        "destination_columns": ("schools",),
        "selection_method": "nearest",
    },
    "healthcare": {
        "destination_columns": ("healthcare",),
        "selection_method": "nearest",
    },
    "work": {
        "destination_columns": ("jobs",),
        "selection_method": "lognormal_weighted",
    },
    "transit": {
        "destination_columns": (
            "transit_stations",
            "bus_stops",
        ),
        "selection_method": "lognormal_weighted",
    },
    "shopping": {
        "destination_columns": ("stores",),
        "selection_method": "lognormal_weighted",
    },
    "recreation": {
        "destination_columns": ("greenspace",),
        "selection_method": "lognormal_weighted",
    },
}
DEMAND_CATEGORY_MAP = {
    "home_school": "elementary_school",
    "home_office": "work",
    "home_healthcare": "healthcare",
    "home_transit": "transit",
    "home_greenspace": "recreation",
    "home_store": "shopping",
}

DEFAULT_DEMAND_CONFIG = (
    Path(__file__).resolve().parent
    / "config"
    / "demand_parameters.csv"
)


OUTPUT_COLUMNS = [
    "origin_node",
    "destination_node",
    "trip_type",
    "selection_method",
    "origin_population",
    "destination_weight",
    "straight_line_distance_m",
    "origin_longitude",
    "origin_latitude",
    "destination_longitude",
    "destination_latitude",
    "count",
    "seed",
]


def clean_node_ids(series: pd.Series) -> pd.Series:
    """Normalize node identifiers loaded from tabular data."""
    return (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    """Raise an informative error when required columns are absent."""
    missing = set(required).difference(frame.columns)

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {sorted(missing)}"
        )


def load_demand_scenario(
    config_path: Path,
    scenario_id: str | int,
) -> dict[str, int]:
    """Load category-specific trip counts for one scenario."""
    config = pd.read_csv(
        config_path,
        dtype="string",
    )

    if config.empty or len(config.columns) < 2:
        raise ValueError(
            "Demand configuration must contain category "
            "and scenario columns."
        )

    category_column = config.columns[0]
    scenario_column = str(scenario_id)

    if scenario_column not in config.columns:
        raise ValueError(
            f"Demand scenario {scenario_column!r} was not "
            f"found. Available scenarios: "
            f"{list(config.columns[1:])}"
        )

    counts: dict[str, int] = {}
    declared_total: int | None = None

    for _, row in config.iterrows():
        category = str(row[category_column]).strip()
        raw_value = row[scenario_column]

        if pd.isna(raw_value) or not str(raw_value).strip():
            continue

        try:
            count = int(float(str(raw_value)))
        except ValueError as error:
            raise ValueError(
                f"Invalid demand count {raw_value!r} "
                f"for category {category!r}."
            ) from error

        if count < 0:
            raise ValueError(
                f"Demand count for {category!r} "
                "may not be negative."
            )

        if category == "TOTAL":
            declared_total = count
            continue

        if category not in DEMAND_CATEGORY_MAP:
            raise ValueError(
                f"Unknown demand category {category!r}. "
                f"Supported categories: "
                f"{sorted(DEMAND_CATEGORY_MAP)}"
            )

        if count > 0:
            trip_type = DEMAND_CATEGORY_MAP[category]

            if trip_type in counts:
                raise ValueError(
                    f"Duplicate demand category for "
                    f"trip type {trip_type!r}."
                )

            counts[trip_type] = count

    calculated_total = sum(counts.values())

    if (
        declared_total is not None
        and declared_total != calculated_total
    ):
        raise ValueError(
            "Demand TOTAL does not equal the sum of "
            f"category counts: {declared_total} != "
            f"{calculated_total}."
        )

    if not counts:
        raise ValueError(
            f"Demand scenario {scenario_column!r} "
            "contains no positive trip counts."
        )

    return counts


def _graphml_local_name(tag: str) -> str:
    """Return a GraphML tag name without its XML namespace."""
    return tag.rsplit("}", 1)[-1]


def load_graph_node_table(
    graph_path: Path,
    projected_crs: str,
) -> pd.DataFrame:
    """Load graph nodes without parsing the full edge section."""
    key_names: dict[str, str] = {}
    rows: list[dict[str, object]] = []

    with graph_path.open("rb") as graph_stream:
        context = ET.iterparse(
            graph_stream,
            events=("start", "end"),
        )

        for event, element in context:
            tag = _graphml_local_name(element.tag)

            # GraphML stores nodes before edges. Trip generation
            # needs only node coordinates, so stop before parsing
            # the large edge section.
            if event == "start" and tag == "edge":
                break

            if event != "end":
                continue

            if tag == "key":
                key_id = element.attrib.get("id")
                key_scope = element.attrib.get("for")
                attribute_name = element.attrib.get(
                    "attr.name"
                )

                if (
                    key_id
                    and attribute_name
                    and key_scope in {"node", "all"}
                ):
                    key_names[key_id] = attribute_name

                element.clear()
                continue

            if tag != "node":
                continue

            node_id = element.attrib.get("id")

            if node_id is None:
                raise ValueError(
                    "GraphML contains a node without an ID."
                )

            attributes: dict[str, str] = {}

            for child in element:
                if (
                    _graphml_local_name(child.tag)
                    != "data"
                ):
                    continue

                key_id = child.attrib.get("key", "")
                attribute_name = key_names.get(key_id)

                if (
                    attribute_name in {"x", "y"}
                    and child.text is not None
                ):
                    attributes[attribute_name] = (
                        child.text
                    )

            if "x" not in attributes or "y" not in attributes:
                raise ValueError(
                    f"Graph node {node_id!r} is missing "
                    "x or y coordinates."
                )

            rows.append(
                {
                    "node_id": str(node_id),
                    "longitude": float(attributes["x"]),
                    "latitude": float(attributes["y"]),
                }
            )

            element.clear()

    nodes = pd.DataFrame(rows)

    if nodes.empty:
        raise ValueError(
            "The GraphML file contains no readable nodes."
        )

    if nodes["node_id"].duplicated().any():
        raise ValueError(
            "The graph contains duplicate node identifiers."
        )

    transformer = Transformer.from_crs(
        "EPSG:4326",
        projected_crs,
        always_xy=True,
    )

    projected_coordinates = np.asarray(
        list(
            transformer.itransform(
                zip(
                    nodes["longitude"],
                    nodes["latitude"],
                )
            )
        ),
        dtype=float,
    )

    if (
        projected_coordinates.ndim != 2
        or projected_coordinates.shape[1] != 2
        or not np.isfinite(
            projected_coordinates
        ).all()
    ):
        raise ValueError(
            "Graph node coordinates could not be "
            "projected successfully."
        )

    nodes["x_m"] = projected_coordinates[:, 0]
    nodes["y_m"] = projected_coordinates[:, 1]

    return nodes


def load_population_origins(
    allocation_path: Path,
    graph_node_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate tract allocations into population per graph node."""
    allocation = pd.read_csv(
        allocation_path,
        dtype={"node_id": "string"},
    )

    require_columns(
        allocation,
        {"node_id", "assigned_population"},
        "Census allocation file",
    )

    allocation["node_id"] = clean_node_ids(
        allocation["node_id"]
    )

    allocation["assigned_population"] = pd.to_numeric(
        allocation["assigned_population"],
        errors="coerce",
    )

    invalid_id = (
        allocation["node_id"].isna()
        | allocation["node_id"].eq("")
    )

    invalid_population = (
        allocation["assigned_population"].isna()
        | ~np.isfinite(allocation["assigned_population"])
        | allocation["assigned_population"].lt(0)
    )

    invalid = invalid_id | invalid_population

    if invalid.any():
        bad_rows = allocation.index[invalid].tolist()[:10]

        raise ValueError(
            "Census allocation contains invalid node IDs or "
            f"population values at rows {bad_rows}."
        )

    grouped = (
        allocation.groupby(
            "node_id",
            as_index=False,
        )["assigned_population"]
        .sum()
        .rename(
            columns={
                "assigned_population": "origin_population"
            }
        )
    )

    unknown_mask = pd.Series(
        False,
        index=grouped.index,
    )

    if graph_node_ids is not None:
        unknown_mask = ~grouped["node_id"].isin(
            graph_node_ids
        )

    positive_mask = grouped["origin_population"].gt(0)

    usable = grouped[
        ~unknown_mask & positive_mask
    ].copy()

    usable = usable.sort_values(
        "node_id"
    ).reset_index(drop=True)

    diagnostics = {
        "allocation_rows": len(allocation),
        "aggregated_allocation_nodes": len(grouped),
        "unknown_graph_nodes": int(unknown_mask.sum()),
        "zero_population_nodes": int(
            (~positive_mask).sum()
        ),
        "usable_origin_nodes": len(usable),
        "source_population": float(
            allocation["assigned_population"].sum()
        ),
        "usable_origin_population": float(
            usable["origin_population"].sum()
        ),
    }

    if usable.empty:
        raise ValueError(
            "No positive-population origins remain "
            "after validation."
        )

    return usable, diagnostics


def load_opportunities(
    opportunities_path: Path,
) -> pd.DataFrame:
    """Load and validate the node-level opportunity table."""
    opportunities = pd.read_csv(
        opportunities_path,
        dtype={"node_id": "string"},
    )

    required = {"node_id"}

    for config in TRIP_TYPES.values():
        required.update(
            config["destination_columns"]
        )

    require_columns(
        opportunities,
        required,
        "Opportunity file",
    )

    opportunities["node_id"] = clean_node_ids(
        opportunities["node_id"]
    )

    if opportunities["node_id"].isna().any():
        raise ValueError(
            "Opportunity node IDs may not be missing."
        )

    if opportunities["node_id"].duplicated().any():
        raise ValueError(
            "Opportunity node IDs must be unique."
        )

    value_columns = sorted(
        required.difference({"node_id"})
    )

    for column in value_columns:
        opportunities[column] = pd.to_numeric(
            opportunities[column],
            errors="coerce",
        )

        invalid = (
            opportunities[column].isna()
            | ~np.isfinite(opportunities[column])
            | opportunities[column].lt(0)
        )

        if invalid.any():
            raise ValueError(
                f"Opportunity column {column!r} "
                "contains invalid values."
            )

    return opportunities


def build_destination_pool(
    opportunities: pd.DataFrame,
    trip_type: str,
    graph_node_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a positive-weight pool for one trip type."""
    if trip_type not in TRIP_TYPES:
        raise ValueError(
            f"Unknown trip type {trip_type!r}. "
            f"Choose from {sorted(TRIP_TYPES)}."
        )

    config = TRIP_TYPES[trip_type]

    destination_columns = list(
        config["destination_columns"]
    )

    require_columns(
        opportunities,
        {"node_id", *destination_columns},
        "Opportunity table",
    )

    pool = opportunities[
        ["node_id", *destination_columns]
    ].copy()

    pool["node_id"] = clean_node_ids(
        pool["node_id"]
    )

    pool["destination_weight"] = pool[
        destination_columns
    ].sum(axis=1)

    unknown_mask = pd.Series(
        False,
        index=pool.index,
    )

    if graph_node_ids is not None:
        unknown_mask = ~pool["node_id"].isin(
            graph_node_ids
        )

    positive_mask = pool[
        "destination_weight"
    ].gt(0)

    usable = pool[
        ~unknown_mask & positive_mask
    ][
        [
            "node_id",
            "destination_weight",
        ]
    ].copy()

    usable = usable.sort_values(
        "node_id"
    ).reset_index(drop=True)

    diagnostics = {
        "trip_type": trip_type,
        "selection_method": config[
            "selection_method"
        ],
        "source_rows": len(pool),
        "unknown_graph_nodes": int(
            unknown_mask.sum()
        ),
        "zero_weight_rows": int(
            (~positive_mask).sum()
        ),
        "usable_destination_nodes": len(usable),
        "total_destination_weight": float(
            usable["destination_weight"].sum()
        ),
    }

    return usable, diagnostics


def _candidate_table(
    origin_node: str,
    destination_pool: pd.DataFrame,
    node_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Attach coordinates and calculate origin distances."""
    if origin_node not in node_lookup.index:
        raise ValueError(
            f"Origin node {origin_node!r} "
            "is not in the graph."
        )

    if destination_pool.empty:
        raise ValueError(
            "Destination pool is empty."
        )

    candidates = destination_pool.copy()

    candidates = candidates[
        candidates["node_id"].isin(
            node_lookup.index
        )
    ]

    if candidates.empty:
        raise ValueError(
            "No destination-pool nodes are "
            "present in the graph."
        )

    non_origin = candidates[
        candidates["node_id"].ne(origin_node)
    ]

    if not non_origin.empty:
        candidates = non_origin

    coordinates = node_lookup.loc[
        candidates["node_id"],
        [
            "longitude",
            "latitude",
            "x_m",
            "y_m",
        ],
    ].reset_index()

    candidates = candidates.merge(
        coordinates,
        on="node_id",
        how="inner",
        validate="one_to_one",
    )

    origin = node_lookup.loc[origin_node]

    candidates[
        "straight_line_distance_m"
    ] = np.hypot(
        candidates["x_m"] - float(origin["x_m"]),
        candidates["y_m"] - float(origin["y_m"]),
    )

    return candidates


def select_nearest_destination(
    origin_node: str,
    destination_pool: pd.DataFrame,
    node_lookup: pd.DataFrame,
) -> pd.Series:
    """Select the closest eligible destination."""
    candidates = _candidate_table(
        origin_node,
        destination_pool,
        node_lookup,
    )

    candidates = candidates.sort_values(
        [
            "straight_line_distance_m",
            "node_id",
        ],
        kind="mergesort",
    )

    return candidates.iloc[0]



def select_distance_weighted_destination(
    origin_node: str,
    destination_pool: pd.DataFrame,
    node_lookup: pd.DataFrame,
    rng: np.random.Generator,
    lognormal_scale_m: float = 2414.0,
    lognormal_sigma: float = 0.7899,
) -> pd.Series:
    """Sample using opportunity and lognormal distance weights."""
    if (
        not math.isfinite(lognormal_scale_m)
        or lognormal_scale_m <= 0
    ):
        raise ValueError(
            "lognormal_scale_m must be positive and finite."
        )

    if (
        not math.isfinite(lognormal_sigma)
        or lognormal_sigma <= 0
    ):
        raise ValueError(
            "lognormal_sigma must be positive and finite."
        )

    candidates = _candidate_table(
        origin_node,
        destination_pool,
        node_lookup,
    )

    distances = np.maximum(
        candidates[
            "straight_line_distance_m"
        ].to_numpy(dtype=float),
        1.0,
    )

    distance_density = lognorm.pdf(
        distances,
        s=float(lognormal_sigma),
        scale=float(lognormal_scale_m),
    )

    sampling_weights = (
        candidates[
            "destination_weight"
        ].to_numpy(dtype=float)
        * distance_density
    )

    total = float(sampling_weights.sum())

    if not math.isfinite(total) or total <= 0:
        sampling_weights = candidates[
            "destination_weight"
        ].to_numpy(dtype=float)

        total = float(sampling_weights.sum())

    if not math.isfinite(total) or total <= 0:
        raise ValueError(
            "Destination sampling weights do not "
            "have a positive sum."
        )

    selected_position = int(
        rng.choice(
            len(candidates),
            p=sampling_weights / total,
        )
    )

    return candidates.iloc[selected_position]



def _prepare_destination_pool(
    destination_pool: pd.DataFrame,
    node_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """Join destination coordinates once per trip type."""
    require_columns(
        destination_pool,
        {
            "node_id",
            "destination_weight",
        },
        "Destination pool",
    )

    required_node_columns = {
        "longitude",
        "latitude",
        "x_m",
        "y_m",
    }

    missing_node_columns = (
        required_node_columns
        .difference(node_lookup.columns)
    )

    if missing_node_columns:
        raise ValueError(
            "Graph node lookup is missing required "
            f"columns: {sorted(missing_node_columns)}"
        )

    coordinates = node_lookup[
        [
            "longitude",
            "latitude",
            "x_m",
            "y_m",
        ]
    ].copy()

    coordinates.index = (
        coordinates.index.astype(str)
    )

    prepared = destination_pool.copy()

    prepared["node_id"] = (
        prepared["node_id"]
        .astype("string")
        .str.strip()
    )

    prepared = prepared.join(
        coordinates,
        on="node_id",
        how="inner",
        validate="one_to_one",
    )

    if len(prepared) != len(destination_pool):
        missing_count = (
            len(destination_pool) - len(prepared)
        )

        raise ValueError(
            f"{missing_count:,} destination nodes "
            "could not be matched to graph coordinates."
        )

    numeric_columns = [
        "destination_weight",
        "longitude",
        "latitude",
        "x_m",
        "y_m",
    ]

    numeric_values = prepared[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Prepared destination coordinates and "
            "weights must be finite."
        )

    return prepared.reset_index(drop=True)


def _prepared_candidate_view(
    origin_node: str,
    prepared_pool: pd.DataFrame,
    node_lookup: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return candidates and distances without a pandas merge."""
    if origin_node not in node_lookup.index:
        raise ValueError(
            f"Origin node {origin_node!r} is not "
            "present in the graph."
        )

    if prepared_pool.empty:
        raise ValueError(
            "Destination pool contains no candidates."
        )

    if len(prepared_pool) > 1:
        candidates = prepared_pool.loc[
            prepared_pool["node_id"].ne(
                origin_node
            )
        ]
    else:
        candidates = prepared_pool

    if candidates.empty:
        candidates = prepared_pool

    origin = node_lookup.loc[origin_node]

    distances = np.hypot(
        candidates["x_m"].to_numpy(dtype=float)
        - float(origin["x_m"]),
        candidates["y_m"].to_numpy(dtype=float)
        - float(origin["y_m"]),
    )

    if not np.isfinite(distances).all():
        raise ValueError(
            "Candidate distances must be finite."
        )

    return candidates, distances


def _draw_destination_counts_from_prepared_pool(
    origin_node: str,
    prepared_pool: pd.DataFrame,
    node_lookup: pd.DataFrame,
    selection_method: str,
    draw_count: int,
    rng: np.random.Generator,
    lognormal_scale_m: float,
    lognormal_sigma: float,
) -> pd.DataFrame:
    """Draw destinations from a prejoined coordinate table."""
    if draw_count <= 0:
        raise ValueError(
            "draw_count must be greater than zero."
        )

    if (
        not math.isfinite(lognormal_scale_m)
        or lognormal_scale_m <= 0
    ):
        raise ValueError(
            "lognormal_scale_m must be positive "
            "and finite."
        )

    if (
        not math.isfinite(lognormal_sigma)
        or lognormal_sigma <= 0
    ):
        raise ValueError(
            "lognormal_sigma must be positive "
            "and finite."
        )

    candidates, distances = (
        _prepared_candidate_view(
            origin_node,
            prepared_pool,
            node_lookup,
        )
    )

    if selection_method == "nearest":
        node_ids = candidates[
            "node_id"
        ].astype(str).to_numpy()

        selected_position = int(
            np.lexsort(
                (
                    node_ids,
                    distances,
                )
            )[0]
        )

        selected = candidates.iloc[
            [selected_position]
        ].copy()

        selected[
            "straight_line_distance_m"
        ] = distances[selected_position]

        selected["count"] = int(draw_count)

        return selected

    if selection_method != "lognormal_weighted":
        raise ValueError(
            f"Unsupported selection method "
            f"{selection_method!r}."
        )

    positive_distances = np.maximum(
        distances,
        1.0,
    )

    distance_density = lognorm.pdf(
        positive_distances,
        s=float(lognormal_sigma),
        scale=float(lognormal_scale_m),
    )

    weights = (
        candidates[
            "destination_weight"
        ].to_numpy(dtype=float)
        * distance_density
    )

    total = float(weights.sum())

    if not math.isfinite(total) or total <= 0:
        weights = candidates[
            "destination_weight"
        ].to_numpy(dtype=float)

        total = float(weights.sum())

    if not math.isfinite(total) or total <= 0:
        raise ValueError(
            "Destination sampling weights do not "
            "have a positive sum."
        )

    draws = rng.multinomial(
        int(draw_count),
        weights / total,
    )

    selected_positions = np.flatnonzero(
        draws > 0
    )

    selected = candidates.iloc[
        selected_positions
    ].copy()

    selected[
        "straight_line_distance_m"
    ] = distances[selected_positions]

    selected["count"] = draws[
        selected_positions
    ].astype(int)

    return selected


def _draw_destination_counts(
    origin_node: str,
    destination_pool: pd.DataFrame,
    node_lookup: pd.DataFrame,
    selection_method: str,
    draw_count: int,
    rng: np.random.Generator,
    lognormal_scale_m: float,
    lognormal_sigma: float,
) -> pd.DataFrame:
    """Draw and aggregate destinations for one origin."""
    if draw_count <= 0:
        raise ValueError(
            "draw_count must be greater than zero."
        )

    candidates = _candidate_table(
        origin_node,
        destination_pool,
        node_lookup,
    )

    if selection_method == "nearest":
        selected = (
            candidates.sort_values(
                [
                    "straight_line_distance_m",
                    "node_id",
                ],
                kind="mergesort",
            )
            .iloc[[0]]
            .copy()
        )

        selected["count"] = int(draw_count)
        return selected

    if selection_method != "lognormal_weighted":
        raise ValueError(
            f"Unsupported selection method "
            f"{selection_method!r}."
        )

    distances = np.maximum(
        candidates[
            "straight_line_distance_m"
        ].to_numpy(dtype=float),
        1.0,
    )

    distance_density = lognorm.pdf(
        distances,
        s=float(lognormal_sigma),
        scale=float(lognormal_scale_m),
    )

    weights = (
        candidates[
            "destination_weight"
        ].to_numpy(dtype=float)
        * distance_density
    )

    total = float(weights.sum())

    if not math.isfinite(total) or total <= 0:
        weights = candidates[
            "destination_weight"
        ].to_numpy(dtype=float)

        total = float(weights.sum())

    if not math.isfinite(total) or total <= 0:
        raise ValueError(
            "Destination sampling weights do not "
            "have a positive sum."
        )

    draws = rng.multinomial(
        int(draw_count),
        weights / total,
    )

    selected_mask = draws > 0
    selected = candidates.loc[selected_mask].copy()
    selected["count"] = draws[selected_mask].astype(int)

    return selected


def generate_trip_pairs(
    nodes: pd.DataFrame,
    origins: pd.DataFrame,
    opportunities: pd.DataFrame,
    trip_counts: dict[str, int],
    seed: int = 26,
    lognormal_scale_m: float = 2414.0,
    lognormal_sigma: float = 0.7899,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate category-specific, aggregated OD demand."""
    if not trip_counts:
        raise ValueError(
            "At least one trip count must be supplied."
        )

    invalid_counts = {
        trip_type: count
        for trip_type, count in trip_counts.items()
        if (
            trip_type not in TRIP_TYPES
            or not isinstance(count, int)
            or count <= 0
        )
    }

    if invalid_counts:
        raise ValueError(
            f"Invalid trip counts: {invalid_counts}"
        )

    require_columns(
        nodes,
        {
            "node_id",
            "longitude",
            "latitude",
            "x_m",
            "y_m",
        },
        "Graph node table",
    )

    require_columns(
        origins,
        {
            "node_id",
            "origin_population",
        },
        "Origin table",
    )

    node_lookup = nodes.set_index(
        "node_id",
        drop=False,
    )

    graph_node_ids = set(
        node_lookup.index.astype(str)
    )

    usable_origins = origins[
        origins["node_id"].isin(graph_node_ids)
        & origins["origin_population"].gt(0)
    ].copy()

    usable_origins = usable_origins.sort_values(
        "node_id"
    ).reset_index(drop=True)

    if usable_origins.empty:
        raise ValueError(
            "No usable population origins are "
            "present in the graph."
        )

    origin_probabilities = usable_origins[
        "origin_population"
    ].to_numpy(dtype=float)

    origin_probabilities /= origin_probabilities.sum()

    origin_population = (
        usable_origins.set_index(
            "node_id"
        )["origin_population"]
    )

    rng = np.random.default_rng(seed)

    destination_pools: dict[
        str,
        pd.DataFrame,
    ] = {}

    pool_diagnostics: dict[
        str,
        dict[str, Any],
    ] = {}

    rows: list[dict[str, Any]] = []
    sampled_origin_counts: dict[str, int] = {}
    empty_pools: list[str] = []

    for trip_type, requested_count in trip_counts.items():
        pool, pool_diagnostic = build_destination_pool(
            opportunities,
            trip_type,
            graph_node_ids,
        )

        destination_pools[trip_type] = pool
        pool_diagnostics[trip_type] = pool_diagnostic

        if pool.empty:
            empty_pools.append(trip_type)
            sampled_origin_counts[trip_type] = 0
            continue

        prepared_pool = _prepare_destination_pool(
            pool,
            node_lookup,
        )

        destination_pools[trip_type] = prepared_pool

        sampled_positions = rng.choice(
            len(usable_origins),
            size=int(requested_count),
            replace=True,
            p=origin_probabilities,
        )

        sampled_nodes = usable_origins.iloc[
            sampled_positions
        ]["node_id"].astype(str)

        origin_draw_counts = (
            sampled_nodes.value_counts(
                sort=False
            )
            .sort_index()
        )

        sampled_origin_counts[trip_type] = len(origin_draw_counts)

        selection_method = TRIP_TYPES[
            trip_type
        ]["selection_method"]

        for origin_node, draw_count in (
            origin_draw_counts.items()
        ):
            origin = node_lookup.loc[
                origin_node
            ]

            selected = (
                _draw_destination_counts_from_prepared_pool(
                    origin_node=str(origin_node),
                    prepared_pool=prepared_pool,
                    node_lookup=node_lookup,
                    selection_method=selection_method,
                    draw_count=int(draw_count),
                    rng=rng,
                    lognormal_scale_m=lognormal_scale_m,
                    lognormal_sigma=lognormal_sigma,
                )
            )

            for destination in selected.itertuples(
                index=False
            ):
                rows.append(
                    {
                        "origin_node": str(
                            origin_node
                        ),
                        "destination_node": str(
                            destination.node_id
                        ),
                        "trip_type": trip_type,
                        "selection_method": (
                            selection_method
                        ),
                        "origin_population": float(
                            origin_population.loc[
                                origin_node
                            ]
                        ),
                        "destination_weight": float(
                            destination.destination_weight
                        ),
                        "straight_line_distance_m": float(
                            destination
                            .straight_line_distance_m
                        ),
                        "origin_longitude": float(
                            origin["longitude"]
                        ),
                        "origin_latitude": float(
                            origin["latitude"]
                        ),
                        "destination_longitude": float(
                            destination.longitude
                        ),
                        "destination_latitude": float(
                            destination.latitude
                        ),
                        "count": int(
                            destination.count
                        ),
                        "seed": int(seed),
                    }
                )

    trips = pd.DataFrame(
        rows,
        columns=OUTPUT_COLUMNS,
    )

    generated_by_type = {
        trip_type: int(
            trips.loc[
                trips["trip_type"].eq(
                    trip_type
                ),
                "count",
            ].sum()
        )
        if not trips.empty
        else 0
        for trip_type in trip_counts
    }

    diagnostics = {
        "seed": int(seed),
        "requested_trip_counts": {
            key: int(value)
            for key, value in trip_counts.items()
        },
        "generated_trip_counts": generated_by_type,
        "requested_total_trips": int(
            sum(trip_counts.values())
        ),
        "generated_total_trips": int(
            trips["count"].sum()
            if not trips.empty
            else 0
        ),
        "unique_od_rows": len(trips),
        "unique_sampled_origins_by_type": (
            sampled_origin_counts
        ),
        "lognormal_scale_m": float(
            lognormal_scale_m
        ),
        "lognormal_sigma": float(
            lognormal_sigma
        ),
        "empty_destination_pools": empty_pools,
        "destination_pools": pool_diagnostics,
    }

    return trips, diagnostics

def write_outputs(
    trips: pd.DataFrame,
    diagnostics: dict[str, Any],
    output_path: Path,
    diagnostics_path: Path,
) -> None:
    """Write OD pairs and diagnostics atomically."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_temporary = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    diagnostics_temporary = (
        diagnostics_path.with_suffix(
            diagnostics_path.suffix + ".tmp"
        )
    )

    trips.to_csv(
        output_temporary,
        index=False,
    )

    diagnostics_temporary.write_text(
        json.dumps(
            diagnostics,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    output_temporary.replace(
        output_path
    )

    diagnostics_temporary.replace(
        diagnostics_path
    )



def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible population-weighted "
            "origin-destination demand."
        )
    )

    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--allocation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--opportunities",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--diagnostics",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--demand-config",
        type=Path,
        default=DEFAULT_DEMAND_CONFIG,
    )

    parser.add_argument(
        "--demand-scenario",
        default="1",
        help=(
            "Scenario column in the demand "
            "configuration CSV."
        ),
    )

    parser.add_argument(
        "--trip-types",
        nargs="+",
        choices=sorted(TRIP_TYPES),
        default=None,
        help=(
            "Optional subset of configured trip types."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=26,
    )

    parser.add_argument(
        "--projected-crs",
        default="EPSG:26986",
        help=(
            "Projected CRS used for "
            "straight-line distances."
        ),
    )

    parser.add_argument(
        "--lognormal-scale-m",
        type=float,
        default=2414.0,
        help=(
            "Lognormal distance scale in metres."
        ),
    )

    parser.add_argument(
        "--lognormal-sigma",
        type=float,
        default=0.7899,
        help=(
            "Lognormal shape parameter used for "
            "distance-weighted destination choice."
        ),
    )

    return parser


def main() -> None:
    """Run the command-line workflow."""
    args = build_parser().parse_args()

    trip_counts = load_demand_scenario(
        args.demand_config,
        args.demand_scenario,
    )

    if args.trip_types is not None:
        requested_types = list(
            dict.fromkeys(args.trip_types)
        )

        trip_counts = {
            trip_type: trip_counts[trip_type]
            for trip_type in requested_types
            if trip_type in trip_counts
        }

        if not trip_counts:
            raise ValueError(
                "None of the selected trip types "
                "has a positive count in the chosen "
                "demand scenario."
            )

    nodes = load_graph_node_table(
        args.graph,
        args.projected_crs,
    )

    graph_node_ids = set(
        nodes["node_id"]
    )

    origins, origin_diagnostics = (
        load_population_origins(
            args.allocation,
            graph_node_ids,
        )
    )

    opportunities = load_opportunities(
        args.opportunities
    )

    trips, diagnostics = generate_trip_pairs(
        nodes=nodes,
        origins=origins,
        opportunities=opportunities,
        trip_counts=trip_counts,
        seed=args.seed,
        lognormal_scale_m=(
            args.lognormal_scale_m
        ),
        lognormal_sigma=args.lognormal_sigma,
    )

    diagnostics["projected_crs"] = (
        args.projected_crs
    )

    diagnostics["demand_config"] = str(
        args.demand_config
    )

    diagnostics["demand_scenario"] = str(
        args.demand_scenario
    )

    diagnostics["origins"] = (
        origin_diagnostics
    )

    write_outputs(
        trips,
        diagnostics,
        args.output,
        args.diagnostics,
    )

    total_trips = int(
        trips["count"].sum()
        if not trips.empty
        else 0
    )

    print(
        f"Generated {total_trips:,} modeled trips "
        f"across {len(trips):,} unique OD rows."
    )

    print(
        f"Wrote trips to {args.output}"
    )

    print(
        "Wrote diagnostics to "
        f"{args.diagnostics}"
    )

if __name__ == "__main__":
    main()
