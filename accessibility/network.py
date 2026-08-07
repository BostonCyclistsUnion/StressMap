"""Build a directed bicycle-routing graph from StressMap LTS outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
import pandas as pd


REQUIRED_LTS_COLUMNS = {
    "u",
    "v",
    "key",
    "LTS_fwd",
    "LTS_rev",
    "bike_allowed_fwd",
    "bike_allowed_rev",
}

DIAGNOSTIC_COLUMNS = [
    "u",
    "v",
    "key",
    "status",
    "direction",
    "directional_lts",
    "bike_allowed",
    "travel_time_seconds",
]


def _coerce_bool(value: Any) -> bool:
    """Convert common CSV and GraphML boolean representations to bool."""
    if value is None or pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    normalized = str(value).strip().lower()

    if normalized in {"true", "t", "yes", "y", "1"}:
        return True

    if normalized in {"false", "f", "no", "n", "0", ""}:
        return False

    raise ValueError(
        "Cannot interpret bicycle-access value as boolean: "
        f"{value!r}"
    )


def _edge_is_reversed(
    lts_row: pd.Series,
    edge_data: dict[str, Any],
) -> bool:
    """Return whether an edge runs opposite its original OSM way."""
    if (
        "reversed" in lts_row.index
        and not pd.isna(lts_row["reversed"])
    ):
        return _coerce_bool(lts_row["reversed"])

    return _coerce_bool(edge_data.get("reversed", False))


def _normalize_edge_columns(
    lts: pd.DataFrame,
) -> pd.DataFrame:
    """Validate edge identifiers and return a normalized copy."""
    missing = REQUIRED_LTS_COLUMNS.difference(lts.columns)

    if missing:
        raise ValueError(
            f"LTS data is missing columns: {sorted(missing)}"
        )

    normalized = lts.copy()

    for column in ("u", "v", "key"):
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="raise",
        ).astype("int64")

    duplicate_mask = normalized.duplicated(
        ["u", "v", "key"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_keys = (
            normalized.loc[
                duplicate_mask,
                ["u", "v", "key"],
            ]
            .drop_duplicates()
            .sort_values(["u", "v", "key"])
            .to_dict("records")
        )

        raise ValueError(
            "LTS data contains duplicate (u, v, key) rows: "
            f"{duplicate_keys[:10]}"
        )

    return normalized


def load_lts_edges(path: str | Path) -> pd.DataFrame:
    """Load and validate a StressMap *_4_all_lts.csv file."""
    return _normalize_edge_columns(
        pd.read_csv(path, low_memory=False)
    )


def build_accessibility_graph(
    graph: nx.MultiDiGraph,
    lts_edges: pd.DataFrame,
    *,
    bike_speed_kph: float = 15.0,
) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    """
    Attach directional LTS and travel time.

    Edges are removed when:
    - no matching LTS row exists,
    - bicycle travel is not allowed in the edge direction,
    - or directional LTS is missing or outside 1 through 4.
    """
    if (
        not math.isfinite(bike_speed_kph)
        or bike_speed_kph <= 0
    ):
        raise ValueError(
            "bike_speed_kph must be positive and finite."
        )

    lts = _normalize_edge_columns(lts_edges)
    lts_index = lts.set_index(
        ["u", "v", "key"],
        verify_integrity=True,
    )

    output = graph.copy()
    meters_per_second = bike_speed_kph * 1000.0 / 3600.0
    diagnostics: list[dict[str, Any]] = []

    for u, v, key, edge_data in list(
        output.edges(keys=True, data=True)
    ):
        edge_id = (int(u), int(v), int(key))

        diagnostic: dict[str, Any] = {
            "u": edge_id[0],
            "v": edge_id[1],
            "key": edge_id[2],
            "status": "",
            "direction": "",
            "directional_lts": pd.NA,
            "bike_allowed": False,
            "travel_time_seconds": pd.NA,
        }

        if edge_id not in lts_index.index:
            diagnostic["status"] = "missing_lts_row"
            diagnostics.append(diagnostic)
            output.remove_edge(u, v, key)
            continue

        row = lts_index.loc[edge_id]

        direction = (
            "rev"
            if _edge_is_reversed(row, edge_data)
            else "fwd"
        )
        diagnostic["direction"] = direction

        allowed = _coerce_bool(
            row[f"bike_allowed_{direction}"]
        )
        diagnostic["bike_allowed"] = allowed

        if not allowed:
            diagnostic["status"] = "bicycle_not_allowed"
            diagnostics.append(diagnostic)
            output.remove_edge(u, v, key)
            continue

        directional_lts = pd.to_numeric(
            pd.Series([row[f"LTS_{direction}"]]),
            errors="coerce",
        ).iloc[0]

        if (
            pd.isna(directional_lts)
            or not math.isfinite(float(directional_lts))
            or not 1 <= float(directional_lts) <= 4
        ):
            diagnostic["status"] = "invalid_directional_lts"
            diagnostics.append(diagnostic)
            output.remove_edge(u, v, key)
            continue

        try:
            length_m = float(edge_data["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Edge {edge_id} has no valid length attribute."
            ) from exc

        if not math.isfinite(length_m) or length_m < 0:
            raise ValueError(
                f"Edge {edge_id} has an invalid length."
            )

        travel_time_seconds = length_m / meters_per_second
        directional_lts_int = int(float(directional_lts))

        edge_data["directional_lts"] = directional_lts_int
        edge_data["bike_allowed"] = True
        edge_data["travel_time_seconds"] = float(
            travel_time_seconds
        )
        edge_data["bike_speed_kph"] = float(bike_speed_kph)
        edge_data["lts_direction"] = direction

        diagnostic["status"] = "kept"
        diagnostic["directional_lts"] = (
            directional_lts_int
        )
        diagnostic["travel_time_seconds"] = float(
            travel_time_seconds
        )
        diagnostics.append(diagnostic)

    diagnostics_df = pd.DataFrame(
        diagnostics,
        columns=DIAGNOSTIC_COLUMNS,
    )

    if not diagnostics_df.empty:
        diagnostics_df = diagnostics_df.sort_values(
            ["u", "v", "key"]
        ).reset_index(drop=True)

    output.graph["bike_speed_kph"] = float(
        bike_speed_kph
    )
    output.graph["accessibility_edge_count"] = (
        output.number_of_edges()
    )

    return output, diagnostics_df


def build_accessibility_graph_from_files(
    graph_path: str | Path,
    lts_csv_path: str | Path,
    *,
    bike_speed_kph: float = 15.0,
) -> tuple[nx.MultiDiGraph, pd.DataFrame]:
    """Load source files and construct the accessibility graph."""
    graph = ox.load_graphml(graph_path)
    lts = load_lts_edges(lts_csv_path)

    return build_accessibility_graph(
        graph,
        lts,
        bike_speed_kph=bike_speed_kph,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a directed bicycle-routing graph using "
            "StressMap directional LTS results."
        )
    )

    parser.add_argument(
        "--graph",
        required=True,
        type=Path,
        help="Input StressMap GraphML file.",
    )

    parser.add_argument(
        "--lts-csv",
        required=True,
        type=Path,
        help="Input StressMap *_4_all_lts.csv file.",
    )

    parser.add_argument(
        "--output-graph",
        required=True,
        type=Path,
        help="Output LTS-aware GraphML file.",
    )

    parser.add_argument(
        "--diagnostics",
        required=True,
        type=Path,
        help="Output edge-diagnostics CSV file.",
    )

    parser.add_argument(
        "--bike-speed-kph",
        type=float,
        default=15.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    routing_graph, diagnostics = (
        build_accessibility_graph_from_files(
            args.graph,
            args.lts_csv,
            bike_speed_kph=args.bike_speed_kph,
        )
    )

    args.output_graph.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.diagnostics.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.save_graphml(
        routing_graph,
        args.output_graph,
    )
    diagnostics.to_csv(
        args.diagnostics,
        index=False,
    )

    counts = (
        diagnostics["status"]
        .value_counts()
        .sort_index()
    )

    print("Accessibility graph created.")
    print(f"Input edges: {len(diagnostics):,}")
    print(
        "Kept edges: "
        f"{routing_graph.number_of_edges():,}"
    )

    for status, count in counts.items():
        print(f"{status}: {count:,}")

    print(f"Saved graph: {args.output_graph}")
    print(f"Saved diagnostics: {args.diagnostics}")


if __name__ == "__main__":
    main()
