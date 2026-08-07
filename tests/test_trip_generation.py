"""Tests for reproducible origin-destination trip generation."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from accessibility.trip_generation import (
    build_destination_pool,
    generate_trip_pairs,
    load_demand_scenario,
    load_graph_node_table,
    select_distance_weighted_destination,
    select_nearest_destination,
)


def make_nodes() -> pd.DataFrame:
    """Create a small projected graph-node table."""
    return pd.DataFrame(
        {
            "node_id": ["A", "B", "C", "D"],
            "longitude": [
                -71.100,
                -71.090,
                -71.070,
                -71.040,
            ],
            "latitude": [
                42.350,
                42.350,
                42.350,
                42.350,
            ],
            "x_m": [
                0.0,
                1000.0,
                3000.0,
                6000.0,
            ],
            "y_m": [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )


def make_opportunities() -> pd.DataFrame:
    """Create node-level destination opportunities."""
    return pd.DataFrame(
        {
            "node_id": ["A", "B", "C", "D"],
            "jobs": [0, 0, 4, 2],
            "schools": [0, 1, 0, 0],
            "transit_stations": [1, 0, 0, 0],
            "bus_stops": [2, 0, 4, 0],
            "greenspace": [0, 0, 1, 0],
            "healthcare": [0, 1, 0, 0],
            "stores": [0, 0, 0, 1],
        }
    )


def test_load_demand_scenario_maps_categories(tmp_path) -> None:
    """BCU demand categories map to StressMap trip types."""
    config = tmp_path / "demand.csv"

    config.write_text(
        (
            "scenario_id,1\n"
            "home_school,4\n"
            "home_office,5\n"
            "home_healthcare,2\n"
            "home_transit,3\n"
            "home_greenspace,1\n"
            "home_store,2\n"
            "TOTAL,17\n"
        ),
        encoding="utf-8",
    )

    counts = load_demand_scenario(
        config,
        "1",
    )

    assert counts == {
        "elementary_school": 4,
        "work": 5,
        "healthcare": 2,
        "transit": 3,
        "recreation": 1,
        "shopping": 2,
    }


def test_load_demand_scenario_rejects_total_mismatch(
    tmp_path,
) -> None:
    """A declared total must equal its category sum."""
    config = tmp_path / "demand.csv"

    config.write_text(
        (
            "scenario_id,1\n"
            "home_school,4\n"
            "home_office,5\n"
            "TOTAL,20\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="TOTAL does not equal",
    ):
        load_demand_scenario(
            config,
            "1",
        )


def test_build_destination_pool_combines_transit_columns() -> None:
    """Transit stations and bus stops form one pool."""
    opportunities = make_opportunities()

    pool, diagnostics = build_destination_pool(
        opportunities,
        "transit",
        {"A", "B", "C", "D"},
    )

    weights = (
        pool.set_index("node_id")[
            "destination_weight"
        ].to_dict()
    )

    assert weights == {
        "A": 3,
        "C": 4,
    }

    assert (
        diagnostics["usable_destination_nodes"]
        == 2
    )

    assert (
        diagnostics["total_destination_weight"]
        == pytest.approx(7.0)
    )


def test_nearest_destination_excludes_origin_when_possible() -> None:
    """Nearest selection avoids a self-trip when alternatives exist."""
    nodes = make_nodes()

    node_lookup = nodes.set_index(
        "node_id",
        drop=False,
    )

    pool = pd.DataFrame(
        {
            "node_id": ["B", "C"],
            "destination_weight": [1.0, 1.0],
        }
    )

    selected = select_nearest_destination(
        "B",
        pool,
        node_lookup,
    )

    assert selected["node_id"] == "C"

    assert selected[
        "straight_line_distance_m"
    ] == pytest.approx(2000.0)


def test_weighted_destination_is_reproducible() -> None:
    """Equal seeds produce the same weighted destination."""
    nodes = make_nodes()

    node_lookup = nodes.set_index(
        "node_id",
        drop=False,
    )

    pool = pd.DataFrame(
        {
            "node_id": ["B", "C", "D"],
            "destination_weight": [
                1.0,
                5.0,
                2.0,
            ],
        }
    )

    first = select_distance_weighted_destination(
        "A",
        pool,
        node_lookup,
        np.random.default_rng(26),
    )

    second = select_distance_weighted_destination(
        "A",
        pool,
        node_lookup,
        np.random.default_rng(26),
    )

    assert first["node_id"] == second["node_id"]

    assert first[
        "straight_line_distance_m"
    ] == pytest.approx(
        second["straight_line_distance_m"]
    )


def test_generate_trip_pairs_preserves_requested_counts() -> None:
    """Aggregated OD rows conserve every requested trip."""
    nodes = make_nodes()

    origins = pd.DataFrame(
        {
            "node_id": ["A", "B"],
            "origin_population": [
                100.0,
                25.0,
            ],
        }
    )

    trip_counts = {
        "elementary_school": 12,
        "work": 20,
    }

    trips, diagnostics = generate_trip_pairs(
        nodes=nodes,
        origins=origins,
        opportunities=make_opportunities(),
        trip_counts=trip_counts,
        seed=26,
    )

    generated_counts = (
        trips.groupby("trip_type")["count"]
        .sum()
        .astype(int)
        .to_dict()
    )

    assert generated_counts == trip_counts
    assert int(trips["count"].sum()) == 32

    assert (
        diagnostics["requested_total_trips"]
        == 32
    )

    assert (
        diagnostics["generated_total_trips"]
        == 32
    )

    assert (
        diagnostics["generated_trip_counts"]
        == trip_counts
    )

    assert (trips["count"] > 0).all()


def test_generate_trip_pairs_is_reproducible() -> None:
    """The same seed produces identical aggregated OD output."""
    nodes = make_nodes()

    origins = pd.DataFrame(
        {
            "node_id": ["A", "B"],
            "origin_population": [
                100.0,
                25.0,
            ],
        }
    )

    arguments = {
        "nodes": nodes,
        "origins": origins,
        "opportunities": make_opportunities(),
        "trip_counts": {
            "elementary_school": 12,
            "work": 20,
        },
        "seed": 26,
    }

    first, first_diagnostics = generate_trip_pairs(
        **arguments
    )

    second, second_diagnostics = generate_trip_pairs(
        **arguments
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

    assert (
        first_diagnostics
        == second_diagnostics
    )


def test_load_graph_node_table_reads_node_coordinates(
    tmp_path,
) -> None:
    """Graph loading reads coordinates without needing edges."""
    graph = nx.MultiDiGraph()

    graph.add_node(
        "A",
        x=-71.10,
        y=42.35,
    )

    graph.add_node(
        "B",
        x=-71.09,
        y=42.36,
    )

    graph.add_edge(
        "A",
        "B",
        length=1000.0,
    )

    graph_path = tmp_path / "graph.graphml"
    nx.write_graphml(graph, graph_path)

    nodes = load_graph_node_table(
        graph_path,
        "EPSG:26986",
    ).sort_values("node_id").reset_index(drop=True)

    assert nodes["node_id"].tolist() == ["A", "B"]

    assert nodes["longitude"].tolist() == pytest.approx(
        [-71.10, -71.09]
    )

    assert nodes["latitude"].tolist() == pytest.approx(
        [42.35, 42.36]
    )

    assert np.isfinite(
        nodes[["x_m", "y_m"]].to_numpy()
    ).all()
