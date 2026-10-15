import math

import networkx as nx
import osmnx as ox
import pandas as pd
import pytest

from accessibility.network import build_accessibility_graph


def make_graph(
    *,
    reversed_value=False,
    length=100.0,
):
    graph = nx.MultiDiGraph()
    graph.add_node(1)
    graph.add_node(2)
    graph.add_edge(
        1,
        2,
        key=0,
        length=length,
        reversed=reversed_value,
    )
    return graph


def make_lts_row(**overrides):
    row = {
        "u": 1,
        "v": 2,
        "key": 0,
        "LTS_fwd": 2,
        "LTS_rev": 4,
        "bike_allowed_fwd": True,
        "bike_allowed_rev": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_forward_edge_uses_forward_lts_and_travel_time():
    graph, diagnostics = build_accessibility_graph(
        make_graph(reversed_value=False),
        make_lts_row(),
        bike_speed_kph=15.0,
    )

    edge = graph[1][2][0]

    assert edge["directional_lts"] == 2
    assert edge["bike_allowed"] is True
    assert edge["lts_direction"] == "fwd"
    assert edge["travel_time_seconds"] == pytest.approx(
        24.0
    )
    assert diagnostics.iloc[0]["status"] == "kept"


def test_reversed_edge_uses_reverse_fields():
    graph, diagnostics = build_accessibility_graph(
        make_graph(reversed_value=True),
        make_lts_row(
            LTS_fwd=1,
            LTS_rev=3,
            bike_allowed_fwd=False,
            bike_allowed_rev=True,
        ),
    )

    assert graph[1][2][0]["directional_lts"] == 3
    assert graph[1][2][0]["lts_direction"] == "rev"
    assert diagnostics.iloc[0]["direction"] == "rev"


def test_lts_row_reversed_value_takes_precedence():
    graph, _ = build_accessibility_graph(
        make_graph(reversed_value=False),
        make_lts_row(
            reversed=True,
            LTS_fwd=1,
            LTS_rev=4,
        ),
    )

    assert graph[1][2][0]["directional_lts"] == 4
    assert graph[1][2][0]["lts_direction"] == "rev"


def test_disallowed_and_invalid_lts_edges_are_removed():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        1,
        2,
        key=0,
        length=50,
        reversed=False,
    )
    graph.add_edge(
        2,
        3,
        key=0,
        length=50,
        reversed=False,
    )

    lts = pd.DataFrame(
        [
            {
                "u": 1,
                "v": 2,
                "key": 0,
                "LTS_fwd": 2,
                "LTS_rev": 2,
                "bike_allowed_fwd": False,
                "bike_allowed_rev": False,
            },
            {
                "u": 2,
                "v": 3,
                "key": 0,
                "LTS_fwd": 0,
                "LTS_rev": 0,
                "bike_allowed_fwd": True,
                "bike_allowed_rev": True,
            },
        ]
    )

    output, diagnostics = build_accessibility_graph(
        graph,
        lts,
    )

    assert output.number_of_edges() == 0

    assert set(diagnostics["status"]) == {
        "bicycle_not_allowed",
        "invalid_directional_lts",
    }


def test_edge_without_lts_row_is_removed_and_reported():
    output, diagnostics = build_accessibility_graph(
        make_graph(),
        make_lts_row().iloc[0:0],
    )

    assert output.number_of_edges() == 0
    assert (
        diagnostics.iloc[0]["status"]
        == "missing_lts_row"
    )


def test_duplicate_lts_edge_ids_raise_error():
    lts = pd.concat(
        [
            make_lts_row(),
            make_lts_row(),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_accessibility_graph(
            make_graph(),
            lts,
        )


def test_invalid_speed_and_length_raise_errors():
    with pytest.raises(
        ValueError,
        match="bike_speed_kph",
    ):
        build_accessibility_graph(
            make_graph(),
            make_lts_row(),
            bike_speed_kph=0,
        )

    with pytest.raises(ValueError, match="length"):
        build_accessibility_graph(
            make_graph(length=-1),
            make_lts_row(),
        )


def test_parallel_edges_are_matched_by_key():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        1,
        2,
        key=0,
        length=100,
        reversed=False,
    )
    graph.add_edge(
        1,
        2,
        key=1,
        length=200,
        reversed=False,
    )

    lts = pd.DataFrame(
        [
            {
                "u": 1,
                "v": 2,
                "key": 0,
                "LTS_fwd": 1,
                "LTS_rev": 1,
                "bike_allowed_fwd": True,
                "bike_allowed_rev": True,
            },
            {
                "u": 1,
                "v": 2,
                "key": 1,
                "LTS_fwd": 4,
                "LTS_rev": 4,
                "bike_allowed_fwd": True,
                "bike_allowed_rev": True,
            },
        ]
    )

    output, _ = build_accessibility_graph(
        graph,
        lts,
    )

    assert output[1][2][0]["directional_lts"] == 1
    assert output[1][2][1]["directional_lts"] == 4

    assert math.isclose(
        output[1][2][1]["travel_time_seconds"],
        48.0,
    )


def test_osm_tags_survive_graphml_round_trip(
    tmp_path,
) -> None:
    """Existing OSM way tags survive graph preparation and saving."""
    graph = make_graph()

    graph.graph["crs"] = "EPSG:4326"

    graph.nodes[1].update(
        x=-71.100,
        y=42.350,
    )

    graph.nodes[2].update(
        x=-71.090,
        y=42.350,
    )

    expected_tags = {
        "highway": "residential",
        "name": "Test Street",
        "cycleway": "lane",
        "surface": "asphalt",
    }

    graph[1][2][0].update(expected_tags)

    output, diagnostics = build_accessibility_graph(
        graph,
        make_lts_row(),
    )

    assert diagnostics.iloc[0]["status"] == "kept"

    prepared_edge = output[1][2][0]

    for attribute, expected in expected_tags.items():
        assert prepared_edge[attribute] == expected

    graph_path = tmp_path / "accessibility.graphml"

    ox.save_graphml(
        output,
        filepath=graph_path,
    )

    reloaded = ox.load_graphml(graph_path)

    source_node = 1 if 1 in reloaded else "1"
    target_node = 2 if 2 in reloaded else "2"

    edge_collection = reloaded.get_edge_data(
        source_node,
        target_node,
    )

    assert edge_collection is not None

    reloaded_edge = next(
        iter(edge_collection.values())
    )

    for attribute, expected in expected_tags.items():
        assert str(reloaded_edge[attribute]) == expected

    assert int(reloaded_edge["directional_lts"]) == 2
    assert float(
        reloaded_edge["travel_time_seconds"]
    ) == pytest.approx(24.0)
