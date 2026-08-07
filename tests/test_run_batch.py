import csv

import networkx as nx
import pandas as pd
import pytest

from accessibility.run_batch import (
    load_origin_table,
    run_batch,
    select_origin_slice,
)


def write_graph(path):
    graph = nx.MultiDiGraph()

    for node in ("A", "B", "C"):
        graph.add_node(node)

    graph.add_edge(
        "A",
        "B",
        length="10",
        cost_typical_adult_Baseline="10",
        cost_low_confidence_adult_Baseline="10",
        cost_child_Baseline="10",
        max_lts="1",
    )

    graph.add_edge(
        "B",
        "C",
        length="10",
        cost_typical_adult_Baseline="20",
        cost_low_confidence_adult_Baseline="30",
        cost_child_Baseline="40",
        max_lts="2",
    )

    nx.write_graphml(graph, path)


def write_opportunities(path):
    pd.DataFrame(
        {
            "node_id": ["C", "A", "B"],
            "longitude": [-70.8, -71.0, -70.9],
            "latitude": [42.0, 42.0, 42.0],
            "jobs": [100, 1, 10],
            "schools": [1, 0, 1],
            "transit_stations": [1, 0, 0],
            "bus_stops": [0, 0, 1],
            "greenspace": [1, 0, 0],
            "healthcare": [1, 0, 0],
            "stores": [1, 0, 0],
        }
    ).to_csv(path, index=False)


def test_origin_slicing_is_deterministic(tmp_path):
    path = tmp_path / "opportunities.csv"
    write_opportunities(path)

    origins = load_origin_table(path)

    assert origins["node_id"].tolist() == [
        "A",
        "B",
        "C",
    ]

    selected = select_origin_slice(
        origins,
        start_index=1,
        end_index=3,
    )

    assert selected["node_id"].tolist() == [
        "B",
        "C",
    ]


def test_batch_resumes_without_duplicates(tmp_path):
    graph_path = tmp_path / "graph.graphml"
    opportunities_path = tmp_path / "opportunities.csv"
    output_path = tmp_path / "results.csv"

    write_graph(graph_path)
    write_opportunities(opportunities_path)

    first = run_batch(
        graph_path,
        opportunities_path,
        output_path,
        cutoff_miles=0.1,
        start_index=0,
        end_index=2,
    )

    assert first.written_count == 2

    second = run_batch(
        graph_path,
        opportunities_path,
        output_path,
        cutoff_miles=0.1,
        start_index=0,
        end_index=2,
    )

    assert second.written_count == 0
    assert second.previously_completed_count == 2

    third = run_batch(
        graph_path,
        opportunities_path,
        output_path,
        cutoff_miles=0.1,
        start_index=0,
        end_index=3,
    )

    assert third.written_count == 1

    result = pd.read_csv(
        output_path,
        dtype={"origin_node": str},
    )

    assert result["origin_node"].tolist() == [
        "A",
        "B",
        "C",
    ]

    assert result["origin_node"].is_unique


def test_duplicate_existing_origin_is_rejected(tmp_path):
    graph_path = tmp_path / "graph.graphml"
    opportunities_path = tmp_path / "opportunities.csv"
    output_path = tmp_path / "results.csv"

    write_graph(graph_path)
    write_opportunities(opportunities_path)

    run_batch(
        graph_path,
        opportunities_path,
        output_path,
        cutoff_miles=0.1,
        start_index=0,
        end_index=1,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(csv.reader(handle))

    with output_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        csv.writer(handle).writerow(rows[1])

    with pytest.raises(
        ValueError,
        match="duplicate origin",
    ):
        run_batch(
            graph_path,
            opportunities_path,
            output_path,
            cutoff_miles=0.1,
            start_index=0,
            end_index=2,
        )


def test_cutoff_mismatch_is_rejected(tmp_path):
    graph_path = tmp_path / "graph.graphml"
    opportunities_path = tmp_path / "opportunities.csv"
    output_path = tmp_path / "results.csv"

    write_graph(graph_path)
    write_opportunities(opportunities_path)

    run_batch(
        graph_path,
        opportunities_path,
        output_path,
        cutoff_miles=0.1,
        start_index=0,
        end_index=1,
    )

    with pytest.raises(
        ValueError,
        match="different cutoff_miles",
    ):
        run_batch(
            graph_path,
            opportunities_path,
            output_path,
            cutoff_miles=0.2,
            start_index=0,
            end_index=2,
        )
