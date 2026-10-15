import json

import networkx as nx
import pandas as pd

from accessibility.build_opportunities import (
    build_opportunity_table,
)


def write_graph(path, nodes):
    graph = nx.DiGraph()

    for node_id, longitude, latitude in nodes:
        graph.add_node(
            str(node_id),
            x=float(longitude),
            y=float(latitude),
        )

    for first, second in zip(nodes, nodes[1:]):
        graph.add_edge(
            str(first[0]),
            str(second[0]),
            length=10.0,
        )

    nx.write_graphml(graph, path)


def test_builds_node_opportunity_table(tmp_path):
    pruned_graph = tmp_path / "pruned.graphml"
    source_graph = tmp_path / "source.graphml"

    write_graph(
        pruned_graph,
        [
            ("1", -71.0000, 42.0000),
            ("2", -70.9999, 42.0000),
            ("3", -70.9998, 42.0000),
        ],
    )

    write_graph(
        source_graph,
        [
            ("1", -71.0000, 42.0000),
            ("2", -70.9999, 42.0000),
            ("3", -70.9998, 42.0000),
            ("4", -70.9997, 42.0000),
        ],
    )

    destinations = tmp_path / "destinations.csv"

    pd.DataFrame(
        {
            "destination_id": ["D1", "D2", "D3"],
            "display_category": [
                "Schools",
                "Stores",
                "Bus stops",
            ],
            "nearest_node_id": ["2", "2", "3"],
            "snap_distance_meters": [10.0, 20.0, 5.0],
        }
    ).to_csv(destinations, index=False)

    lodes = tmp_path / "lodes.csv"

    pd.DataFrame(
        {
            "origin_node": ["1", "1", "2"],
            "destination_node": ["2", "2", "4"],
            "employees": [5, 3, 7],
        }
    ).to_csv(lodes, index=False)

    output_path = tmp_path / "opportunities.csv"
    diagnostics_path = tmp_path / "diagnostics.json"
    reassignment_path = tmp_path / "reassignments.csv"

    output = build_opportunity_table(
        graph_path=pruned_graph,
        source_graph_path=source_graph,
        destinations_path=destinations,
        lodes_path=lodes,
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        reassignment_path=reassignment_path,
        max_job_reassignment_distance_m=100.0,
    )

    output = output.set_index("node_id")

    assert len(output) == 3
    assert output["jobs"].sum() == 15
    assert output.loc["2", "jobs"] == 8
    assert output.loc["3", "jobs"] == 7

    assert output.loc["2", "schools"] == 1
    assert output.loc["2", "stores"] == 1
    assert output.loc["3", "bus_stops"] == 1

    diagnostics = json.loads(
        diagnostics_path.read_text()
    )

    assert diagnostics["jobs"]["source_jobs"] == 15
    assert diagnostics["jobs"]["reassigned_jobs"] == 7
    assert diagnostics["jobs"]["excluded_jobs"] == 0


def test_excludes_removed_job_node_beyond_limit(tmp_path):
    pruned_graph = tmp_path / "pruned.graphml"
    source_graph = tmp_path / "source.graphml"

    write_graph(
        pruned_graph,
        [
            ("1", -71.0000, 42.0000),
            ("2", -70.9999, 42.0000),
        ],
    )

    write_graph(
        source_graph,
        [
            ("1", -71.0000, 42.0000),
            ("2", -70.9999, 42.0000),
            ("9", -70.9900, 42.0000),
        ],
    )

    destinations = tmp_path / "destinations.csv"

    pd.DataFrame(
        {
            "destination_id": ["D1"],
            "display_category": ["Schools"],
            "nearest_node_id": ["2"],
            "snap_distance_meters": [10.0],
        }
    ).to_csv(destinations, index=False)

    lodes = tmp_path / "lodes.csv"

    pd.DataFrame(
        {
            "origin_node": ["1"],
            "destination_node": ["9"],
            "employees": [12],
        }
    ).to_csv(lodes, index=False)

    output_path = tmp_path / "opportunities.csv"
    diagnostics_path = tmp_path / "diagnostics.json"
    reassignment_path = tmp_path / "reassignments.csv"

    output = build_opportunity_table(
        graph_path=pruned_graph,
        source_graph_path=source_graph,
        destinations_path=destinations,
        lodes_path=lodes,
        output_path=output_path,
        diagnostics_path=diagnostics_path,
        reassignment_path=reassignment_path,
        max_job_reassignment_distance_m=100.0,
    )

    assert output["jobs"].sum() == 0

    diagnostics = json.loads(
        diagnostics_path.read_text()
    )

    assert diagnostics["jobs"]["reassigned_jobs"] == 0
    assert diagnostics["jobs"]["excluded_jobs"] == 12
