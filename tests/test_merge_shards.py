from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from accessibility.merge_shards import (
    expected_shard_ranges,
    validate_and_merge_shards,
)
from accessibility.run_batch import run_batch


def write_graph(path: Path) -> None:
    graph = nx.MultiDiGraph()
    nodes = ("A", "B", "C", "D", "E")

    graph.add_nodes_from(nodes)

    for source, target in zip(nodes, nodes[1:]):
        attributes = {
            "length": "10",
            "cost_typical_adult_Baseline": "12",
            "cost_low_confidence_adult_Baseline": "14",
            "cost_child_Baseline": "16",
            "max_lts": "2",
        }

        graph.add_edge(
            source,
            target,
            **attributes,
        )

        graph.add_edge(
            target,
            source,
            **attributes,
        )

    nx.write_graphml(graph, path)


def write_opportunities(path: Path) -> None:
    pd.DataFrame(
        {
            "node_id": ["C", "A", "E", "B", "D"],
            "longitude": [
                -70.8,
                -71.0,
                -70.6,
                -70.9,
                -70.7,
            ],
            "latitude": [
                42.0,
                42.0,
                42.0,
                42.0,
                42.0,
            ],
            "jobs": [30, 10, 50, 20, 40],
            "schools": [0, 1, 0, 1, 0],
            "transit_stations": [0, 0, 1, 0, 0],
            "bus_stops": [1, 0, 0, 1, 0],
            "greenspace": [0, 1, 0, 0, 1],
            "healthcare": [0, 0, 1, 0, 0],
            "stores": [1, 0, 0, 1, 0],
        }
    ).to_csv(path, index=False)


def create_valid_shards(tmp_path: Path):
    graph_path = tmp_path / "graph.graphml"
    opportunities_path = tmp_path / "opportunities.csv"
    shard_dir = tmp_path / "shards"

    shard_dir.mkdir()

    write_graph(graph_path)
    write_opportunities(opportunities_path)

    run_batch(
        graph_path,
        opportunities_path,
        shard_dir / "profile_access_shard_00.csv",
        cutoff_miles=0.1,
        start_index=0,
        end_index=3,
        progress_every=10,
    )

    run_batch(
        graph_path,
        opportunities_path,
        shard_dir / "profile_access_shard_01.csv",
        cutoff_miles=0.1,
        start_index=3,
        end_index=5,
        progress_every=10,
    )

    return opportunities_path, shard_dir


def test_expected_shard_ranges():
    assert expected_shard_ranges(5, 2) == (
        (0, 3),
        (3, 5),
    )

    ranges = expected_shard_ranges(96_232, 32)

    assert ranges[0] == (0, 3008)
    assert ranges[-1] == (93_248, 96_232)


def test_validates_and_merges_shards(tmp_path):
    opportunities_path, shard_dir = (
        create_valid_shards(tmp_path)
    )

    output_path = tmp_path / "merged.csv"
    diagnostics_path = tmp_path / "diagnostics.json"

    merged = validate_and_merge_shards(
        opportunities_path,
        shard_dir,
        output_path,
        num_shards=2,
        cutoff_miles=0.1,
        diagnostics_path=diagnostics_path,
    )

    saved = pd.read_csv(
        output_path,
        dtype={"origin_node": str},
    )

    assert len(merged) == 5
    assert len(saved) == 5

    assert saved["origin_node"].tolist() == [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]

    assert saved["origin_node"].is_unique
    assert diagnostics_path.is_file()


def test_missing_shard_is_rejected(tmp_path):
    opportunities_path, shard_dir = (
        create_valid_shards(tmp_path)
    )

    (
        shard_dir
        / "profile_access_shard_01.csv"
    ).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="Missing shard file",
    ):
        validate_and_merge_shards(
            opportunities_path,
            shard_dir,
            tmp_path / "merged.csv",
            num_shards=2,
            cutoff_miles=0.1,
        )


def test_wrong_origin_order_is_rejected(tmp_path):
    opportunities_path, shard_dir = (
        create_valid_shards(tmp_path)
    )

    shard_path = (
        shard_dir
        / "profile_access_shard_00.csv"
    )

    shard = pd.read_csv(
        shard_path,
        dtype={"origin_node": str},
    )

    shard = shard.iloc[::-1].reset_index(drop=True)
    shard.to_csv(shard_path, index=False)

    with pytest.raises(
        ValueError,
        match="origin ordering mismatch",
    ):
        validate_and_merge_shards(
            opportunities_path,
            shard_dir,
            tmp_path / "merged.csv",
            num_shards=2,
            cutoff_miles=0.1,
        )


def test_incorrect_relative_value_is_rejected(tmp_path):
    opportunities_path, shard_dir = (
        create_valid_shards(tmp_path)
    )

    shard_path = (
        shard_dir
        / "profile_access_shard_00.csv"
    )

    shard = pd.read_csv(
        shard_path,
        dtype={"origin_node": str},
    )

    shard.loc[
        0,
        "typical_adult_jobs_relative",
    ] = 9.0

    shard.to_csv(shard_path, index=False)

    with pytest.raises(ValueError):
        validate_and_merge_shards(
            opportunities_path,
            shard_dir,
            tmp_path / "merged.csv",
            num_shards=2,
            cutoff_miles=0.1,
        )
