import json

import pandas as pd
import pytest

from accessibility.population_weighting import (
    aggregate_node_population,
    attach_population_to_accessibility,
    build_population_attached_nodes,
    load_population_allocation,
)


def accessibility_frame():
    return pd.DataFrame(
        {
            "origin_node": ["A", "B", "C"],
            "longitude": [-71.0, -70.9, -70.8],
            "distance_jobs": [100, 200, 300],
        }
    )


def allocation_frame():
    return pd.DataFrame(
        {
            "node_id": ["A", "A", "B"],
            "GEOID": ["T1", "T2", "T1"],
            "assigned_population": [
                10.0,
                5.0,
                20.0,
            ],
        }
    )


def test_aggregates_and_attaches_population():
    allocation = allocation_frame()

    aggregated = aggregate_node_population(
        allocation
    )

    assert aggregated[
        "assigned_total_population"
    ].sum() == pytest.approx(35.0)

    joined, diagnostics = (
        attach_population_to_accessibility(
            accessibility_frame(),
            allocation,
        )
    )

    assert joined["origin_node"].tolist() == [
        "A",
        "B",
        "C",
    ]

    assert joined[
        "assigned_total_population"
    ].tolist() == [
        15.0,
        20.0,
        0.0,
    ]

    assert joined[
        "contributing_tract_count"
    ].tolist() == [
        2,
        1,
        0,
    ]

    assert diagnostics[
        "zero_allocation_node_count"
    ] == 1

    assert diagnostics[
        "multiple_tract_node_count"
    ] == 1


def test_unknown_allocation_node_is_rejected():
    allocation = allocation_frame()
    allocation.loc[len(allocation)] = [
        "D",
        "T3",
        4.0,
    ]

    with pytest.raises(
        ValueError,
        match="absent from the accessibility data",
    ):
        attach_population_to_accessibility(
            accessibility_frame(),
            allocation,
        )


def test_duplicate_node_tract_pair_is_rejected(
    tmp_path,
):
    path = tmp_path / "allocation.csv"

    allocation = allocation_frame()
    allocation.loc[len(allocation)] = [
        "A",
        "T1",
        1.0,
    ]

    allocation.to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="duplicate node-and-tract pairs",
    ):
        load_population_allocation(path)


@pytest.mark.parametrize(
    "value",
    [-1, "invalid"],
)
def test_invalid_population_is_rejected(
    tmp_path,
    value,
):
    path = tmp_path / "allocation.csv"

    allocation = allocation_frame()

    if isinstance(value, str):
        allocation["assigned_population"] = (
            allocation["assigned_population"].astype(object)
        )

    allocation.loc[0, "assigned_population"] = value
    allocation.to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_population_allocation(path)


def test_writes_population_output_and_diagnostics(
    tmp_path,
):
    accessibility_path = (
        tmp_path / "accessibility.csv"
    )

    allocation_path = (
        tmp_path / "allocation.csv"
    )

    output_path = (
        tmp_path / "nodes_with_population.csv"
    )

    diagnostics_path = (
        tmp_path / "diagnostics.json"
    )

    accessibility_frame().to_csv(
        accessibility_path,
        index=False,
    )

    allocation_frame().to_csv(
        allocation_path,
        index=False,
    )

    summary = build_population_attached_nodes(
        accessibility_path,
        allocation_path,
        output_path,
        diagnostics_path=diagnostics_path,
    )

    saved = pd.read_csv(
        output_path,
        dtype={"origin_node": str},
    )

    diagnostics = json.loads(
        diagnostics_path.read_text()
    )

    assert summary.accessibility_node_count == 3
    assert summary.assigned_population_total == (
        pytest.approx(35.0)
    )

    assert len(saved) == 3
    assert len(saved.columns) == 5

    assert saved[
        "assigned_total_population"
    ].sum() == pytest.approx(35.0)

    assert diagnostics[
        "zero_allocation_node_count"
    ] == 1
