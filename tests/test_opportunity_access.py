import networkx as nx
import pandas as pd
import pytest

from accessibility.opportunity_access import (
    OpportunityIndex,
    calculate_opportunity_accessibility,
)


def opportunity_frame(rows):
    columns = [
        "node_id",
        "jobs",
        "schools",
        "transit_stations",
        "bus_stops",
        "greenspace",
        "healthcare",
        "stores",
    ]

    return pd.DataFrame(rows, columns=columns)


def add_edge(
    graph,
    source,
    target,
    *,
    length,
    typical_cost=None,
    max_lts=1,
):
    graph.add_edge(
        source,
        target,
        length=str(length),
        cost_typical_adult_Baseline=str(
            length
            if typical_cost is None
            else typical_cost
        ),
        max_lts=str(max_lts),
    )


def test_sums_opportunities_at_reachable_nodes():
    graph = nx.MultiDiGraph()

    add_edge(graph, "A", "B", length=100)
    add_edge(graph, "B", "C", length=100)

    opportunities = OpportunityIndex.from_frame(
        opportunity_frame(
            [
                ["A", 1, 0, 0, 0, 0, 0, 0],
                ["B", 2, 1, 0, 1, 0, 0, 0],
                ["C", 4, 2, 1, 0, 1, 1, 1],
            ]
        )
    )

    result = calculate_opportunity_accessibility(
        graph=graph,
        opportunities=opportunities,
        origin_node="A",
        budget=150,
        weight_attribute="length",
    )

    assert result.reachable_node_count == 2
    assert result.opportunity_totals["jobs"] == 3
    assert result.opportunity_totals["schools"] == 1
    assert result.opportunity_totals["bus_stops"] == 1
    assert result.opportunity_totals["stores"] == 0


def test_profiles_change_reachable_opportunities():
    graph = nx.MultiDiGraph()

    add_edge(
        graph,
        "A",
        "B",
        length=100,
        typical_cost=150,
        max_lts=2,
    )

    add_edge(
        graph,
        "B",
        "C",
        length=100,
        typical_cost=300,
        max_lts=3,
    )

    opportunities = OpportunityIndex.from_frame(
        opportunity_frame(
            [
                ["A", 1, 0, 0, 0, 0, 0, 0],
                ["B", 10, 1, 0, 0, 0, 0, 0],
                ["C", 100, 0, 1, 0, 0, 0, 0],
            ]
        )
    )

    distance = calculate_opportunity_accessibility(
        graph,
        opportunities,
        "A",
        250,
        "length",
    )

    typical = calculate_opportunity_accessibility(
        graph,
        opportunities,
        "A",
        250,
        "cost_typical_adult_Baseline",
    )

    low_stress = calculate_opportunity_accessibility(
        graph,
        opportunities,
        "A",
        500,
        "length",
        max_lts=2,
    )

    assert distance.opportunity_totals["jobs"] == 111
    assert typical.opportunity_totals["jobs"] == 11
    assert low_stress.opportunity_totals["jobs"] == 11

    assert distance.opportunity_totals[
        "transit_stations"
    ] == 1

    assert typical.opportunity_totals[
        "transit_stations"
    ] == 0

    assert low_stress.opportunity_totals[
        "transit_stations"
    ] == 0


def test_missing_reachable_node_raises_error():
    graph = nx.MultiDiGraph()
    add_edge(graph, "A", "B", length=10)

    opportunities = OpportunityIndex.from_frame(
        opportunity_frame(
            [
                ["A", 1, 0, 0, 0, 0, 0, 0],
            ]
        )
    )

    with pytest.raises(
        KeyError,
        match="absent from the opportunity table",
    ):
        calculate_opportunity_accessibility(
            graph,
            opportunities,
            "A",
            20,
            "length",
        )


def test_rejects_duplicate_and_negative_values():
    duplicate = opportunity_frame(
        [
            ["A", 1, 0, 0, 0, 0, 0, 0],
            ["A", 2, 0, 0, 0, 0, 0, 0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        OpportunityIndex.from_frame(duplicate)

    negative = opportunity_frame(
        [
            ["A", -1, 0, 0, 0, 0, 0, 0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="contains negative values",
    ):
        OpportunityIndex.from_frame(negative)
