import math

import networkx as nx
import pandas as pd
import pytest

from accessibility.opportunity_access import (
    OpportunityIndex,
)
from accessibility.profile_access import (
    ProfileSpec,
    calculate_multi_profile_accessibility,
    relative_access,
)


def opportunity_frame(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "node_id",
            "jobs",
            "schools",
            "transit_stations",
            "bus_stops",
            "greenspace",
            "healthcare",
            "stores",
        ],
    )


def add_edge(
    graph,
    source,
    target,
    *,
    length,
    typical,
    low_confidence,
    child,
    max_lts,
):
    graph.add_edge(
        source,
        target,
        length=str(length),
        cost_typical_adult_Baseline=str(typical),
        cost_low_confidence_adult_Baseline=str(
            low_confidence
        ),
        cost_child_Baseline=str(child),
        max_lts=str(max_lts),
    )


def test_combines_profile_totals_and_ratios():
    graph = nx.MultiDiGraph()

    add_edge(
        graph,
        "A",
        "B",
        length=100,
        typical=150,
        low_confidence=200,
        child=250,
        max_lts=2,
    )

    add_edge(
        graph,
        "B",
        "C",
        length=100,
        typical=300,
        low_confidence=400,
        child=500,
        max_lts=3,
    )

    opportunities = OpportunityIndex.from_frame(
        opportunity_frame(
            [
                ["A", 1, 0, 0, 0, 0, 0, 0],
                ["B", 10, 1, 0, 1, 0, 0, 0],
                ["C", 100, 0, 1, 0, 1, 1, 1],
            ]
        )
    )

    result = calculate_multi_profile_accessibility(
        graph=graph,
        opportunities=opportunities,
        origin_node="A",
        budget=250,
    )

    record = result.to_record()

    assert record["distance_jobs"] == 111
    assert record["distance_reachable_node_count"] == 3

    for profile in (
        "typical_adult",
        "low_confidence_adult",
        "child",
        "strict_lts_1_2",
    ):
        assert record[f"{profile}_jobs"] == 11

        assert record[
            f"{profile}_jobs_relative"
        ] == pytest.approx(11 / 111)

        assert record[
            f"{profile}_transit_stations_relative"
        ] == 0

    assert record[
        "typical_adult_schools_relative"
    ] == 1


def test_zero_baseline_produces_nan_ratios():
    graph = nx.MultiDiGraph()
    graph.add_node("A")

    opportunities = OpportunityIndex.from_frame(
        opportunity_frame(
            [
                ["A", 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )

    result = calculate_multi_profile_accessibility(
        graph,
        opportunities,
        origin_node="A",
        budget=100,
    )

    record = result.to_record()

    assert record["distance_jobs"] == 0

    assert math.isnan(
        record["typical_adult_jobs_relative"]
    )

    assert math.isnan(
        record["strict_lts_1_2_schools_relative"]
    )


def test_relative_access_and_profile_validation():
    assert relative_access(0, 10) == 0
    assert relative_access(10, 10) == 1
    assert math.isnan(relative_access(0, 0))

    with pytest.raises(ValueError):
        relative_access(-1, 10)

    duplicate_profiles = (
        ProfileSpec("distance", "length"),
        ProfileSpec("distance", "length"),
    )

    graph = nx.MultiDiGraph()
    graph.add_node("A")

    opportunities = OpportunityIndex.from_frame(
        opportunity_frame(
            [
                ["A", 0, 0, 0, 0, 0, 0, 0],
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        calculate_multi_profile_accessibility(
            graph,
            opportunities,
            "A",
            10,
            profiles=duplicate_profiles,
        )
