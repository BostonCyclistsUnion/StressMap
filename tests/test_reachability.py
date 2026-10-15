import networkx as nx
import pytest

from accessibility.reachability import (
    bounded_reachable_nodes,
    cost_field_for_profile,
    miles_to_meters,
)


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


def test_distance_budget_respects_direction():
    graph = nx.MultiDiGraph()

    add_edge(graph, "A", "B", length=100)
    add_edge(graph, "B", "C", length=100)

    result = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=150,
        weight_attribute="length",
    )

    assert result.reachable_nodes == {"A", "B"}
    assert result.best_cost_to_node == {
        "A": 0.0,
        "B": 100.0,
    }

    reverse = bounded_reachable_nodes(
        graph,
        origin_node="C",
        budget=500,
        weight_attribute="length",
    )

    assert reverse.reachable_nodes == {"C"}


def test_profile_cost_changes_reachability():
    graph = nx.MultiDiGraph()

    add_edge(
        graph,
        "A",
        "B",
        length=100,
        typical_cost=150,
    )

    add_edge(
        graph,
        "B",
        "C",
        length=100,
        typical_cost=300,
    )

    distance = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=250,
        weight_attribute="length",
    )

    weighted = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=250,
        weight_attribute=(
            "cost_typical_adult_Baseline"
        ),
    )

    assert distance.reachable_nodes == {"A", "B", "C"}
    assert weighted.reachable_nodes == {"A", "B"}


def test_strict_lts_filter_excludes_high_stress_edge():
    graph = nx.MultiDiGraph()

    add_edge(
        graph,
        "A",
        "B",
        length=100,
        max_lts=2,
    )

    add_edge(
        graph,
        "B",
        "C",
        length=100,
        max_lts=3,
    )

    unrestricted = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=500,
        weight_attribute="length",
    )

    low_stress = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=500,
        weight_attribute="length",
        max_lts=2,
    )

    assert unrestricted.reachable_nodes == {
        "A",
        "B",
        "C",
    }

    assert low_stress.reachable_nodes == {
        "A",
        "B",
    }


def test_strict_lts_excludes_unknown_values():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        "A",
        "B",
        length="100",
        cost_typical_adult_Baseline="100",
        max_lts="",
    )

    graph.add_edge(
        "A",
        "C",
        length="100",
        cost_typical_adult_Baseline="100",
    )

    result = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=500,
        weight_attribute="length",
        max_lts=2,
    )

    assert result.reachable_nodes == {"A"}


def test_strict_lts_rejects_malformed_value():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        "A",
        "B",
        length="100",
        cost_typical_adult_Baseline="100",
        max_lts="unknown",
    )

    with pytest.raises(
        ValueError,
        match="non-numeric 'max_lts'",
    ):
        bounded_reachable_nodes(
            graph,
            origin_node="A",
            budget=500,
            weight_attribute="length",
            max_lts=2,
        )


def test_parallel_edges_use_the_cheapest_route():
    graph = nx.MultiDiGraph()

    add_edge(
        graph,
        "A",
        "B",
        length=300,
    )

    add_edge(
        graph,
        "A",
        "B",
        length=100,
    )

    result = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=150,
        weight_attribute="length",
    )

    assert result.best_cost_to_node["B"] == 100.0


def test_zero_budget_and_validation():
    graph = nx.MultiDiGraph()
    graph.add_node("A")

    result = bounded_reachable_nodes(
        graph,
        origin_node="A",
        budget=0,
        weight_attribute="length",
    )

    assert result.reachable_nodes == {"A"}

    assert cost_field_for_profile("distance") == "length"
    assert miles_to_meters(1) == pytest.approx(1609.344)

    with pytest.raises(ValueError):
        bounded_reachable_nodes(
            graph,
            origin_node="A",
            budget=-1,
            weight_attribute="length",
        )

    with pytest.raises(nx.NodeNotFound):
        bounded_reachable_nodes(
            graph,
            origin_node="missing",
            budget=10,
            weight_attribute="length",
        )

    undirected = nx.Graph()
    undirected.add_node("A")

    with pytest.raises(nx.NetworkXError):
        bounded_reachable_nodes(
            undirected,
            origin_node="A",
            budget=10,
            weight_attribute="length",
        )
