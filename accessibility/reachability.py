"""Bounded reachable-node calculations for accessibility analysis."""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass
import heapq
import itertools
import math

import networkx as nx


METERS_PER_MILE = 1609.344
DEFAULT_CUTOFF_MILES = 1.5

PROFILE_COST_FIELDS = {
    "distance": "length",
    "typical_adult": "cost_typical_adult_Baseline",
    "low_confidence_adult": (
        "cost_low_confidence_adult_Baseline"
    ),
    "child": "cost_child_Baseline",
}


@dataclass(frozen=True)
class ReachabilityResult:
    """Shortest reachable-node costs within one routing budget."""

    origin_node: Hashable
    weight_attribute: str
    budget: float
    best_cost_to_node: dict[Hashable, float]
    processed_node_count: int
    max_lts: float | None

    @property
    def reachable_nodes(self) -> frozenset[Hashable]:
        """Return all nodes reachable within the budget."""
        return frozenset(self.best_cost_to_node)

    @property
    def reachable_node_count(self) -> int:
        """Return the number of reachable nodes, including the origin."""
        return len(self.best_cost_to_node)


def cost_field_for_profile(profile: str) -> str:
    """Return the graph edge-cost field for a named profile."""
    normalized = str(profile).strip().lower()

    try:
        return PROFILE_COST_FIELDS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown routing profile {profile!r}. "
            f"Expected one of {sorted(PROFILE_COST_FIELDS)}."
        ) from exc


def miles_to_meters(miles: float) -> float:
    """Convert a finite, nonnegative mileage value to meters."""
    try:
        numeric_miles = float(miles)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Miles must be numeric, received {miles!r}."
        ) from exc

    if not math.isfinite(numeric_miles) or numeric_miles < 0:
        raise ValueError(
            "Miles must be finite and nonnegative, "
            f"received {miles!r}."
        )

    return numeric_miles * METERS_PER_MILE


def _numeric_edge_value(
    data: Mapping,
    attribute: str,
    edge_id: tuple,
) -> float:
    """Return one validated nonnegative numeric edge attribute."""
    if attribute not in data:
        raise KeyError(
            f"Edge {edge_id} is missing required attribute "
            f"{attribute!r}."
        )

    try:
        value = float(data[attribute])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Edge {edge_id} has non-numeric {attribute!r}: "
            f"{data[attribute]!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"Edge {edge_id} has non-finite {attribute!r}: "
            f"{value!r}."
        )

    if value < 0:
        raise ValueError(
            f"Edge {edge_id} has negative {attribute!r}: "
            f"{value!r}."
        )

    return value


def _out_edges(graph: nx.Graph, node: Hashable):
    """Yield outgoing edges in a uniform four-value form."""
    if graph.is_multigraph():
        yield from graph.out_edges(
            node,
            keys=True,
            data=True,
        )
        return

    for source, target, data in graph.out_edges(
        node,
        data=True,
    ):
        yield source, target, None, data


def bounded_reachable_nodes(
    graph: nx.Graph,
    origin_node: Hashable,
    budget: float,
    weight_attribute: str,
    *,
    max_lts: float | None = None,
    lts_attribute: str = "max_lts",
) -> ReachabilityResult:
    """Find nodes reachable through fully traversable directed edges.

    The budget and edge weights use the same units. For the StressMap
    graph, both ``length`` and the precomputed rider-profile costs are
    measured in meter-equivalent units.

    When ``max_lts`` is supplied, edges whose maximum LTS exceeds that
    threshold are excluded before traversal.
    """
    if not graph.is_directed():
        raise nx.NetworkXError(
            "Accessibility routing requires a directed graph."
        )

    if origin_node not in graph:
        raise nx.NodeNotFound(
            f"Origin node {origin_node!r} is not in the graph."
        )

    try:
        numeric_budget = float(budget)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Budget must be numeric, received {budget!r}."
        ) from exc

    if not math.isfinite(numeric_budget) or numeric_budget < 0:
        raise ValueError(
            "Budget must be finite and nonnegative, "
            f"received {budget!r}."
        )

    numeric_max_lts = None

    if max_lts is not None:
        try:
            numeric_max_lts = float(max_lts)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"max_lts must be numeric, received {max_lts!r}."
            ) from exc

        if (
            not math.isfinite(numeric_max_lts)
            or numeric_max_lts < 0
        ):
            raise ValueError(
                "max_lts must be finite and nonnegative, "
                f"received {max_lts!r}."
            )

    best_cost_to_node: dict[Hashable, float] = {
        origin_node: 0.0
    }

    sequence = itertools.count()

    queue: list[tuple[float, int, Hashable]] = [
        (0.0, next(sequence), origin_node)
    ]

    processed_node_count = 0

    while queue:
        accumulated_cost, _, current_node = heapq.heappop(
            queue
        )

        if accumulated_cost > best_cost_to_node[current_node]:
            continue

        processed_node_count += 1

        for (
            _,
            next_node,
            edge_key,
            data,
        ) in _out_edges(graph, current_node):
            edge_id = (
                current_node,
                next_node,
                edge_key,
            )

            if numeric_max_lts is not None:
                raw_lts = data.get(lts_attribute)

                # Unknown LTS values cannot be included in a
                # strict low-stress network.
                if raw_lts is None:
                    continue

                if (
                    isinstance(raw_lts, str)
                    and not raw_lts.strip()
                ):
                    continue

                edge_lts = _numeric_edge_value(
                    data,
                    lts_attribute,
                    edge_id,
                )

                if edge_lts > numeric_max_lts:
                    continue

            edge_cost = _numeric_edge_value(
                data,
                weight_attribute,
                edge_id,
            )

            new_cost = accumulated_cost + edge_cost

            # Full-edge boundary: an edge counts only when the
            # complete edge can be traversed within the budget.
            if new_cost > numeric_budget:
                continue

            previous_best = best_cost_to_node.get(next_node)

            if (
                previous_best is None
                or new_cost < previous_best
            ):
                best_cost_to_node[next_node] = new_cost

                heapq.heappush(
                    queue,
                    (
                        new_cost,
                        next(sequence),
                        next_node,
                    ),
                )

    return ReachabilityResult(
        origin_node=origin_node,
        weight_attribute=weight_attribute,
        budget=numeric_budget,
        best_cost_to_node=best_cost_to_node,
        processed_node_count=processed_node_count,
        max_lts=numeric_max_lts,
    )
