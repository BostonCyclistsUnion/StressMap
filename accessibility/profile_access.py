"""Calculate and compare multiple opportunity-access profiles."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
import math

import networkx as nx

from accessibility.opportunity_access import (
    OPPORTUNITY_COLUMNS,
    OpportunityAccessibilityResult,
    OpportunityIndex,
    calculate_opportunity_accessibility,
)


@dataclass(frozen=True)
class ProfileSpec:
    """Routing configuration for one accessibility profile."""

    name: str
    weight_attribute: str
    max_lts: float | None = None


DEFAULT_PROFILE_SPECS = (
    ProfileSpec(
        name="distance",
        weight_attribute="length",
    ),
    ProfileSpec(
        name="typical_adult",
        weight_attribute="cost_typical_adult_Baseline",
    ),
    ProfileSpec(
        name="low_confidence_adult",
        weight_attribute="cost_low_confidence_adult_Baseline",
    ),
    ProfileSpec(
        name="child",
        weight_attribute="cost_child_Baseline",
    ),
    ProfileSpec(
        name="strict_lts_1_2",
        weight_attribute="length",
        max_lts=2,
    ),
)


def relative_access(
    numerator: float,
    denominator: float,
) -> float:
    """Return relative access or NaN for a zero baseline."""
    numeric_numerator = float(numerator)
    numeric_denominator = float(denominator)

    if (
        not math.isfinite(numeric_numerator)
        or numeric_numerator < 0
    ):
        raise ValueError(
            "The relative-access numerator must be finite "
            "and nonnegative."
        )

    if (
        not math.isfinite(numeric_denominator)
        or numeric_denominator < 0
    ):
        raise ValueError(
            "The relative-access denominator must be finite "
            "and nonnegative."
        )

    if numeric_denominator == 0:
        return math.nan

    ratio = numeric_numerator / numeric_denominator

    if math.isclose(
        ratio,
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return 0.0

    if math.isclose(
        ratio,
        1.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return 1.0

    return ratio


def _validate_profile_specs(
    profiles: Iterable[ProfileSpec],
) -> tuple[ProfileSpec, ...]:
    """Validate and normalize a collection of profile definitions."""
    normalized = tuple(profiles)

    if not normalized:
        raise ValueError(
            "At least one accessibility profile is required."
        )

    names = []

    for profile in normalized:
        name = str(profile.name).strip()

        if not name:
            raise ValueError("Profile names may not be blank.")

        if not name.replace("_", "").isalnum():
            raise ValueError(
                "Profile names may contain only letters, "
                "numbers, and underscores."
            )

        if not str(profile.weight_attribute).strip():
            raise ValueError(
                f"Profile {name!r} has a blank weight attribute."
            )

        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError("Profile names must be unique.")

    if "distance" not in names:
        raise ValueError(
            "Profiles must include a 'distance' baseline."
        )

    distance = normalized[names.index("distance")]

    if (
        distance.weight_attribute != "length"
        or distance.max_lts is not None
    ):
        raise ValueError(
            "The distance baseline must use unrestricted "
            "'length' routing."
        )

    return normalized


@dataclass(frozen=True)
class MultiProfileAccessibilityResult:
    """Opportunity accessibility results for one origin."""

    origin_node: Hashable
    budget: float
    profile_results: dict[
        str,
        OpportunityAccessibilityResult,
    ]

    def to_record(self) -> dict:
        """Return one wide record containing totals and ratios."""
        baseline = self.profile_results["distance"]

        record = {
            "origin_node": self.origin_node,
            "budget": self.budget,
        }

        for profile_name, result in self.profile_results.items():
            record[
                f"{profile_name}_reachable_node_count"
            ] = result.reachable_node_count

            record[
                f"{profile_name}_processed_node_count"
            ] = result.processed_node_count

            for column in OPPORTUNITY_COLUMNS:
                value = result.opportunity_totals[column]

                record[
                    f"{profile_name}_{column}"
                ] = value

                if profile_name == "distance":
                    continue

                baseline_value = (
                    baseline.opportunity_totals[column]
                )

                record[
                    f"{profile_name}_{column}_relative"
                ] = relative_access(
                    value,
                    baseline_value,
                )

        return record


def _validate_profile_result(
    profile_name: str,
    result: OpportunityAccessibilityResult,
    baseline: OpportunityAccessibilityResult,
) -> None:
    """Check that constrained access does not exceed baseline."""
    if result.reachable_node_count > baseline.reachable_node_count:
        raise RuntimeError(
            f"Profile {profile_name!r} reached more nodes "
            "than the unrestricted distance baseline."
        )

    for column in OPPORTUNITY_COLUMNS:
        profile_value = result.opportunity_totals[column]
        baseline_value = baseline.opportunity_totals[column]

        if (
            profile_value > baseline_value
            and not math.isclose(
                profile_value,
                baseline_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"Profile {profile_name!r} produced "
                f"{profile_value} reachable {column}, exceeding "
                f"the distance baseline of {baseline_value}."
            )


def calculate_multi_profile_accessibility(
    graph: nx.Graph,
    opportunities: OpportunityIndex,
    origin_node: Hashable,
    budget: float,
    *,
    profiles: Iterable[ProfileSpec] = (
        DEFAULT_PROFILE_SPECS
    ),
    lts_attribute: str = "max_lts",
) -> MultiProfileAccessibilityResult:
    """Calculate all configured access profiles for one origin."""
    validated_profiles = _validate_profile_specs(profiles)

    results = {}

    for profile in validated_profiles:
        results[profile.name] = (
            calculate_opportunity_accessibility(
                graph=graph,
                opportunities=opportunities,
                origin_node=origin_node,
                budget=budget,
                weight_attribute=profile.weight_attribute,
                max_lts=profile.max_lts,
                lts_attribute=lts_attribute,
            )
        )

    baseline = results["distance"]

    for profile_name, result in results.items():
        if profile_name == "distance":
            continue

        _validate_profile_result(
            profile_name,
            result,
            baseline,
        )

    return MultiProfileAccessibilityResult(
        origin_node=origin_node,
        budget=baseline.budget,
        profile_results=results,
    )
