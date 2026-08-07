import json

import numpy as np
import pandas as pd
import pytest

from accessibility.opportunity_access import OPPORTUNITY_COLUMNS
from accessibility.population_summaries import (
    PROFILE_NAMES,
    build_population_weighted_summaries,
    load_population_nodes,
    summarize_region,
    summarize_tracts,
)


PROFILE_MULTIPLIERS = {
    "distance": 1.0,
    "typical_adult": 0.5,
    "low_confidence_adult": 0.4,
    "child": 0.2,
    "strict_lts_1_2": 0.8,
}


def node_frame() -> pd.DataFrame:
    data = {
        "origin_node": ["A", "B", "C"],
        "assigned_total_population": [
            15.0,
            20.0,
            0.0,
        ],
    }

    distance_values = np.array(
        [10.0, 0.0, 20.0]
    )

    for profile in PROFILE_NAMES:
        multiplier = PROFILE_MULTIPLIERS[
            profile
        ]

        for opportunity in OPPORTUNITY_COLUMNS:
            data[
                f"{profile}_{opportunity}"
            ] = (
                distance_values
                * multiplier
            )

    return pd.DataFrame(data)


def allocation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": [
                "A",
                "A",
                "B",
                "C",
            ],
            "GEOID": [
                "T1",
                "T2",
                "T1",
                "T3",
            ],
            "assigned_population": [
                10.0,
                5.0,
                20.0,
                0.0,
            ],
        }
    )


def summary_row(
    frame: pd.DataFrame,
    *,
    profile: str,
    opportunity: str,
    geoid: str | None = None,
) -> pd.Series:
    selected = frame[
        (frame["profile"] == profile)
        & (
            frame["opportunity"]
            == opportunity
        )
    ]

    if geoid is not None:
        selected = selected[
            selected["GEOID"] == geoid
        ]

    assert len(selected) == 1

    return selected.iloc[0]


def test_regional_population_weighted_summary():
    regional = summarize_region(
        node_frame()
    )

    assert len(regional) == 35

    row = summary_row(
        regional,
        profile="typical_adult",
        opportunity="jobs",
    )

    assert row[
        "accessibility_node_count"
    ] == 3

    assert row[
        "positive_population_node_count"
    ] == 2

    assert row[
        "total_population"
    ] == pytest.approx(35.0)

    assert row[
        "population_weighted_access_sum"
    ] == pytest.approx(75.0)

    assert row[
        "population_weighted_mean_access"
    ] == pytest.approx(75.0 / 35.0)

    assert row[
        "population_weighted_distance_access_sum"
    ] == pytest.approx(150.0)

    assert row[
        "population_weighted_mean_distance_access"
    ] == pytest.approx(150.0 / 35.0)

    assert row[
        "ratio_of_weighted_means"
    ] == pytest.approx(0.5)

    assert row[
        "valid_relative_population"
    ] == pytest.approx(15.0)

    assert row[
        "zero_distance_access_population"
    ] == pytest.approx(20.0)

    assert row[
        "valid_relative_population_share"
    ] == pytest.approx(15.0 / 35.0)

    assert row[
        "population_weighted_mean_origin_relative"
    ] == pytest.approx(0.5)


def test_tract_summary_preserves_allocation_weights():
    tract = summarize_tracts(
        node_frame(),
        allocation_frame(),
    )

    assert len(tract) == 105
    assert tract["GEOID"].nunique() == 3

    t1 = summary_row(
        tract,
        geoid="T1",
        profile="typical_adult",
        opportunity="jobs",
    )

    assert t1[
        "allocation_row_count"
    ] == 2

    assert t1[
        "allocated_node_count"
    ] == 2

    assert t1[
        "tract_population"
    ] == pytest.approx(30.0)

    assert t1[
        "population_weighted_access_sum"
    ] == pytest.approx(50.0)

    assert t1[
        "population_weighted_distance_access_sum"
    ] == pytest.approx(100.0)

    assert t1[
        "ratio_of_weighted_means"
    ] == pytest.approx(0.5)

    assert t1[
        "valid_relative_population"
    ] == pytest.approx(10.0)

    assert t1[
        "zero_distance_access_population"
    ] == pytest.approx(20.0)

    t3 = summary_row(
        tract,
        geoid="T3",
        profile="typical_adult",
        opportunity="jobs",
    )

    assert t3[
        "tract_population"
    ] == pytest.approx(0.0)

    assert np.isnan(
        t3[
            "population_weighted_mean_access"
        ]
    )

    assert np.isnan(
        t3[
            "ratio_of_weighted_means"
        ]
    )

    assert np.isnan(
        t3[
            "valid_relative_population_share"
        ]
    )


def test_builds_summary_files_and_diagnostics(
    tmp_path,
):
    nodes_path = tmp_path / "nodes.csv"
    allocation_path = (
        tmp_path / "allocation.csv"
    )

    regional_path = (
        tmp_path / "regional.csv"
    )

    tract_path = tmp_path / "tract.csv"

    diagnostics_path = (
        tmp_path / "diagnostics.json"
    )

    node_frame().to_csv(
        nodes_path,
        index=False,
    )

    allocation_frame().to_csv(
        allocation_path,
        index=False,
    )

    result = (
        build_population_weighted_summaries(
            nodes_path,
            allocation_path,
            regional_path,
            tract_path,
            diagnostics_path=(
                diagnostics_path
            ),
        )
    )

    regional = pd.read_csv(
        regional_path
    )

    tract = pd.read_csv(
        tract_path,
        dtype={"GEOID": str},
    )

    diagnostics = json.loads(
        diagnostics_path.read_text()
    )

    assert result.node_count == 3
    assert result.tract_count == 3
    assert result.total_population == (
        pytest.approx(35.0)
    )

    assert result.regional_row_count == 35
    assert result.tract_row_count == 105

    assert (
        result.zero_population_tract_count
        == 1
    )

    assert len(regional) == 35
    assert len(tract) == 105

    assert diagnostics[
        "regional_row_count"
    ] == 35

    assert diagnostics[
        "tract_row_count"
    ] == 105

    assert diagnostics[
        "zero_population_tract_count"
    ] == 1


def test_population_mismatch_is_rejected(
    tmp_path,
):
    nodes = node_frame()

    nodes.loc[
        nodes["origin_node"] == "A",
        "assigned_total_population",
    ] = 16.0

    nodes_path = tmp_path / "nodes.csv"
    allocation_path = (
        tmp_path / "allocation.csv"
    )

    nodes.to_csv(
        nodes_path,
        index=False,
    )

    allocation_frame().to_csv(
        allocation_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "does not match the raw "
            "tract-to-node allocation"
        ),
    ):
        build_population_weighted_summaries(
            nodes_path,
            allocation_path,
            tmp_path / "regional.csv",
            tmp_path / "tract.csv",
        )


def test_unknown_allocation_node_is_rejected(
    tmp_path,
):
    allocation = allocation_frame()

    allocation.loc[
        len(allocation)
    ] = [
        "D",
        "T4",
        4.0,
    ]

    nodes_path = tmp_path / "nodes.csv"
    allocation_path = (
        tmp_path / "allocation.csv"
    )

    node_frame().to_csv(
        nodes_path,
        index=False,
    )

    allocation.to_csv(
        allocation_path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="unknown nodes",
    ):
        build_population_weighted_summaries(
            nodes_path,
            allocation_path,
            tmp_path / "regional.csv",
            tmp_path / "tract.csv",
        )


def test_profile_access_above_distance_is_rejected(
    tmp_path,
):
    nodes = node_frame()

    nodes.loc[
        0,
        "child_jobs",
    ] = (
        nodes.loc[
            0,
            "distance_jobs",
        ]
        + 1.0
    )

    path = tmp_path / "nodes.csv"

    nodes.to_csv(
        path,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "child_jobs exceeds "
            "distance_jobs"
        ),
    ):
        load_population_nodes(path)
