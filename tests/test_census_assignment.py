import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from census.assignment import assign_population_to_nodes_by_tract_area


def test_population_is_conserved_for_simple_tract():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(10, 0)],
        crs="EPSG:26986",
    )
    tracts = gpd.GeoDataFrame(
        {"GEOID": ["test"], "population": [100]},
        geometry=[box(-5, -5, 15, 5)],
        crs="EPSG:26986",
    )

    nodes_out, allocation, unassigned = assign_population_to_nodes_by_tract_area(
        nodes,
        tracts,
        projected_crs="EPSG:26986",
        tract_filter_method="none",
    )

    assert allocation["assigned_population"].sum() == pytest.approx(100)
    assert nodes_out["assigned_population"].sum() == pytest.approx(100)
    assert allocation["area_share"].sum() == pytest.approx(1)
    assert (allocation["assigned_population"] >= 0).all()
    assert unassigned.empty


def test_single_candidate_node_receives_entire_population():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        crs="EPSG:26986",
    )
    tracts = gpd.GeoDataFrame(
        {"GEOID": ["single"], "population": [75]},
        geometry=[box(-5, -5, 5, 5)],
        crs="EPSG:26986",
    )

    nodes_out, allocation, unassigned = assign_population_to_nodes_by_tract_area(
        nodes,
        tracts,
        projected_crs="EPSG:26986",
        tract_filter_method="none",
    )

    assert len(allocation) == 1
    assert allocation.iloc[0]["area_share"] == pytest.approx(1)
    assert nodes_out.iloc[0]["assigned_population"] == pytest.approx(75)
    assert unassigned.empty


def test_invalid_parameters_raise_errors():
    nodes = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs="EPSG:26986")
    tracts = gpd.GeoDataFrame(
        {"GEOID": ["test"], "population": [1]},
        geometry=[box(-1, -1, 1, 1)],
        crs="EPSG:26986",
    )

    with pytest.raises(ValueError, match="candidate_buffer_m"):
        assign_population_to_nodes_by_tract_area(
            nodes,
            tracts,
            candidate_buffer_m=-1,
        )

    with pytest.raises(ValueError, match="min_region_overlap_share"):
        assign_population_to_nodes_by_tract_area(
            nodes,
            tracts,
            min_region_overlap_share=1.1,
        )


def test_tract_without_candidate_nodes_is_reported():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        index=[1001],
        crs="EPSG:26986",
    )
    tracts = gpd.GeoDataFrame(
        {
            "GEOID": ["far"],
            "population": [200],
        },
        geometry=[box(1000, 1000, 1020, 1020)],
        crs="EPSG:26986",
    )

    nodes_out, allocation, unassigned = (
        assign_population_to_nodes_by_tract_area(
            nodes,
            tracts,
            projected_crs="EPSG:26986",
            candidate_buffer_m=0,
            tract_filter_method="none",
        )
    )

    assert allocation.empty
    assert allocation.columns.tolist() == [
        "node_id",
        "GEOID",
        "area_share",
        "raw_area_share",
        "tract_coverage_ratio",
        "assigned_population",
    ]
    assert nodes_out["assigned_population"].sum() == 0
    assert len(unassigned) == 1
    assert unassigned.iloc[0]["GEOID"] == "far"
    assert unassigned.iloc[0]["population"] == pytest.approx(200)
    assert unassigned.iloc[0]["reason"] == "no_candidate_nodes"


def test_invalid_population_values_raise_errors():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        crs="EPSG:26986",
    )

    negative = gpd.GeoDataFrame(
        {
            "GEOID": ["negative"],
            "population": [-1],
        },
        geometry=[box(-1, -1, 1, 1)],
        crs="EPSG:26986",
    )

    with pytest.raises(ValueError, match="negative"):
        assign_population_to_nodes_by_tract_area(
            nodes,
            negative,
            projected_crs="EPSG:26986",
            tract_filter_method="none",
        )

    missing = gpd.GeoDataFrame(
        {
            "GEOID": ["missing"],
            "population": [None],
        },
        geometry=[box(-1, -1, 1, 1)],
        crs="EPSG:26986",
    )

    with pytest.raises(ValueError, match="numeric"):
        assign_population_to_nodes_by_tract_area(
            nodes,
            missing,
            projected_crs="EPSG:26986",
            tract_filter_method="none",
        )


def test_duplicate_tract_ids_raise_error():
    nodes = gpd.GeoDataFrame(
        geometry=[Point(0, 0)],
        crs="EPSG:26986",
    )
    tracts = gpd.GeoDataFrame(
        {
            "GEOID": ["duplicate", "duplicate"],
            "population": [10, 20],
        },
        geometry=[
            box(-2, -2, 0, 2),
            box(0, -2, 2, 2),
        ],
        crs="EPSG:26986",
    )

    with pytest.raises(ValueError, match="duplicate"):
        assign_population_to_nodes_by_tract_area(
            nodes,
            tracts,
            projected_crs="EPSG:26986",
            tract_filter_method="none",
        )
