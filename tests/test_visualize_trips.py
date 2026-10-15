"""Tests for diagnostic OD trip visualization."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from accessibility.visualize_trips import (
    DIAGNOSTIC_DESCRIPTION,
    render_html,
    sample_trip_rows,
    trips_to_geojson,
    validate_trip_table,
    write_visualization_outputs,
)


def make_trips() -> pd.DataFrame:
    """Create a small aggregated OD trip table."""
    return pd.DataFrame(
        {
            "origin_node": [
                "A",
                "A",
                "B",
                "C",
            ],
            "destination_node": [
                "B",
                "C",
                "D",
                "D",
            ],
            "trip_type": [
                "elementary_school",
                "work",
                "shopping",
                "recreation",
            ],
            "selection_method": [
                "nearest",
                "lognormal_weighted",
                "lognormal_weighted",
                "lognormal_weighted",
            ],
            "origin_longitude": [
                -71.10,
                -71.10,
                -71.09,
                -71.07,
            ],
            "origin_latitude": [
                42.35,
                42.35,
                42.35,
                42.35,
            ],
            "destination_longitude": [
                -71.09,
                -71.07,
                -71.04,
                -71.04,
            ],
            "destination_latitude": [
                42.35,
                42.35,
                42.35,
                42.35,
            ],
            "straight_line_distance_m": [
                1000.0,
                3000.0,
                5000.0,
                3000.0,
            ],
            "count": [
                4,
                20,
                2,
                7,
            ],
        }
    )


def test_validate_trip_table_rejects_missing_columns() -> None:
    """Required visualization fields must be present."""
    trips = make_trips().drop(
        columns=["origin_longitude"]
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_trip_table(trips)


def test_validate_trip_table_rejects_fractional_counts() -> None:
    """Aggregated trip counts must be whole numbers."""
    trips = make_trips()
    trips["count"] = trips["count"].astype(float)
    trips.loc[0, "count"] = 1.5

    with pytest.raises(
        ValueError,
        match="positive whole numbers",
    ):
        validate_trip_table(trips)


def test_sample_trip_rows_is_reproducible() -> None:
    """The same seed returns the same weighted subset."""
    trips = make_trips()

    first = sample_trip_rows(
        trips,
        maximum_trips=2,
        seed=26,
    )

    second = sample_trip_rows(
        trips,
        maximum_trips=2,
        seed=26,
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

    assert len(first) == 2


def test_geojson_contains_straight_line_metadata() -> None:
    """GeoJSON explicitly identifies diagnostic geometry."""
    trips = make_trips().iloc[[0]]

    geojson = trips_to_geojson(trips)

    assert geojson["type"] == "FeatureCollection"
    assert geojson["feature_count"] == 1
    assert geojson["represented_trip_count"] == 4
    assert geojson["description"] == (
        DIAGNOSTIC_DESCRIPTION
    )

    feature = geojson["features"][0]

    assert feature["geometry"] == {
        "type": "LineString",
        "coordinates": [
            [-71.10, 42.35],
            [-71.09, 42.35],
        ],
    }

    assert (
        feature["properties"]["is_routed_path"]
        is False
    )

    assert (
        feature["properties"]["geometry_role"]
        == "diagnostic_straight_line_od"
    )


def test_render_html_labels_lines_as_not_routed() -> None:
    """The HTML map clearly states its limitation."""
    geojson = trips_to_geojson(
        make_trips().iloc[[0]]
    )

    html = render_html(geojson)

    assert "Diagnostic OD visualization" in html
    assert "not bicycle routes" in html
    assert "not a routed path" in html
    assert "__GEOJSON_PAYLOAD__" not in html


def test_write_visualization_outputs(tmp_path) -> None:
    """HTML and GeoJSON outputs are both written."""
    output_html = tmp_path / "trips.html"
    output_geojson = tmp_path / "trips.geojson"

    geojson = write_visualization_outputs(
        make_trips(),
        output_html,
        output_geojson,
    )

    assert output_html.exists()
    assert output_geojson.exists()

    stored_geojson = json.loads(
        output_geojson.read_text(
            encoding="utf-8"
        )
    )

    assert stored_geojson == geojson
    assert geojson["feature_count"] == 4
    assert geojson["represented_trip_count"] == 33

    html = output_html.read_text(
        encoding="utf-8"
    )

    assert "leaflet" in html.lower()
    assert "not bicycle routes" in html
