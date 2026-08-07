"""End-to-end validation for generated OD spatial outputs."""

from __future__ import annotations

import json

import pandas as pd

from accessibility.trip_generation import (
    generate_trip_pairs,
    write_outputs,
)
from accessibility.visualize_trips import (
    load_trip_table,
    sample_trip_rows,
    write_visualization_outputs,
)


def test_trip_generation_to_spatial_outputs(
    tmp_path,
) -> None:
    """Generated demand can be written, loaded, and mapped."""
    nodes = pd.DataFrame(
        {
            "node_id": ["A", "B", "C", "D"],
            "longitude": [
                -71.100,
                -71.090,
                -71.080,
                -71.070,
            ],
            "latitude": [
                42.350,
                42.350,
                42.350,
                42.350,
            ],
            "x_m": [
                0.0,
                1000.0,
                2000.0,
                3000.0,
            ],
            "y_m": [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    origins = pd.DataFrame(
        {
            "node_id": ["A", "B", "C"],
            "origin_population": [
                100.0,
                50.0,
                25.0,
            ],
        }
    )

    opportunities = pd.DataFrame(
        {
            "node_id": ["A", "B", "C", "D"],
            "jobs": [0, 0, 3, 2],
            "schools": [0, 1, 0, 1],
            "transit_stations": [1, 0, 0, 0],
            "bus_stops": [0, 1, 1, 0],
            "greenspace": [0, 0, 1, 1],
            "healthcare": [0, 1, 0, 1],
            "stores": [0, 0, 1, 2],
        }
    )

    trip_counts = {
        "elementary_school": 3,
        "healthcare": 3,
        "work": 3,
        "transit": 3,
        "shopping": 3,
        "recreation": 3,
    }

    trips, diagnostics = generate_trip_pairs(
        nodes=nodes,
        origins=origins,
        opportunities=opportunities,
        trip_counts=trip_counts,
        seed=26,
    )

    trips_path = tmp_path / "trips.csv"
    diagnostics_path = tmp_path / "diagnostics.json"

    write_outputs(
        trips=trips,
        diagnostics=diagnostics,
        output_path=trips_path,
        diagnostics_path=diagnostics_path,
    )

    loaded = load_trip_table(trips_path)

    sampled = sample_trip_rows(
        loaded,
        maximum_trips=100,
        seed=26,
    )

    html_path = tmp_path / "trips.html"
    geojson_path = tmp_path / "trips.geojson"

    geojson = write_visualization_outputs(
        trips=sampled,
        output_html=html_path,
        output_geojson=geojson_path,
    )

    diagnostics_payload = json.loads(
        diagnostics_path.read_text(
            encoding="utf-8"
        )
    )

    geojson_payload = json.loads(
        geojson_path.read_text(
            encoding="utf-8"
        )
    )

    html = html_path.read_text(
        encoding="utf-8"
    )

    assert trips_path.is_file()
    assert diagnostics_path.is_file()
    assert html_path.is_file()
    assert geojson_path.is_file()

    assert int(loaded["count"].sum()) == 18
    assert set(loaded["trip_type"]) == set(trip_counts)

    assert (
        diagnostics_payload["generated_trip_counts"]
        == trip_counts
    )

    assert (
        diagnostics_payload["generated_total_trips"]
        == 18
    )

    assert geojson_payload == geojson
    assert geojson["feature_count"] == len(loaded)
    assert geojson["represented_trip_count"] == 18

    assert all(
        feature["properties"]["is_routed_path"] is False
        for feature in geojson["features"]
    )

    assert all(
        feature["properties"]["geometry_role"]
        == "diagnostic_straight_line_od"
        for feature in geojson["features"]
    )

    assert "__GEOJSON_PAYLOAD__" not in html
    assert "not bicycle routes" in html
