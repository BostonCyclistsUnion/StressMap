"""Create diagnostic straight-line maps of generated OD demand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "origin_node",
    "destination_node",
    "trip_type",
    "selection_method",
    "origin_longitude",
    "origin_latitude",
    "destination_longitude",
    "destination_latitude",
    "straight_line_distance_m",
    "count",
}

DIAGNOSTIC_DESCRIPTION = (
    "Straight-line origin-destination connections for diagnostic "
    "visualization only. These lines are not routed bicycle paths "
    "and do not represent LTS routing results."
)


def validate_trip_table(trips: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize generated OD rows."""
    missing = REQUIRED_COLUMNS.difference(trips.columns)

    if missing:
        raise ValueError(
            "Trip table is missing required columns: "
            f"{sorted(missing)}"
        )

    validated = trips.copy()

    coordinate_columns = [
        "origin_longitude",
        "origin_latitude",
        "destination_longitude",
        "destination_latitude",
        "straight_line_distance_m",
    ]

    for column in coordinate_columns:
        validated[column] = pd.to_numeric(
            validated[column],
            errors="coerce",
        )

    validated["count"] = pd.to_numeric(
        validated["count"],
        errors="coerce",
    )

    numeric_values = validated[
        [
            *coordinate_columns,
            "count",
        ]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Trip coordinates, distances, and counts "
            "must all be finite numeric values."
        )

    invalid_coordinates = (
        validated["origin_longitude"].lt(-180)
        | validated["origin_longitude"].gt(180)
        | validated["destination_longitude"].lt(-180)
        | validated["destination_longitude"].gt(180)
        | validated["origin_latitude"].lt(-90)
        | validated["origin_latitude"].gt(90)
        | validated["destination_latitude"].lt(-90)
        | validated["destination_latitude"].gt(90)
    )

    if invalid_coordinates.any():
        raise ValueError(
            "Trip table contains coordinates outside "
            "valid longitude or latitude ranges."
        )

    if validated["straight_line_distance_m"].lt(0).any():
        raise ValueError(
            "Straight-line distances may not be negative."
        )

    invalid_counts = (
        validated["count"].le(0)
        | validated["count"].mod(1).ne(0)
    )

    if invalid_counts.any():
        raise ValueError(
            "Trip counts must be positive whole numbers."
        )

    validated["count"] = validated["count"].astype(int)

    identifier_columns = [
        "origin_node",
        "destination_node",
        "trip_type",
        "selection_method",
    ]

    for column in identifier_columns:
        validated[column] = (
            validated[column]
            .astype("string")
            .str.strip()
        )

        if (
            validated[column].isna().any()
            or validated[column].eq("").any()
        ):
            raise ValueError(
                f"Trip column {column!r} may not contain "
                "missing or blank values."
            )

    return validated


def load_trip_table(trips_path: Path) -> pd.DataFrame:
    """Load and validate an aggregated OD trip CSV."""
    trips = pd.read_csv(
        trips_path,
        dtype={
            "origin_node": "string",
            "destination_node": "string",
            "trip_type": "string",
            "selection_method": "string",
        },
    )

    if trips.empty:
        raise ValueError(
            "The trip table contains no OD rows."
        )

    return validate_trip_table(trips)


def sample_trip_rows(
    trips: pd.DataFrame,
    maximum_trips: int,
    seed: int,
) -> pd.DataFrame:
    """Select a reproducible count-weighted subset of OD rows."""
    if maximum_trips <= 0:
        raise ValueError(
            "maximum_trips must be greater than zero."
        )

    validated = validate_trip_table(trips)

    ordered = validated.sort_values(
        [
            "trip_type",
            "origin_node",
            "destination_node",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(ordered) <= maximum_trips:
        return ordered

    weights = ordered["count"].to_numpy(dtype=float)
    total_weight = float(weights.sum())

    if total_weight <= 0:
        raise ValueError(
            "Trip-row sampling weights do not have "
            "a positive sum."
        )

    rng = np.random.default_rng(seed)

    selected_positions = rng.choice(
        len(ordered),
        size=maximum_trips,
        replace=False,
        p=weights / total_weight,
    )

    selected_positions = np.sort(selected_positions)

    return ordered.iloc[
        selected_positions
    ].reset_index(drop=True)


def trips_to_geojson(
    trips: pd.DataFrame,
) -> dict[str, Any]:
    """Convert OD rows to straight-line GeoJSON features."""
    validated = validate_trip_table(trips)
    features: list[dict[str, Any]] = []

    for row in validated.itertuples(index=False):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [
                        float(row.origin_longitude),
                        float(row.origin_latitude),
                    ],
                    [
                        float(row.destination_longitude),
                        float(row.destination_latitude),
                    ],
                ],
            },
            "properties": {
                "origin_node": str(row.origin_node),
                "destination_node": str(
                    row.destination_node
                ),
                "trip_type": str(row.trip_type),
                "selection_method": str(
                    row.selection_method
                ),
                "count": int(row.count),
                "straight_line_distance_m": float(
                    row.straight_line_distance_m
                ),
                "geometry_role": (
                    "diagnostic_straight_line_od"
                ),
                "is_routed_path": False,
            },
        }

        features.append(feature)

    return {
        "type": "FeatureCollection",
        "name": "diagnostic_straight_line_od_trips",
        "description": DIAGNOSTIC_DESCRIPTION,
        "feature_count": len(features),
        "represented_trip_count": int(
            validated["count"].sum()
        ),
        "features": features,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta
    name="viewport"
    content="width=device-width, initial-scale=1"
  >
  <title>Diagnostic OD Trip Map</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIINfQ3ynh3Z8E0pA0TMMZbE1I7h7h6j3gE="
    crossorigin=""
  >
  <style>
    html,
    body {
      height: 100%;
      margin: 0;
      font-family:
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
    }

    #map {
      height: 100%;
      width: 100%;
    }

    .diagnostic-note {
      max-width: 360px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid #777;
      border-radius: 4px;
      color: #222;
      font-size: 13px;
      line-height: 1.4;
    }

    .diagnostic-note strong {
      display: block;
      margin-bottom: 4px;
    }

    .legend {
      padding: 8px 10px;
      background: rgba(255, 255, 255, 0.95);
      border: 1px solid #777;
      border-radius: 4px;
      color: #222;
      font-size: 12px;
      line-height: 1.5;
    }

    .legend-row {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .legend-swatch {
      width: 18px;
      height: 3px;
      display: inline-block;
    }
  </style>
</head>
<body>
  <div
    id="map"
    role="region"
    aria-label="Diagnostic straight-line origin-destination map"
  ></div>

  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>

  <script>
    const tripData = __GEOJSON_PAYLOAD__;

    const categoryColors = {
      elementary_school: "#8c564b",
      healthcare: "#d62728",
      work: "#1f77b4",
      transit: "#9467bd",
      shopping: "#ff7f0e",
      recreation: "#2ca02c"
    };

    const map = L.map("map", {
      preferCanvas: true
    });

    L.tileLayer(
      "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      {
        maxZoom: 19,
        attribution:
          "&copy; OpenStreetMap contributors"
      }
    ).addTo(map);

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function popupContent(properties) {
      const miles =
        properties.straight_line_distance_m / 1609.344;

      return [
        "<strong>" +
          escapeHtml(properties.trip_type) +
          "</strong>",
        "Origin: " +
          escapeHtml(properties.origin_node),
        "Destination: " +
          escapeHtml(properties.destination_node),
        "Modeled trips: " +
          escapeHtml(properties.count),
        "Selection: " +
          escapeHtml(properties.selection_method),
        "Straight-line distance: " +
          miles.toFixed(2) +
          " miles",
        "<em>Diagnostic line only; not a routed path.</em>"
      ].join("<br>");
    }

    const layer = L.geoJSON(tripData, {
      style(feature) {
        const tripType =
          feature.properties.trip_type;

        const count = Math.max(
          Number(feature.properties.count) || 1,
          1
        );

        return {
          color:
            categoryColors[tripType] ||
            "#555555",
          weight: Math.min(
            1.2 + Math.log10(count + 1),
            6
          ),
          opacity: 0.55
        };
      },

      onEachFeature(feature, featureLayer) {
        featureLayer.bindPopup(
          popupContent(feature.properties)
        );
      }
    }).addTo(map);

    const bounds = layer.getBounds();

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [20, 20]
      });
    } else {
      map.setView([42.36, -71.06], 11);
    }

    const note = L.control({
      position: "topright"
    });

    note.onAdd = function onAdd() {
      const container =
        L.DomUtil.create(
          "div",
          "diagnostic-note"
        );

      container.innerHTML =
        "<strong>Diagnostic OD visualization</strong>" +
        "Lines connect origins and destinations " +
        "directly. They are not bicycle routes and " +
        "do not represent LTS routing results.<br><br>" +
        "Displayed OD rows: " +
        tripData.feature_count.toLocaleString() +
        "<br>Represented modeled trips: " +
        tripData.represented_trip_count.toLocaleString();

      return container;
    };

    note.addTo(map);

    const categories = Array.from(
      new Set(
        tripData.features.map(
          feature =>
            feature.properties.trip_type
        )
      )
    ).sort();

    if (categories.length > 0) {
      const legend = L.control({
        position: "bottomright"
      });

      legend.onAdd = function onAdd() {
        const container =
          L.DomUtil.create(
            "div",
            "legend"
          );

        const rows = categories.map(
          category => {
            const color =
              categoryColors[category] ||
              "#555555";

            return (
              '<div class="legend-row">' +
              '<span class="legend-swatch" ' +
              'style="background:' +
              color +
              '"></span>' +
              "<span>" +
              escapeHtml(category) +
              "</span>" +
              "</div>"
            );
          }
        );

        container.innerHTML = rows.join("");
        return container;
      };

      legend.addTo(map);
    }
  </script>
</body>
</html>
"""


def render_html(
    geojson: dict[str, Any],
) -> str:
    """Render a standalone Leaflet diagnostic map."""
    payload = json.dumps(
        geojson,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    payload = payload.replace(
        "</",
        "<\\/",
    )

    return HTML_TEMPLATE.replace(
        "__GEOJSON_PAYLOAD__",
        payload,
    )


def write_visualization_outputs(
    trips: pd.DataFrame,
    output_html: Path,
    output_geojson: Path,
) -> dict[str, Any]:
    """Write diagnostic HTML and GeoJSON outputs."""
    geojson = trips_to_geojson(trips)
    html = render_html(geojson)

    output_html.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_geojson.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    html_temporary = output_html.with_suffix(
        output_html.suffix + ".tmp"
    )

    geojson_temporary = output_geojson.with_suffix(
        output_geojson.suffix + ".tmp"
    )

    html_temporary.write_text(
        html,
        encoding="utf-8",
    )

    geojson_temporary.write_text(
        json.dumps(
            geojson,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    html_temporary.replace(output_html)
    geojson_temporary.replace(output_geojson)

    return geojson


def build_parser() -> argparse.ArgumentParser:
    """Build the visualization command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a diagnostic straight-line map "
            "from generated OD demand."
        )
    )

    parser.add_argument(
        "--trips",
        required=True,
        type=Path,
        help="Generated aggregated OD trip CSV.",
    )

    parser.add_argument(
        "--output-html",
        required=True,
        type=Path,
        help="Output interactive HTML map.",
    )

    parser.add_argument(
        "--output-geojson",
        required=True,
        type=Path,
        help="Output straight-line GeoJSON.",
    )

    parser.add_argument(
        "--maximum-trips",
        type=int,
        default=2000,
        help=(
            "Maximum number of unique OD rows "
            "included in the diagnostic map."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=26,
        help=(
            "Seed for reproducible count-weighted "
            "OD-row sampling."
        ),
    )

    return parser


def main() -> None:
    """Run the diagnostic visualization workflow."""
    args = build_parser().parse_args()

    trips = load_trip_table(args.trips)

    sampled = sample_trip_rows(
        trips,
        maximum_trips=args.maximum_trips,
        seed=args.seed,
    )

    geojson = write_visualization_outputs(
        sampled,
        args.output_html,
        args.output_geojson,
    )

    print(
        "Created diagnostic straight-line map "
        f"with {geojson['feature_count']:,} OD rows."
    )

    print(
        "Represented modeled trips: "
        f"{geojson['represented_trip_count']:,}"
    )

    print(
        "These lines are diagnostic connections, "
        "not routed bicycle paths."
    )

    print(f"HTML map: {args.output_html}")
    print(f"GeoJSON: {args.output_geojson}")


if __name__ == "__main__":
    main()
