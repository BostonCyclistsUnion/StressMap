"""Create a diagnostic map of Census population assigned to graph nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import folium
import geopandas as gpd
import pandas as pd
from folium.plugins import MarkerCluster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Census population assigned to graph nodes."
    )
    parser.add_argument("--tracts", required=True, type=Path)
    parser.add_argument("--nodes", required=True, type=Path)
    parser.add_argument("--allocation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tract-id-column", default="GEOID")
    parser.add_argument("--population-column", default="population")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    nodes = gpd.read_file(args.nodes)
    tracts = gpd.read_file(args.tracts)
    allocation = pd.read_csv(
        args.allocation,
        dtype={args.tract_id_column: str},
    )

    missing_tract_columns = {
        args.tract_id_column,
        args.population_column,
    }.difference(tracts.columns)
    if missing_tract_columns:
        raise ValueError(
            "Tract file is missing columns: "
            f"{sorted(missing_tract_columns)}"
        )

    if args.tract_id_column not in allocation.columns:
        raise ValueError(
            "Allocation file is missing column: "
            f"{args.tract_id_column}"
        )

    if "assigned_population" not in nodes.columns:
        raise ValueError(
            "Node file is missing column: assigned_population"
        )

    tracts[args.tract_id_column] = (
        tracts[args.tract_id_column].astype(str)
    )
    assigned_tract_ids = set(allocation[args.tract_id_column])
    tracts = tracts[tracts[args.tract_id_column].isin(assigned_tract_ids)]
    tracts = tracts.to_crs("EPSG:4326")

    nodes = nodes.to_crs("EPSG:4326")
    nodes = nodes[nodes["assigned_population"] > 0].copy()
    if nodes.empty:
        raise ValueError("The node file contains no positive assigned_population values.")

    map_object = folium.Map(
        location=[nodes.geometry.y.mean(), nodes.geometry.x.mean()],
        zoom_start=12,
        tiles="cartodbpositron",
        prefer_canvas=True,
    )

    folium.GeoJson(
        tracts,
        name="Assigned Census tracts",
        style_function=lambda _feature: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.0,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[
                args.tract_id_column,
                args.population_column,
            ],
            aliases=["Tract ID", "Tract population"],
            localize=True,
        ),
    ).add_to(map_object)

    low_population = nodes[nodes["assigned_population"] <= 50]
    high_population = nodes[nodes["assigned_population"] > 50]

    for _, row in low_population.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=1.5,
            weight=0,
            fill=True,
            fill_opacity=0.35,
            popup=f"Assigned population: {row['assigned_population']:.4f}",
        ).add_to(map_object)

    cluster = MarkerCluster(name="Higher assigned-population nodes").add_to(map_object)
    for _, row in high_population.iterrows():
        population = float(row["assigned_population"])
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=min(18, max(4, population**0.5 / 2)),
            weight=1,
            fill=True,
            fill_opacity=0.65,
            popup=f"Assigned population: {population:.2f}",
        ).add_to(cluster)

    folium.LayerControl().add_to(map_object)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(args.output)
    print(f"Saved: {args.output}")
    print(f"Assigned nodes shown: {len(nodes):,}")
    print(f"Assigned tracts shown: {len(tracts):,}")


if __name__ == "__main__":
    main()
