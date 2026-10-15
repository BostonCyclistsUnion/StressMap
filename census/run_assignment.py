"""Assign Census tract population to graph nodes."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

from census.assignment import assign_population_to_nodes_by_tract_area


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign Census tract population to graph nodes by tract area."
    )
    parser.add_argument("--graph", required=True, type=Path, help="Input GraphML file.")
    parser.add_argument(
        "--tracts",
        required=True,
        type=Path,
        help="GeoJSON or other vector file containing tract geometry and population.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated node and allocation files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="census_assignment",
        help="Prefix used for generated filenames.",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        help="Optional region boundary file. If omitted, the graph-node convex hull is used.",
    )
    parser.add_argument("--population-column", default="population")
    parser.add_argument("--tract-id-column", default="GEOID")
    parser.add_argument("--projected-crs", default="EPSG:26986")
    parser.add_argument("--candidate-buffer-m", type=float, default=100.0)
    parser.add_argument("--minimum-boundary-overlap", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph: {args.graph}")
    graph = ox.load_graphml(args.graph)
    nodes, _ = ox.graph_to_gdfs(graph)

    print(f"Loading tracts: {args.tracts}")
    tracts = gpd.read_file(args.tracts)

    missing_columns = {
        args.population_column,
        args.tract_id_column,
    }.difference(tracts.columns)
    if missing_columns:
        raise ValueError(f"Tract file is missing columns: {sorted(missing_columns)}")

    boundary = gpd.read_file(args.boundary) if args.boundary else None

    print("Assigning population...")
    nodes_with_population, allocation, unassigned = (
        assign_population_to_nodes_by_tract_area(
            nodes,
            tracts,
            population_col=args.population_column,
            tract_id_col=args.tract_id_column,
            projected_crs=args.projected_crs,
            candidate_buffer_m=args.candidate_buffer_m,
            tract_filter_method="convex_hull",
            region_boundary_gdf=boundary,
            min_region_overlap_share=args.minimum_boundary_overlap,
            verbose=True,
        )
    )

    prefix = args.output_prefix
    nodes_gpkg = args.output_dir / f"{prefix}_nodes.gpkg"
    nodes_parquet = args.output_dir / f"{prefix}_nodes.parquet"
    nodes_web = args.output_dir / f"{prefix}_nodes_web.geojson"
    allocation_csv = args.output_dir / f"{prefix}_node_tract_allocation.csv"
    allocation_parquet = args.output_dir / f"{prefix}_node_tract_allocation.parquet"
    unassigned_csv = args.output_dir / f"{prefix}_unassigned_tracts.csv"
    unassigned_parquet = args.output_dir / f"{prefix}_unassigned_tracts.parquet"

    nodes_with_population.to_file(nodes_gpkg, driver="GPKG")
    nodes_with_population.to_parquet(nodes_parquet)
    nodes_with_population.to_crs("EPSG:4326").to_file(nodes_web, driver="GeoJSON")
    allocation.to_csv(allocation_csv, index=False)
    allocation.to_parquet(allocation_parquet, index=False)
    unassigned.to_csv(unassigned_csv, index=False)
    unassigned.to_parquet(unassigned_parquet, index=False)

    excluded_reasons = {
        "outside_region",
        "below_minimum_region_overlap",
    }

    excluded = unassigned[
        unassigned["reason"].isin(excluded_reasons)
    ].copy()

    selected_unassigned = unassigned[
        ~unassigned["reason"].isin(excluded_reasons)
    ].copy()

    assigned_tract_ids = set(
        allocation[args.tract_id_column].astype(str)
    )
    selected_unassigned_ids = set(
        selected_unassigned[args.tract_id_column].astype(str)
    )
    selected_tract_ids = (
        assigned_tract_ids | selected_unassigned_ids
    )

    tract_ids = tracts[args.tract_id_column].astype(str)
    populations = pd.to_numeric(
        tracts[args.population_column],
        errors="raise",
    )

    input_population = populations.sum()
    selected_population = populations[
        tract_ids.isin(selected_tract_ids)
    ].sum()
    node_population = nodes_with_population[
        "assigned_population"
    ].sum()
    unassigned_population = (
        selected_population - node_population
    )

    print("Done.")
    print(f"Input tracts: {len(tracts):,}")
    print(f"Excluded tracts: {len(excluded):,}")
    print(f"Selected tracts: {len(selected_tract_ids):,}")
    print(
        "Assigned tracts: "
        f"{allocation[args.tract_id_column].nunique():,}"
    )
    print(
        "Unassigned selected tracts: "
        f"{len(selected_unassigned):,}"
    )
    print(f"Allocation rows: {len(allocation):,}")
    print(
        "Assigned nodes: "
        f"{(nodes_with_population['assigned_population'] > 0).sum():,}"
    )
    print(f"Input population: {input_population:,.6f}")
    print(
        f"Selected population: {selected_population:,.6f}"
    )
    print(
        f"Assigned node population: {node_population:,.6f}"
    )
    print(
        f"Unassigned selected population: "
        f"{unassigned_population:,.6f}"
    )
    print(f"Saved outputs under: {args.output_dir}")


if __name__ == "__main__":
    main()
