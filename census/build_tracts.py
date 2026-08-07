"""Join ACS population estimates to Census tract geometry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


def build_tract_population_file(
    tract_shapefile: Path,
    acs_json: Path,
    output: Path,
) -> gpd.GeoDataFrame:
    """Join ACS total population to Census tract polygons and save GeoJSON."""
    tracts = gpd.read_file(tract_shapefile)

    with acs_json.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not raw or len(raw) < 2:
        raise ValueError("ACS JSON must contain a header row and at least one data row.")

    header, rows = raw[0], raw[1:]
    required_columns = {"NAME", "B01003_001E", "state", "county", "tract"}
    missing = required_columns.difference(header)
    if missing:
        raise ValueError(f"ACS JSON is missing required columns: {sorted(missing)}")

    acs = pd.DataFrame(rows, columns=header)
    acs["GEOID"] = acs["state"] + acs["county"] + acs["tract"]
    acs["population"] = pd.to_numeric(acs["B01003_001E"], errors="coerce")
    acs["tract_name"] = acs["NAME"]
    acs = acs[["GEOID", "tract_name", "population"]]

    tracts["GEOID"] = tracts["GEOID"].astype(str)
    result = tracts.merge(acs, on="GEOID", how="left")
    result = result[["GEOID", "tract_name", "population", "geometry"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output, driver="GeoJSON")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join ACS population estimates to Census tract geometry."
    )
    parser.add_argument(
        "--tract-shapefile",
        required=True,
        type=Path,
        help="Path to the Census TIGER/Line tract .shp file.",
    )
    parser.add_argument(
        "--acs-json",
        required=True,
        type=Path,
        help="Path to an ACS API JSON response containing B01003_001E.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output GeoJSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_tract_population_file(
        tract_shapefile=args.tract_shapefile,
        acs_json=args.acs_json,
        output=args.output,
    )
    print(f"Saved: {args.output}")
    print(f"Tracts: {len(result):,}")
    print(f"Missing population rows: {result['population'].isna().sum():,}")


if __name__ == "__main__":
    main()
