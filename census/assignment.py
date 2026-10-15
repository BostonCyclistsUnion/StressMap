"""Assign Census tract population to graph nodes."""

import pandas as pd
from pyproj import CRS
from shapely.geometry import MultiPoint
from shapely.ops import voronoi_diagram
from scipy.spatial import cKDTree


def assign_population_to_nodes_by_tract_area(
    nodes_gdf,
    tracts_gdf,
    population_col="population",
    tract_id_col="GEOID",
    projected_crs="EPSG:26986",
    candidate_buffer_m=0,
    tract_filter_method="convex_hull",
    region_boundary_gdf=None,
    min_region_overlap_share=0.0,
    verbose=False,
):
    """
    Assign census tract population to graph nodes deterministically.

    For each tract:
    1. Find candidate graph nodes inside the tract, or within a buffer around it.
    2. Build Voronoi-style nearest-node regions from those candidate nodes.
    3. Clip each Voronoi region to the tract itself.
    4. Assign each clipped region to its nearest node using a KDTree.
    5. Normalize area shares so each tract sums exactly to 1.
    6. Assign population proportionally by normalized area share.

    Parameters
    ----------
    candidate_buffer_m : float, default 0
        Buffer distance in meters around each tract used to find nearby candidate nodes.
        Use 0 to only consider nodes inside/intersecting the tract.
        Use values like 50 or 100 to allow nearby outside nodes to receive population
        if they are closest to part of the tract.

    tract_filter_method : {"convex_hull", "envelope", "none"}, default "convex_hull"
        Method used to pre-filter census tracts before assignment.
        "convex_hull" keeps tracts intersecting the graph node convex hull.
        "envelope" keeps the old rectangular bounding-box behavior.
        "none" skips this pre-filter.

    region_boundary_gdf : geopandas.GeoDataFrame, optional
        Optional actual region/city boundary polygon.
        If provided, this boundary is used instead of tract_filter_method.

    min_region_overlap_share : float, default 0.0
        Minimum share of a census tract's area that must overlap the region
        boundary to keep the tract. Use 0.0 to keep any intersecting tract.
        Use values like 0.01, 0.05, 0.10, or 0.50 to remove tracts that only
        barely touch the boundary.

    Output diagnostics
    ------------------
    raw_area_share : float
        The unnormalized clipped Voronoi area share for a node within a tract.

    tract_coverage_ratio : float
        Sum of raw_area_share values for a tract before normalization.
        Values should be very close to 1.0. If much lower than 1.0,
        normalization may be hiding incomplete tract coverage.
    """

    nodes = nodes_gdf.copy()
    tracts = tracts_gdf.copy()

    if nodes.crs is None:
        raise ValueError("nodes_gdf must have a CRS.")

    if tracts.crs is None:
        raise ValueError("tracts_gdf must have a CRS.")

    required_columns = {population_col, tract_id_col}
    missing_columns = required_columns.difference(tracts.columns)
    if missing_columns:
        raise ValueError(
            f"Tract data is missing required columns: {sorted(missing_columns)}"
        )

    if tracts[tract_id_col].isna().any():
        raise ValueError(f"{tract_id_col} contains missing values.")

    tract_ids_as_text = tracts[tract_id_col].astype(str)
    if tract_ids_as_text.duplicated().any():
        duplicate_ids = sorted(
            tract_ids_as_text[tract_ids_as_text.duplicated()].unique()
        )
        raise ValueError(
            f"{tract_id_col} contains duplicate values: {duplicate_ids}"
        )

    population_values = pd.to_numeric(
        tracts[population_col],
        errors="coerce",
    )

    if population_values.isna().any():
        raise ValueError(
            f"{population_col} must contain only valid numeric values."
        )

    if (population_values < 0).any():
        raise ValueError(
            f"{population_col} must not contain negative values."
        )

    tracts[population_col] = population_values

    candidate_buffer_m = float(candidate_buffer_m)
    min_region_overlap_share = float(min_region_overlap_share)

    unassigned_rows = []

    def record_unassigned(tract, reason):
        unassigned_rows.append(
            {
                tract_id_col: tract[tract_id_col],
                population_col: tract[population_col],
                "reason": reason,
            }
        )

    if candidate_buffer_m < 0:
        raise ValueError("candidate_buffer_m must be non-negative.")

    if min_region_overlap_share < 0 or min_region_overlap_share > 1:
        raise ValueError("min_region_overlap_share must be between 0 and 1.")

    nodes["node_id"] = nodes.index

    target_crs = CRS.from_user_input(projected_crs)

    if nodes.crs != target_crs:
        nodes = nodes.to_crs(target_crs)

    if tracts.crs != target_crs:
        tracts = tracts.to_crs(target_crs)

    if region_boundary_gdf is not None:
        region_boundary = region_boundary_gdf.copy()

        if region_boundary.crs != target_crs:
            region_boundary = region_boundary.to_crs(target_crs)

        region_boundary = region_boundary.geometry.union_all()

    elif tract_filter_method == "convex_hull":
        region_boundary = nodes.geometry.union_all().convex_hull

    elif tract_filter_method == "envelope":
        region_boundary = nodes.geometry.union_all().envelope

    elif tract_filter_method in ("none", None):
        region_boundary = None

    else:
        raise ValueError(
            "tract_filter_method must be one of: "
            "'convex_hull', 'envelope', or 'none'."
        )

    if region_boundary is not None:
        intersects_region = tracts.geometry.intersects(region_boundary)

        for _, tract in tracts.loc[~intersects_region].iterrows():
            record_unassigned(tract, "outside_region")

        tracts = tracts.loc[intersects_region].copy()

        if min_region_overlap_share > 0 and not tracts.empty:
            tract_areas = tracts.geometry.area
            overlap_areas = tracts.geometry.intersection(region_boundary).area
            overlap_share = overlap_areas / tract_areas
            meets_overlap = overlap_share >= min_region_overlap_share

            for _, tract in tracts.loc[~meets_overlap].iterrows():
                record_unassigned(
                    tract,
                    "below_minimum_region_overlap",
                )

            tracts = tracts.loc[meets_overlap].copy()

    node_sindex = nodes.sindex
    allocation_rows = []

    total_tracts = len(tracts)

    for tract_counter, (_, tract) in enumerate(tracts.iterrows(), start=1):
        if verbose and (tract_counter == 1 or tract_counter % 25 == 0):
            print(f"Processing tract {tract_counter}/{total_tracts}")

        tract_geom = tract.geometry

        if tract_geom is None or tract_geom.is_empty:
            record_unassigned(tract, "missing_or_empty_geometry")
            continue

        tract_area = tract_geom.area

        if tract_area == 0:
            record_unassigned(tract, "zero_area_geometry")
            continue

        candidate_geom = (
            tract_geom.buffer(candidate_buffer_m)
            if candidate_buffer_m > 0
            else tract_geom
        )

        candidate_idx = list(node_sindex.query(candidate_geom, predicate="intersects"))
        tract_nodes = nodes.iloc[candidate_idx].copy()

        if tract_nodes.empty:
            record_unassigned(tract, "no_candidate_nodes")
            continue

        tract_nodes = tract_nodes[tract_nodes.geometry.intersects(candidate_geom)].copy()

        if tract_nodes.empty:
            record_unassigned(tract, "no_candidate_nodes")
            continue

        tract_nodes = tract_nodes.sort_values("node_id")

        tract_rows = []

        if len(tract_nodes) == 1:
            tract_rows.append(
                {
                    "node_id": tract_nodes.iloc[0]["node_id"],
                    tract_id_col: tract[tract_id_col],
                    "area_share": 1.0,
                }
            )

        else:
            points = list(tract_nodes.geometry)
            coords = [(point.x, point.y) for point in points]
            multipoint = MultiPoint(coords)

            vor = voronoi_diagram(
                multipoint,
                envelope=candidate_geom.envelope,
                edges=False,
            )

            node_ids = list(tract_nodes["node_id"])
            tree = cKDTree(coords)

            area_by_node = {}

            for cell in vor.geoms:
                clipped = cell.intersection(tract_geom)

                if clipped.is_empty:
                    continue

                clipped_area = clipped.area

                if clipped_area == 0:
                    continue

                rep = clipped.representative_point()
                _, nearest_pos = tree.query((rep.x, rep.y))
                node_id = node_ids[nearest_pos]

                area_by_node[node_id] = area_by_node.get(node_id, 0.0) + clipped_area

            for node_id, clipped_area in area_by_node.items():
                tract_rows.append(
                    {
                        "node_id": node_id,
                        tract_id_col: tract[tract_id_col],
                        "area_share": clipped_area / tract_area,
                    }
                )

        for row in tract_rows:
            row["raw_area_share"] = row["area_share"]

        share_total = sum(row["raw_area_share"] for row in tract_rows)

        if share_total == 0:
            record_unassigned(tract, "zero_voronoi_coverage")
            continue

        for row in tract_rows:
            row["tract_coverage_ratio"] = share_total

            normalized_share = row["raw_area_share"] / share_total
            row["area_share"] = normalized_share
            row["assigned_population"] = tract[population_col] * normalized_share
            allocation_rows.append(row)

    allocation_columns = [
        "node_id",
        tract_id_col,
        "area_share",
        "raw_area_share",
        "tract_coverage_ratio",
        "assigned_population",
    ]
    allocation = pd.DataFrame(
        allocation_rows,
        columns=allocation_columns,
    )

    unassigned_columns = [
        tract_id_col,
        population_col,
        "reason",
    ]
    unassigned = pd.DataFrame(
        unassigned_rows,
        columns=unassigned_columns,
    )

    if not unassigned.empty:
        unassigned = unassigned.sort_values(
            [tract_id_col, "reason"],
            key=lambda series: series.astype(str),
        ).reset_index(drop=True)

    if allocation.empty:
        output = nodes.copy()
        output["assigned_population"] = 0
        return output, allocation, unassigned

    allocation = allocation.sort_values(
        [tract_id_col, "node_id"]
    ).reset_index(drop=True)

    node_population = allocation.groupby(
        "node_id",
        as_index=False,
    )["assigned_population"].sum()

    output = nodes.merge(
        node_population,
        on="node_id",
        how="left",
    )
    output["assigned_population"] = (
        output["assigned_population"].fillna(0)
    )

    return output, allocation, unassigned
