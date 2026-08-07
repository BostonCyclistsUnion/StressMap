# Census-to-node population assignment

This workflow joins American Community Survey population estimates to official Census tract geometry and assigns tract population to graph nodes using deterministic area-based allocation.

## Included files

- `census/build_tracts.py`: joins ACS total population to Census tract geometry.
- `census/assignment.py`: reusable population-assignment algorithm.
- `census/run_assignment.py`: command-line runner for a GraphML network.
- `census/visualize_assignment.py`: optional diagnostic HTML map.
- `tests/test_census_assignment.py`: population-conservation and input-validation tests.

## Required source data

The workflow expects:

1. An official Census TIGER/Line tract shapefile.
2. An ACS API JSON response containing `NAME`, `B01003_001E`, `state`, `county`, and `tract`.
3. An OSMnx-compatible GraphML network.

Source data and generated outputs belong under `data/`, which is excluded from Git.

## Build the tract population file

```bash
python -m census.build_tracts \
  --tract-shapefile data/raw/census/ma_tracts_2024/tl_2024_25_tract.shp \
  --acs-json data/raw/census/ma_acs_population_2024.json \
  --output data/processed/census/ma_tracts_population.geojson
```

## Assign population to nodes

```bash
python -m census.run_assignment \
  --graph data/processed/network.graphml \
  --tracts data/processed/census/ma_tracts_population.geojson \
  --output-dir data/processed/census/results \
  --output-prefix greater_boston
```

An optional boundary file can be supplied with `--boundary`. Without one, the graph-node convex hull is used to select intersecting tracts.

## Create a diagnostic map

```bash
python -m census.visualize_assignment \
  --tracts data/processed/census/ma_tracts_population.geojson \
  --nodes data/processed/census/results/greater_boston_nodes_web.geojson \
  --allocation data/processed/census/results/greater_boston_node_tract_allocation.csv \
  --output data/processed/census/results/greater_boston_assignment_map.html
```

## Method

For each tract, the assignment algorithm:

1. Projects graph nodes and tracts into a meter-based coordinate system.
2. Finds graph nodes inside or near the tract.
3. Builds Voronoi-style nearest-node regions.
4. Clips each region to the tract polygon.
5. calculates each node's share of tract area.
6. Normalizes shares so each assigned tract sums to one.
7. Assigns tract population proportionally to those normalized shares.
8. Sorts results by tract ID and node ID for reproducibility.

The allocation output includes `area_share`, `raw_area_share`, `tract_coverage_ratio`, and `assigned_population` for validation.

## Limitations

This is an area-based approximation. It does not represent the actual building-level or household-level distribution of residents inside a tract. The output should be interpreted as modeled population assigned to nearby network nodes.
