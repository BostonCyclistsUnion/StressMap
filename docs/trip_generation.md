# Population-weighted trip generation

This workflow creates reproducible origin-destination demand from
Census-assigned population and node-level jobs and amenities.

Generated CSV, JSON, HTML, and GeoJSON files belong under `data/` or another
external processed-data directory. Generated data is not committed to Git.

## Inputs

The trip generator requires:

1. A GraphML bicycle network containing node longitude and latitude.
2. A Census tract-to-node allocation CSV containing:
   - `node_id`
   - `assigned_population`
3. A node opportunity CSV containing:
   - `node_id`
   - `jobs`
   - `schools`
   - `transit_stations`
   - `bus_stops`
   - `greenspace`
   - `healthcare`
   - `stores`
4. A demand-scenario CSV.

## Supported trip types

| Trip type | Destination pool | Selection method |
|---|---|---|
| `elementary_school` | Schools | Nearest eligible destination |
| `healthcare` | Healthcare | Nearest eligible destination |
| `work` | Jobs | Opportunity- and distance-weighted |
| `transit` | Transit stations and bus stops | Opportunity- and distance-weighted |
| `shopping` | Stores | Opportunity- and distance-weighted |
| `recreation` | Greenspace | Opportunity- and distance-weighted |

Origins are sampled according to assigned Census population.

Weighted destinations use both the opportunity count and a lognormal
straight-line distance model. A fixed random seed makes the generated,
aggregated OD output reproducible.

Repeated selections of the same origin, destination, and trip type are stored
as one row with a positive integer `count`.

## Generate OD demand

Run:

    python -m accessibility.trip_generation \
      --graph data/processed/network.graphml \
      --allocation data/processed/census/node_tract_allocation.csv \
      --opportunities data/processed/accessibility/opportunities_by_node.csv \
      --demand-config accessibility/config/demand_parameters.csv \
      --demand-scenario 1 \
      --seed 26 \
      --output data/processed/accessibility/trips_scenario1_seed26.csv \
      --diagnostics data/processed/accessibility/trips_scenario1_seed26.json

The diagnostics file reports:

- requested and generated trip totals;
- generated totals by trip type;
- usable population origins;
- destination-pool sizes and weights;
- empty destination pools;
- distance-model parameters;
- and the random seed.

## Create a diagnostic spatial map

Run:

    python -m accessibility.visualize_trips \
      --trips data/processed/accessibility/trips_scenario1_seed26.csv \
      --output-html data/processed/accessibility/trips_scenario1_seed26.html \
      --output-geojson data/processed/accessibility/trips_scenario1_seed26.geojson \
      --maximum-trips 2000 \
      --seed 26

When the trip table contains more rows than the display limit, OD rows are
selected reproducibly using their modeled trip counts as sampling weights.

The map lines are direct origin-destination connections used to inspect the
generated demand. They are not routed bicycle paths and do not represent LTS
route geometries.

Cumulative opportunity access under unrestricted distance, rider-cost, and
strict LTS profiles is documented in `docs/opportunity_accessibility.md`.

## Validation

The implementation validates:

- required input columns;
- unique and valid graph-node identifiers;
- nonnegative finite population and opportunity values;
- positive integer trip counts;
- exact conservation of configured trip totals;
- valid coordinates and distances;
- reproducibility under a fixed seed;
- and explicit diagnostic labeling of non-routed map geometry.

Run the relevant tests with:

    PYTHONPATH=. python -m pytest -q \
      tests/test_trip_generation.py \
      tests/test_visualize_trips.py \
      tests/test_accessibility_workflow.py
