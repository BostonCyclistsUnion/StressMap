# Opportunity accessibility analysis

This workflow measures how many jobs and amenities can be reached from each
network node under several cycling-stress profiles. It also attaches modeled
Census population and produces regional and Census-tract population-weighted
summaries.

Generated data belongs under `data/`, which is excluded from Git.

## Included files

- `accessibility/build_opportunities.py`: builds node-level job and amenity totals.
- `accessibility/reachability.py`: performs bounded directed graph searches.
- `accessibility/opportunity_access.py`: sums opportunities over reachable nodes.
- `accessibility/profile_access.py`: compares multiple cycling profiles.
- `accessibility/run_batch.py`: runs deterministic, resumable origin slices.
- `accessibility/trip_generation.py`: generates population-weighted OD demand.
- `accessibility/visualize_trips.py`: creates diagnostic OD HTML and GeoJSON outputs.
- `accessibility/merge_shards.py`: validates and merges batch outputs.
- `accessibility/population_weighting.py`: attaches Census-assigned population.
- `accessibility/population_summaries.py`: creates regional and tract summaries.
- `jobs/run_opportunity_access_array.slurm`: runs accessibility shards with Slurm.

## Required inputs

The workflow expects:

1. A pruned directed GraphML network used for accessibility analysis.
2. The corresponding source GraphML network before pruning.
3. A standardized destination CSV whose destinations are snapped to graph nodes.
4. A LODES workplace-pair CSV containing employee totals.
5. A Census tract-to-node allocation CSV containing `node_id`, `GEOID`, and
   `assigned_population`.

The graph must include `length`, the three baseline rider-cost fields, and
`max_lts`.

## Opportunity categories

The opportunity table contains jobs, schools, transit stations, bus stops,
greenspace, healthcare, and stores. Jobs represent workplace employee totals.
Amenity columns represent destination counts.

## Accessibility profiles

The analysis uses five profiles:

| Profile | Edge cost or restriction |
|---|---|
| `distance` | Edge `length` |
| `typical_adult` | `cost_typical_adult_Baseline` |
| `low_confidence_adult` | `cost_low_confidence_adult_Baseline` |
| `child` | `cost_child_Baseline` |
| `strict_lts_1_2` | Edge `length`, restricted to `max_lts <= 2` |

The default travel budget is 1.5 miles. Searches follow edge direction and use
the complete edge cost when determining whether a node is reachable.

## Build node-level opportunities

```bash
python -m accessibility.build_opportunities \
  --graph data/processed/network_pruned.graphml \
  --source-graph data/processed/network.graphml \
  --destinations data/processed/destinations.csv \
  --lodes data/processed/lodes_pairs.csv \
  --output data/processed/accessibility/opportunities_by_node.csv \
  --diagnostics data/processed/accessibility/opportunity_diagnostics.json \
  --job-reassignments data/processed/accessibility/job_reassignments.csv
```

The output has one row per analysis-graph node, including coordinates and all
seven opportunity categories. Jobs attached to nodes removed during pruning
may be reassigned to a nearby retained node. Reassignments and excluded jobs
are recorded in the audit CSV.

## Run node accessibility

Run a deterministic origin slice locally:

```bash
python -m accessibility.run_batch \
  --graph data/processed/network_pruned.graphml \
  --opportunities data/processed/accessibility/opportunities_by_node.csv \
  --output data/processed/accessibility/profile_access_slice.csv \
  --cutoff-miles 1.5 \
  --start-index 0 \
  --end-index 1000
```

Origins are sorted deterministically by node ID. Existing valid output rows
are detected, so an interrupted slice can resume without duplicating origins.

For larger networks, supply the environment variables required by the Slurm
script and submit the array job:

```bash
sbatch jobs/run_opportunity_access_array.slurm
```

Each array task writes one deterministic shard.

## Validate and merge shards

```bash
python -m accessibility.merge_shards \
  --opportunities data/processed/accessibility/opportunities_by_node.csv \
  --shard-dir data/processed/accessibility/profile_access_shards \
  --output data/processed/accessibility/profile_access_all_nodes.csv \
  --diagnostics data/processed/accessibility/profile_access_merge_diagnostics.json \
  --num-shards 32 \
  --cutoff-miles 1.5
```

The merger validates expected shard names and ranges, schemas, origin IDs,
coordinates, cutoff values, nonnegative totals, relative values, and profile
ordering constraints. It writes the merged file atomically only after all
shards pass validation.

## Attach modeled population

```bash
python -m accessibility.population_weighting \
  --accessibility data/processed/accessibility/profile_access_all_nodes.csv \
  --allocation data/processed/census/results/node_tract_allocation.csv \
  --output data/processed/accessibility/profile_access_all_nodes_with_population.csv \
  --diagnostics data/processed/accessibility/population_join_diagnostics.json
```

The tract-to-node allocation is aggregated to one population total per node.
Accessibility nodes without an allocation row are retained with zero assigned
population. Population conservation and node coverage are validated before the
output is written.

The raw tract-to-node allocation remains necessary for tract summaries because
one network node can receive population from more than one Census tract.

## Build population-weighted summaries

```bash
python -m accessibility.population_summaries \
  --nodes data/processed/accessibility/profile_access_all_nodes_with_population.csv \
  --allocation data/processed/census/results/node_tract_allocation.csv \
  --regional-output data/processed/accessibility/regional_population_weighted_accessibility.csv \
  --tract-output data/processed/accessibility/tract_population_weighted_accessibility.csv \
  --diagnostics data/processed/accessibility/population_weighted_accessibility_diagnostics.json
```

The regional output has one row for every profile and opportunity pair.
The tract output has one row for every Census tract, profile, and opportunity
combination.

## Relative-access measures

For a constrained profile, origin-level relative access is:

```text
profile opportunity access / distance opportunity access
```

This value is undefined when unrestricted distance access is zero.

Population-weighted summaries therefore report two relative measures:

- `ratio_of_weighted_means`: weighted profile access divided by weighted
  distance access.
- `population_weighted_mean_origin_relative`: weighted mean of valid
  origin-level relative values.

The first measure uses the full population in the access totals. The second
excludes population at origins where distance access is zero. The columns
`valid_relative_population`, `zero_distance_access_population`, and
`valid_relative_population_share` report that denominator coverage.

## Main outputs

### `opportunities_by_node.csv`

One row per graph node with coordinates and the seven opportunity counts.

### `profile_access_all_nodes.csv`

One row per origin node containing coordinates, cutoff values, reachable and
processed node counts, reachable opportunity totals, and constrained-profile
access relative to unrestricted distance access.

### `profile_access_all_nodes_with_population.csv`

The complete node accessibility table plus `assigned_total_population` and
`contributing_tract_count`.

### `regional_population_weighted_accessibility.csv`

One row per profile and opportunity with regional population-weighted totals,
means, relative measures, and denominator coverage.

### `tract_population_weighted_accessibility.csv`

One row per `GEOID`, profile, and opportunity. Tract calculations retain each
tract-to-node population contribution instead of assigning a shared node to
only one dominant tract.

## Validation

The pipeline validates:

- unique graph and origin node IDs;
- numeric, finite, and nonnegative opportunity and population values;
- population conservation;
- deterministic origin and shard ordering;
- profile access not exceeding unrestricted distance access;
- relative metrics within floating-point tolerance;
- agreement between regional totals and summed tract totals.

Run the test suite with:

```bash
PYTHONPATH=. python -m pytest -q tests
```

## Modeled OD demand and diagnostic maps

Population-weighted trip generation and diagnostic straight-line OD
visualization are documented in `docs/trip_generation.md`.

These modeled trips complement the cumulative opportunity-access workflow.
The diagnostic OD lines are not routed bicycle paths.

## Limitations

- Accessibility is cumulative opportunity access within a fixed network-cost
  budget, not a prediction of actual travel behavior.
- LODES employment represents workplace jobs rather than current job openings.
- Amenity totals depend on the completeness and classification of the supplied
  destination data.
- Census population is assigned to graph nodes using an area-based model and
  does not represent exact household locations.
- Results depend on graph topology, directionality, edge attributes, pruning,
  destination snapping, rider-cost assumptions, and the selected travel budget.
