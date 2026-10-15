"""Run resumable opportunity-access calculations for origin slices."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import time

import networkx as nx
import numpy as np
import pandas as pd

from accessibility.opportunity_access import (
    OPPORTUNITY_COLUMNS,
    OpportunityIndex,
)
from accessibility.profile_access import (
    DEFAULT_PROFILE_SPECS,
    ProfileSpec,
    calculate_multi_profile_accessibility,
)
from accessibility.reachability import (
    DEFAULT_CUTOFF_MILES,
    miles_to_meters,
)


REQUIRED_ORIGIN_COLUMNS = (
    "node_id",
    "longitude",
    "latitude",
)


@dataclass(frozen=True)
class BatchRunSummary:
    """Summary of one resumable batch invocation."""

    selected_origin_count: int
    previously_completed_count: int
    written_count: int
    output_path: Path
    elapsed_seconds: float


def output_columns(
    profiles: tuple[ProfileSpec, ...] = DEFAULT_PROFILE_SPECS,
) -> list[str]:
    """Return the deterministic output CSV schema."""
    columns = [
        "origin_node",
        "longitude",
        "latitude",
        "cutoff_miles",
        "cutoff_meters",
    ]

    for profile in profiles:
        prefix = profile.name

        columns.extend(
            [
                f"{prefix}_reachable_node_count",
                f"{prefix}_processed_node_count",
            ]
        )

        for opportunity in OPPORTUNITY_COLUMNS:
            columns.append(f"{prefix}_{opportunity}")

            if profile.name != "distance":
                columns.append(
                    f"{prefix}_{opportunity}_relative"
                )

    return columns


def load_origin_table(path: Path) -> pd.DataFrame:
    """Load and validate origin nodes and coordinates."""
    frame = pd.read_csv(
        path,
        dtype={"node_id": "string"},
    )

    missing = set(REQUIRED_ORIGIN_COLUMNS).difference(
        frame.columns
    )

    if missing:
        raise ValueError(
            "Opportunity table is missing origin columns: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame["node_id"] = (
        frame["node_id"]
        .astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    if frame["node_id"].isna().any():
        raise ValueError("Origin node IDs may not be missing.")

    if frame["node_id"].eq("").any():
        raise ValueError("Origin node IDs may not be blank.")

    if frame["node_id"].duplicated().any():
        raise ValueError("Origin node IDs must be unique.")

    for column in ("longitude", "latitude"):
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError(
                f"Origin column {column!r} must contain "
                "only finite numeric values."
            )

        frame[column] = values

    return frame.sort_values(
        "node_id",
        kind="mergesort",
    ).reset_index(drop=True)


def select_origin_slice(
    origins: pd.DataFrame,
    start_index: int = 0,
    end_index: int | None = None,
) -> pd.DataFrame:
    """Select a deterministic half-open origin index range."""
    if start_index < 0:
        raise ValueError("start_index must be nonnegative.")

    if end_index is not None:
        if end_index < 0:
            raise ValueError("end_index must be nonnegative.")

        if end_index < start_index:
            raise ValueError(
                "end_index must be greater than or equal "
                "to start_index."
            )

    return origins.iloc[start_index:end_index].copy()


def load_completed_origins(
    output_path: Path,
    expected_columns: list[str],
    valid_origins: set[str],
    cutoff_miles: float,
    cutoff_meters: float,
) -> set[str]:
    """Validate an existing output and return completed origins."""
    if not output_path.exists():
        return set()

    if output_path.stat().st_size == 0:
        return set()

    completed = set()

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != expected_columns:
            raise ValueError(
                "Existing output schema does not match the "
                "current batch schema."
            )

        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"Malformed output row {row_number}."
                )

            missing_values = [
                column
                for column in expected_columns
                if row.get(column) in (None, "")
            ]

            if missing_values:
                raise ValueError(
                    f"Output row {row_number} has blank fields: "
                    f"{missing_values[:10]}"
                )

            origin = str(row["origin_node"]).strip()

            if origin not in valid_origins:
                raise ValueError(
                    f"Output row {row_number} contains unknown "
                    f"origin {origin!r}."
                )

            if origin in completed:
                raise ValueError(
                    f"Output contains duplicate origin {origin!r}."
                )

            try:
                row_cutoff_miles = float(row["cutoff_miles"])
                row_cutoff_meters = float(row["cutoff_meters"])
            except ValueError as exc:
                raise ValueError(
                    f"Output row {row_number} has invalid cutoff "
                    "values."
                ) from exc

            if not math.isclose(
                row_cutoff_miles,
                cutoff_miles,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Existing output uses a different cutoff_miles "
                    f"value at row {row_number}."
                )

            if not math.isclose(
                row_cutoff_meters,
                cutoff_meters,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "Existing output uses a different cutoff_meters "
                    f"value at row {row_number}."
                )

            completed.add(origin)

    return completed


def build_origin_record(
    graph: nx.Graph,
    opportunities: OpportunityIndex,
    origin_node: str,
    longitude: float,
    latitude: float,
    cutoff_miles: float,
    cutoff_meters: float,
    profiles: tuple[ProfileSpec, ...] = DEFAULT_PROFILE_SPECS,
) -> dict:
    """Calculate and flatten all profiles for one origin."""
    result = calculate_multi_profile_accessibility(
        graph=graph,
        opportunities=opportunities,
        origin_node=origin_node,
        budget=cutoff_meters,
        profiles=profiles,
    )

    profile_record = result.to_record()

    profile_record.pop("origin_node")
    profile_record.pop("budget")

    return {
        "origin_node": origin_node,
        "longitude": float(longitude),
        "latitude": float(latitude),
        "cutoff_miles": float(cutoff_miles),
        "cutoff_meters": float(cutoff_meters),
        **profile_record,
    }


def run_batch(
    graph_path: Path,
    opportunities_path: Path,
    output_path: Path,
    *,
    cutoff_miles: float = DEFAULT_CUTOFF_MILES,
    start_index: int = 0,
    end_index: int | None = None,
    checkpoint_every: int = 1,
    progress_every: int = 1,
    profiles: tuple[ProfileSpec, ...] = DEFAULT_PROFILE_SPECS,
) -> BatchRunSummary:
    """Run or resume one deterministic origin slice."""
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive.")

    if progress_every <= 0:
        raise ValueError("progress_every must be positive.")

    started = time.perf_counter()
    cutoff_meters = miles_to_meters(cutoff_miles)

    origin_table = load_origin_table(opportunities_path)
    selected = select_origin_slice(
        origin_table,
        start_index=start_index,
        end_index=end_index,
    )

    expected_columns = output_columns(profiles)
    valid_origins = set(origin_table["node_id"])

    completed = load_completed_origins(
        output_path=output_path,
        expected_columns=expected_columns,
        valid_origins=valid_origins,
        cutoff_miles=float(cutoff_miles),
        cutoff_meters=cutoff_meters,
    )

    selected_ids = set(selected["node_id"])
    previously_completed = selected_ids.intersection(completed)

    pending = selected[
        ~selected["node_id"].isin(completed)
    ].copy()

    print(f"Selected origins: {len(selected):,}")
    print(
        "Previously completed in slice:",
        f"{len(previously_completed):,}",
    )
    print(f"Pending origins: {len(pending):,}")

    if pending.empty:
        return BatchRunSummary(
            selected_origin_count=len(selected),
            previously_completed_count=len(
                previously_completed
            ),
            written_count=0,
            output_path=output_path,
            elapsed_seconds=(
                time.perf_counter() - started
            ),
        )

    opportunities = OpportunityIndex.from_frame(
        origin_table
    )

    print("Loading graph...")
    graph_started = time.perf_counter()
    graph = nx.read_graphml(graph_path)

    print(
        "Graph loaded in "
        f"{time.perf_counter() - graph_started:.1f} seconds"
    )

    absent_from_graph = sorted(
        set(pending["node_id"]).difference(graph.nodes)
    )

    if absent_from_graph:
        raise ValueError(
            "Selected origins are absent from the graph. "
            f"Examples: {absent_from_graph[:10]}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_header = (
        not output_path.exists()
        or output_path.stat().st_size == 0
    )

    written = 0

    with output_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=expected_columns,
            extrasaction="raise",
        )

        if write_header:
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())

        for row in pending.itertuples(index=False):
            record = build_origin_record(
                graph=graph,
                opportunities=opportunities,
                origin_node=str(row.node_id),
                longitude=float(row.longitude),
                latitude=float(row.latitude),
                cutoff_miles=float(cutoff_miles),
                cutoff_meters=cutoff_meters,
                profiles=profiles,
            )

            writer.writerow(record)
            written += 1

            if written % checkpoint_every == 0:
                handle.flush()
                os.fsync(handle.fileno())

            if (
                written % progress_every == 0
                or written == len(pending)
            ):
                print(
                    f"Completed {written:,} of "
                    f"{len(pending):,} pending origins"
                )

        handle.flush()
        os.fsync(handle.fileno())

    return BatchRunSummary(
        selected_origin_count=len(selected),
        previously_completed_count=len(
            previously_completed
        ),
        written_count=written,
        output_path=output_path,
        elapsed_seconds=time.perf_counter() - started,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable multi-profile opportunity access "
            "for a deterministic origin slice."
        )
    )

    parser.add_argument(
        "--graph",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--opportunities",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--cutoff-miles",
        type=float,
        default=DEFAULT_CUTOFF_MILES,
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--end-index",
        type=int,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = run_batch(
        graph_path=args.graph,
        opportunities_path=args.opportunities,
        output_path=args.output,
        cutoff_miles=args.cutoff_miles,
        start_index=args.start_index,
        end_index=args.end_index,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
    )

    print()
    print(f"Written this run: {summary.written_count:,}")
    print(f"Output: {summary.output_path}")
    print(
        "Elapsed seconds:",
        f"{summary.elapsed_seconds:.1f}",
    )


if __name__ == "__main__":
    main()
