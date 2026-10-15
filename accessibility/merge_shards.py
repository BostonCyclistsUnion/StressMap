"""Validate and merge deterministic accessibility shard CSVs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from accessibility.opportunity_access import OPPORTUNITY_COLUMNS
from accessibility.profile_access import DEFAULT_PROFILE_SPECS
from accessibility.reachability import DEFAULT_CUTOFF_MILES, miles_to_meters
from accessibility.run_batch import load_origin_table, output_columns


def expected_shard_ranges(total_origins: int, num_shards: int):
    """Return deterministic half-open shard ranges."""
    if total_origins < 0:
        raise ValueError("total_origins must be nonnegative")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    size = math.ceil(total_origins / num_shards) if total_origins else 0
    return tuple(
        (index * size, min((index + 1) * size, total_origins))
        for index in range(num_shards)
    )


def _numeric(frame, column, shard_index, allow_missing=False):
    raw = frame[column]
    values = pd.to_numeric(raw, errors="coerce")

    if (raw.notna() & values.isna()).any():
        raise ValueError(
            f"Shard {shard_index} column {column!r} is non-numeric"
        )

    if not allow_missing and values.isna().any():
        raise ValueError(
            f"Shard {shard_index} column {column!r} has missing values"
        )

    if not np.isfinite(
        values.dropna().to_numpy(dtype=float)
    ).all():
        raise ValueError(
            f"Shard {shard_index} column {column!r} is non-finite"
        )

    return values.astype(float)


def _not_greater(smaller, larger, label, shard_index):
    small = smaller.to_numpy(dtype=float)
    large = larger.to_numpy(dtype=float)

    bad = (small > large) & ~np.isclose(
        small,
        large,
        rtol=1e-12,
        atol=1e-9,
    )

    if bad.any():
        row = int(np.flatnonzero(bad)[0])

        raise ValueError(
            f"Shard {shard_index} violates {label} at row {row}"
        )


def _validate_shard(
    path,
    shard_index,
    expected_origins,
    columns,
    cutoff_miles,
    cutoff_meters,
    profile_names,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing shard file: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype={"origin_node": "string"},
    )

    if frame.columns.tolist() != columns:
        raise ValueError(
            f"Shard {shard_index} schema mismatch"
        )

    if len(frame) != len(expected_origins):
        raise ValueError(
            f"Shard {shard_index} has {len(frame)} rows; "
            f"expected {len(expected_origins)}"
        )

    ids = (
        frame["origin_node"]
        .astype("string")
        .str.strip()
    )

    expected_ids = (
        expected_origins["node_id"]
        .astype(str)
        .tolist()
    )

    if (
        ids.isna().any()
        or ids.eq("").any()
        or ids.duplicated().any()
    ):
        raise ValueError(
            f"Shard {shard_index} has invalid origin IDs"
        )

    if ids.astype(str).tolist() != expected_ids:
        raise ValueError(
            f"Shard {shard_index} origin ordering mismatch"
        )

    frame["origin_node"] = ids

    relative_columns = {
        column
        for column in columns
        if column.endswith("_relative")
    }

    for column in columns:
        if column != "origin_node":
            frame[column] = _numeric(
                frame,
                column,
                shard_index,
                allow_missing=column in relative_columns,
            )

    for coordinate in ("longitude", "latitude"):
        if not np.allclose(
            frame[coordinate],
            expected_origins[coordinate],
            rtol=0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Shard {shard_index} {coordinate} mismatch"
            )

    if not np.allclose(
        frame["cutoff_miles"],
        cutoff_miles,
        atol=1e-12,
    ):
        raise ValueError(
            f"Shard {shard_index} cutoff_miles mismatch"
        )

    if not np.allclose(
        frame["cutoff_meters"],
        cutoff_meters,
        atol=1e-9,
    ):
        raise ValueError(
            f"Shard {shard_index} cutoff_meters mismatch"
        )

    for profile in profile_names:
        for measure in (
            "reachable_node_count",
            "processed_node_count",
        ):
            column = f"{profile}_{measure}"
            values = frame[column].to_numpy(dtype=float)

            if (
                (values < 0).any()
                or not np.allclose(
                    values,
                    np.round(values),
                )
            ):
                raise ValueError(
                    f"Shard {shard_index} invalid count "
                    f"column {column}"
                )

        for opportunity in OPPORTUNITY_COLUMNS:
            column = f"{profile}_{opportunity}"

            if (frame[column] < 0).any():
                raise ValueError(
                    f"Shard {shard_index} negative totals "
                    f"in {column}"
                )

    for profile in profile_names:
        if profile == "distance":
            continue

        for measure in (
            "reachable_node_count",
            "processed_node_count",
        ):
            _not_greater(
                frame[f"{profile}_{measure}"],
                frame[f"distance_{measure}"],
                (
                    f"{profile}_{measure} <= "
                    f"distance_{measure}"
                ),
                shard_index,
            )

        for opportunity in OPPORTUNITY_COLUMNS:
            numerator_column = (
                f"{profile}_{opportunity}"
            )

            denominator_column = (
                f"distance_{opportunity}"
            )

            relative_column = (
                f"{profile}_{opportunity}_relative"
            )

            _not_greater(
                frame[numerator_column],
                frame[denominator_column],
                (
                    f"{numerator_column} <= "
                    f"{denominator_column}"
                ),
                shard_index,
            )

            numerator = frame[
                numerator_column
            ].to_numpy(dtype=float)

            denominator = frame[
                denominator_column
            ].to_numpy(dtype=float)

            relative = frame[
                relative_column
            ].to_numpy(dtype=float)

            zero = np.isclose(
                denominator,
                0,
                atol=1e-12,
            )

            if not np.isnan(
                relative[zero]
            ).all():
                raise ValueError(
                    f"Shard {shard_index} "
                    f"{relative_column} must be missing "
                    "when baseline is zero"
                )

            if np.isnan(
                relative[~zero]
            ).any():
                raise ValueError(
                    f"Shard {shard_index} "
                    f"{relative_column} is missing"
                )

            if not np.allclose(
                relative[~zero],
                (
                    numerator[~zero]
                    / denominator[~zero]
                ),
                rtol=1e-10,
                atol=1e-10,
            ):
                raise ValueError(
                    f"Shard {shard_index} "
                    f"{relative_column} is incorrect"
                )

            present = relative[
                ~np.isnan(relative)
            ]

            if (
                (present < -1e-12).any()
                or (present > 1 + 1e-12).any()
            ):
                raise ValueError(
                    f"Shard {shard_index} "
                    f"{relative_column} outside [0, 1]"
                )

    if all(
        name in profile_names
        for name in (
            "typical_adult",
            "low_confidence_adult",
            "child",
        )
    ):
        for measure in (
            "reachable_node_count",
            "processed_node_count",
            *OPPORTUNITY_COLUMNS,
        ):
            _not_greater(
                frame[
                    f"low_confidence_adult_{measure}"
                ],
                frame[
                    f"typical_adult_{measure}"
                ],
                (
                    f"low_confidence_adult_{measure} <= "
                    f"typical_adult_{measure}"
                ),
                shard_index,
            )

            _not_greater(
                frame[f"child_{measure}"],
                frame[
                    f"low_confidence_adult_{measure}"
                ],
                (
                    f"child_{measure} <= "
                    f"low_confidence_adult_{measure}"
                ),
                shard_index,
            )

    return frame, {
        "shard_index": shard_index,
        "path": str(path),
        "row_count": len(frame),
        "first_origin": (
            expected_ids[0]
            if expected_ids
            else None
        ),
        "last_origin": (
            expected_ids[-1]
            if expected_ids
            else None
        ),
    }


def validate_and_merge_shards(
    opportunities_path,
    shard_dir,
    output_path,
    *,
    num_shards=32,
    cutoff_miles=DEFAULT_CUTOFF_MILES,
    diagnostics_path=None,
):
    """Validate all shards and write one merged CSV."""
    profiles = DEFAULT_PROFILE_SPECS

    profile_names = tuple(
        profile.name
        for profile in profiles
    )

    origins = load_origin_table(
        Path(opportunities_path)
    )

    columns = output_columns(profiles)

    ranges = expected_shard_ranges(
        len(origins),
        num_shards,
    )

    cutoff_meters = miles_to_meters(
        cutoff_miles
    )

    frames = []
    diagnostics = []

    for index, (start, end) in enumerate(ranges):
        path = (
            Path(shard_dir)
            / f"profile_access_shard_{index:02d}.csv"
        )

        frame, details = _validate_shard(
            path,
            index,
            origins.iloc[start:end],
            columns,
            float(cutoff_miles),
            cutoff_meters,
            profile_names,
        )

        frames.append(frame)
        diagnostics.append(details)

        print(
            f"Validated shard {index:02d}: "
            f"{len(frame):,} rows"
        )

    merged = pd.concat(
        frames,
        ignore_index=True,
    )

    expected_ids = (
        origins["node_id"]
        .astype(str)
        .tolist()
    )

    if len(merged) != len(origins):
        raise ValueError(
            "Merged row count mismatch"
        )

    if merged[
        "origin_node"
    ].duplicated().any():
        raise ValueError(
            "Merged dataset contains duplicate origins"
        )

    if (
        merged["origin_node"]
        .astype(str)
        .tolist()
        != expected_ids
    ):
        raise ValueError(
            "Merged origin ordering mismatch"
        )

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = output_path.with_name(
        output_path.name + ".tmp"
    )

    merged.to_csv(
        temporary,
        index=False,
    )

    os.replace(
        temporary,
        output_path,
    )

    if diagnostics_path is not None:
        diagnostics_path = Path(
            diagnostics_path
        )

        diagnostics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = diagnostics_path.with_name(
            diagnostics_path.name + ".tmp"
        )

        payload = {
            "num_shards": num_shards,
            "row_count": len(merged),
            "column_count": len(
                merged.columns
            ),
            "cutoff_miles": float(
                cutoff_miles
            ),
            "cutoff_meters": cutoff_meters,
            "output_path": str(output_path),
            "shards": diagnostics,
        }

        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        os.replace(
            temporary,
            diagnostics_path,
        )

    return merged


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and merge "
            "opportunity-access shards."
        )
    )

    parser.add_argument(
        "--opportunities",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--shard-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--diagnostics",
        type=Path,
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--cutoff-miles",
        type=float,
        default=DEFAULT_CUTOFF_MILES,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    merged = validate_and_merge_shards(
        opportunities_path=(
            args.opportunities
        ),
        shard_dir=args.shard_dir,
        output_path=args.output,
        num_shards=args.num_shards,
        cutoff_miles=args.cutoff_miles,
        diagnostics_path=args.diagnostics,
    )

    print()
    print("SHARD MERGE VALIDATION PASSED")
    print(f"Shards: {args.num_shards:,}")
    print(f"Rows: {len(merged):,}")
    print(
        f"Columns: "
        f"{len(merged.columns):,}"
    )
    print(f"Output: {args.output}")

    if args.diagnostics is not None:
        print(
            f"Diagnostics: "
            f"{args.diagnostics}"
        )


if __name__ == "__main__":
    main()
