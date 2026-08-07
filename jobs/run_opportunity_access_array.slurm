#!/bin/bash
#SBATCH --job-name=issue26-access
#SBATCH --partition=cpu
#SBATCH --array=0-31%8
#SBATCH --output=logs/issue26-access-%A-%a.out
#SBATCH --error=logs/issue26-access-%A-%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bcu-viz

export PYTHONPATH=.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${GRAPH:?Export GRAPH when submitting the job}"
: "${OPPORTUNITY_DIR:?Export OPPORTUNITY_DIR when submitting the job}"

NUM_SHARDS="${NUM_SHARDS:-32}"

if (( NUM_SHARDS <= 0 )); then
    echo "NUM_SHARDS must be positive." >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
    echo "Array task exceeds NUM_SHARDS." >&2
    exit 1
fi

OPPORTUNITIES="${OPPORTUNITY_DIR}/opportunities_by_node.csv"
OUTPUT_DIR="${OPPORTUNITY_DIR}/profile_access_shards"

test -f "${GRAPH}"
test -f "${OPPORTUNITIES}"

mkdir -p "${OUTPUT_DIR}"

TOTAL_ORIGINS="$(
python - <<'PY'
import os
from pathlib import Path

import pandas as pd

path = (
    Path(os.environ["OPPORTUNITY_DIR"])
    / "opportunities_by_node.csv"
)

print(
    len(
        pd.read_csv(
            path,
            usecols=["node_id"],
        )
    )
)
PY
)"

SHARD_SIZE=$((
    (TOTAL_ORIGINS + NUM_SHARDS - 1) / NUM_SHARDS
))

START_INDEX=$((
    SLURM_ARRAY_TASK_ID * SHARD_SIZE
))

END_INDEX=$((
    START_INDEX + SHARD_SIZE
))

if (( END_INDEX > TOTAL_ORIGINS )); then
    END_INDEX="${TOTAL_ORIGINS}"
fi

if (( START_INDEX >= TOTAL_ORIGINS )); then
    echo "Shard has no origins; exiting."
    exit 0
fi

printf -v SHARD_LABEL "%02d" \
    "${SLURM_ARRAY_TASK_ID}"

OUTPUT_PATH="${OUTPUT_DIR}/profile_access_shard_${SHARD_LABEL}.csv"

echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Host: $(hostname)"
echo "Started: $(date --iso-8601=seconds)"
echo "Commit: $(git rev-parse HEAD)"
echo "Graph: ${GRAPH}"
echo "Opportunities: ${OPPORTUNITIES}"
echo "Total origins: ${TOTAL_ORIGINS}"
echo "Number of shards: ${NUM_SHARDS}"
echo "Shard size: ${SHARD_SIZE}"
echo "Start index: ${START_INDEX}"
echo "End index: ${END_INDEX}"
echo "Output: ${OUTPUT_PATH}"

python -u -m accessibility.run_batch \
    --graph "${GRAPH}" \
    --opportunities "${OPPORTUNITIES}" \
    --output "${OUTPUT_PATH}" \
    --cutoff-miles 1.5 \
    --start-index "${START_INDEX}" \
    --end-index "${END_INDEX}" \
    --checkpoint-every 50 \
    --progress-every 100

echo "Finished: $(date --iso-8601=seconds)"
