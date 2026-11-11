#!/usr/bin/env bash
#SBATCH --job-name=qasp-cartpole
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Install the same QASP (including qasp.rl.rllib) and Ray 2.60.0 on every node.
# Supply account/partition on the sbatch command line for your cluster.
set -euo pipefail
: "${QASP_PYTHON:?Absolute environment Python path required}"
: "${QASP_EXAMPLE:?Absolute path to examples/rllib_cartpole.py required}"
: "${QASP_OUTPUT:?Shared output directory required}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONSAFEPATH=1 PYTHONDONTWRITEBYTECODE=1 RAY_USAGE_STATS_ENABLED=0
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qasp-mpl-${SLURM_JOB_ID}"
unset PYTHONPATH
ray_cli="$(dirname "$QASP_PYTHON")/ray"
head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
workers="${QASP_WORKERS:-$((SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK - 1))}"
mkdir -p "$QASP_OUTPUT"
cd "$QASP_OUTPUT"
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    "$ray_cli" symmetric-run \
    --address "${head_node}:${QASP_RAY_PORT:-6379}" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus "$SLURM_CPUS_PER_TASK" --num-gpus 0 -- \
    "$QASP_PYTHON" -I -u "$QASP_EXAMPLE" \
    --address auto --algorithm "${QASP_ALGORITHM:-ppo}" \
    --workers "$workers" --iterations "${QASP_ITERATIONS:-100}" \
    --output "$QASP_OUTPUT/${QASP_ALGORITHM:-ppo}-${SLURM_JOB_ID}"
