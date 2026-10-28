#!/bin/bash
#SBATCH --job-name=bandit_softmax_q6
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --array=0-59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH -o out_bandit_softmax_q6_%A_%a.out
#SBATCH -e err_bandit_softmax_q6_%A_%a.err

ml CMake/3.29.3-GCCcore-13.3.0 \
   OpenMPI/5.0.3-GCC-13.3.0 \
   Ninja/1.12.1-GCCcore-13.3.0 \
   Python/3.12.3-GCCcore-13.3.0

source /projects/EEHPC-DEV-2026D02-152/pl-kokkos-mpi/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

MODES=(
  "Z"
  "Z_ZZ_linear"
  "Z_ZZ_all"
)

REPEATS=20

IDX=${SLURM_ARRAY_TASK_ID}
MODE_IDX=$(( IDX / REPEATS ))
RUN_ID=$(( IDX % REPEATS ))

export OBS_MODE=${MODES[$MODE_IDX]}
export RUN_ID=${RUN_ID}
export BASE_SEED=1234
export BANDIT_ENV_PATH=bandit_env_6f_20a_100c.npz

echo "[INFO] job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] OBS_MODE=${OBS_MODE} RUN_ID=${RUN_ID} ENV=${BANDIT_ENV_PATH} host=$(hostname)"

/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
    -o time_bandit_softmax_q6_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt \
    srun python qrl_bandit_softmax_q6.py