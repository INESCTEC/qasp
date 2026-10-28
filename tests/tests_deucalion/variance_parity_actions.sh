#!/bin/bash
#SBATCH --job-name=pl_variance_actions
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -o out_variance_actions_%A_%a.out
#SBATCH -e err_variance_actions_%A_%a.err

ml CMake/3.29.3-GCCcore-13.3.0 \
   OpenMPI/5.0.3-GCC-13.3.0 \
   Ninja/1.12.1-GCCcore-13.3.0 \
   Python/3.12.3-GCCcore-13.3.0

source /projects/EEHPC-DEV-2026D02-152/pl-kokkos-mpi/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

ACTION_SET=(2 4 8 16 32)

IDX=${SLURM_ARRAY_TASK_ID}
export N_ACTIONS=${ACTION_SET[$IDX]}

echo "[INFO] job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] n_actions=${N_ACTIONS} host=$(hostname)"

/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
    -o time_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt \
    srun python variance_parity_actions.py