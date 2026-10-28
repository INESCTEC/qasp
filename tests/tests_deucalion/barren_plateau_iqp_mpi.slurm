#!/bin/bash
#SBATCH --job-name=qasp_iqp_bp
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --array=0-3%1
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --output=out_iqp_bp_%A_%a.out
#SBATCH --error=err_iqp_bp_%A_%a.err

set -euo pipefail

ml CMake/3.29.3-GCCcore-13.3.0 \
   OpenMPI/5.0.3-GCC-13.3.0 \
   Ninja/1.12.1-GCCcore-13.3.0 \
   Python/3.12.3-GCCcore-13.3.0

source /projects/EEHPC-DEV-2026D02-152/pl-kokkos-mpi/bin/activate

QASP_ROOT=/projects/EEHPC-DEV-2026D02-152/QASP
RUN_DIR=${QASP_ROOT}/tests/tests_deucalion
PYTHON_FILE=${RUN_DIR}/barren_plateau_iqp_mpi.py

QUBITS=(28 29 30 31)
export N_QUBITS=${QUBITS[${SLURM_ARRAY_TASK_ID}]}
export N_SAMPLES=${N_SAMPLES:-32}
export N_LAYERS=${N_LAYERS:-2}
export SEED=${SEED:-19}
export OUTPUT_DIR=${OUTPUT_DIR:-${RUN_DIR}/results/barren_plateau_iqp_mpi/${SLURM_ARRAY_JOB_ID}}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export KOKKOS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONNOUSERSITE=1

cd "${RUN_DIR}"

echo "[INFO] job=${SLURM_ARRAY_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] n_qubits=${N_QUBITS} nodes=${SLURM_JOB_NUM_NODES} tasks=${SLURM_NTASKS}"
echo "[INFO] n_samples=${N_SAMPLES} n_layers=${N_LAYERS} output_dir=${OUTPUT_DIR}"
echo "[INFO] node_list=${SLURM_JOB_NODELIST}"

/usr/bin/time -f "elapsed=%E cpu=%P maxrss=%MKB" \
    -o "time_iqp_bp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" \
    srun --cpu-bind=cores python -u "${PYTHON_FILE}" \
        --n-qubits "${N_QUBITS}" \
        --n-samples "${N_SAMPLES}" \
        --n-layers "${N_LAYERS}" \
        --seed "${SEED}" \
        --output-dir "${OUTPUT_DIR}"
