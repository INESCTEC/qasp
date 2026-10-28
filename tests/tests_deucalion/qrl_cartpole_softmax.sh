#!/bin/bash
#SBATCH --job-name=pl_qrl_array
#SBATCH --account=eehpc-dev-2026d02-152a
#SBATCH --partition=normal-arm
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -o out_softmax_%A_%a.out
#SBATCH -e err_softmax_%A_%a.err

ml CMake/3.29.3-GCCcore-13.3.0 \
   OpenMPI/5.0.3-GCC-13.3.0 \
   Ninja/1.12.1-GCCcore-13.3.0 \
   Python/3.12.3-GCCcore-13.3.0

source /projects/EEHPC-DEV-2026D02-152/pl-kokkos-mpi/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export N_QUBITS=6

GROUP_ID=${SLURM_ARRAY_TASK_ID}
BASE_RUN=$(( GROUP_ID * 3 ))

echo "[INFO] job=${SLURM_ARRAY_JOB_ID} group=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "[INFO] launching RUN_IDs ${BASE_RUN}, $((BASE_RUN+1)), $((BASE_RUN+2))"

srun --exclusive -N1 -n1 bash -c "export N_QUBITS=${N_QUBITS}; export RUN_ID=${BASE_RUN}; /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' -o time_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${BASE_RUN}.txt python qrl_softmax.py" &
srun --exclusive -N1 -n1 bash -c "export N_QUBITS=${N_QUBITS}; export RUN_ID=$((BASE_RUN+1)); /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' -o time_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$((BASE_RUN+1)).txt python qrl_softmax.py" &
srun --exclusive -N1 -n1 bash -c "export N_QUBITS=${N_QUBITS}; export RUN_ID=$((BASE_RUN+2)); /usr/bin/time -f 'elapsed=%E cpu=%P maxrss=%MKB' -o time_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_$((BASE_RUN+2)).txt python qrl_softmax.py" &

wait