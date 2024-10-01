#!/bin/bash
#SBATCH --job-name=pi_parallel_1n1c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=partition
#SBATCH --output=pi_parallel_1n1c.out
echo "Jobs are started..."
srun --mpi=pmix_v4 python3 pi_parallel_1n4c.py

