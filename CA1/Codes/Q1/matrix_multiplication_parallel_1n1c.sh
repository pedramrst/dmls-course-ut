#!/bin/bash
#SBATCH --job-name=matrix_multiplication_parallel_1n1c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=partition
#SBATCH --mem=400
#SBATCH --output=matrix_multiplication_parallel_1n1c.out
echo "Jobs are started..."
srun --mpi=pmix_v4 python3 matrix_multiplication_parallel_1n1c.py

