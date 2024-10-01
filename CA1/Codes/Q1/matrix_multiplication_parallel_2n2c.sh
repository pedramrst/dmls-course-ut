#!/bin/bash
#SBATCH --job-name=matrix_multiplication_parallel_2n2c
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --partition=partition
#SBATCH --mem=400
#SBATCH --output=matrix_multiplication_parallel_2n2c.out
echo "Jobs are started..."
srun --mpi=pmix_v4 python3 matrix_multiplication_parallel_2n2c.py

