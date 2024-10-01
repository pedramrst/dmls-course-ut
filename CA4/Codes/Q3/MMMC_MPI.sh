#!/bin/bash
#SBATCH --partition=partition
#SBATCH --job-name=multi_node_2_m
#SBATCH --mem=400
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --output="/home/rostami/pytorch_ddp/experiments/2_nodes_2_cores_m/slurm.log"

echo "--------------------------------------------"
echo "Nodelist=$SLURM_JOB_NODELIST"
echo "Number of nodes=$SLURM_JOB_NUM_NODES"
echo "Ntasks per node=$SLURM_NTASKS_PER_NODE"
echo "--------------------------------------------"

export WORLD_SIZE=4
export MASTER_PORT=23456
export RENDEZVOUS_ID=$RANDOM
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR:MASTER_PORT=$MASTER_ADDR:$MASTER_PORT"
export OMP_NUM_THREADS=1
echo "--------------------------------------------"

mpirun -np 4 --bind-to core --report-bindings --display-allocation python MMMC_MPI.py --world-size 4 --log-path "/home/rostami/pytorch_ddp/experiments/2_nodes_2_cores_m/logs" --backend mpi > "/home/rostami/pytorch_ddp/experiments/2_nodes_2_cores_m/torchrun.log"

