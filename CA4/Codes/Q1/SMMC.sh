#!/bin/bash
#SBATCH --partition=partition
#SBATCH --job-name=single_node_2
#SBATCH --mem=400
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --output="/home/rostami/pytorch_ddp/experiments/1_node_2_cores/slurm.log"

echo "--------------------------------------------"
echo "Nodelist=$SLURM_JOB_NODELIST"
echo "Number of nodes=$SLURM_JOB_NUM_NODES"
echo "Ntasks per node=$SLURM_NTASKS_PER_NODE"
echo "--------------------------------------------"

export WORLD_SIZE=2
export OMP_NUM_THREADS=1
echo "--------------------------------------------"

srun torchrun --standalone --nnodes=1 --nproc_per_node=2 SMMC.py --world-size 2 --log-path "/home/rostami/pytorch_ddp/experiments/1_node_2_cores/logs" > "/home/rostami/pytorch_ddp/experiments/1_node_2_cores/torchrun.log"