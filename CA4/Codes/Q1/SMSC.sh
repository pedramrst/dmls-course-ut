#!/bin/bash
#SBATCH --partition=partition
#SBATCH --job-name=single_node_1
#SBATCH --mem=400
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output="/home/rostami/pytorch_ddp/experiments/1_node_1_core/slurm.log"

echo "--------------------------------------------"
echo "Nodelist=$SLURM_JOB_NODELIST"
echo "Number of nodes=$SLURM_JOB_NUM_NODES"
echo "Ntasks per node=$SLURM_NTASKS_PER_NODE"
echo "--------------------------------------------"

export WORLD_SIZE=1
export OMP_NUM_THREADS=1
echo "--------------------------------------------"

srun torchrun --standalone --nnodes=1 --nproc_per_node=1 SMSC.py --world-size 1 --log-path "/home/rostami/pytorch_ddp/experiments/1_node_1_core/logs" > "/home/rostami/pytorch_ddp/experiments/1_node_1_core/torchrun.log"