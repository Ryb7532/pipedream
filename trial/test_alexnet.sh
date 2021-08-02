#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N test
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn/7.4 nccl/2.4.2

#export NCCL_DEBUG=INFO
#export NCCL_P2P_DISABLE=1
#export NCCL_LAUNCH_MODE=PARALLEL
./python.sh -m launch --nnodes 1 --node_rank 0 --nproc_per_node 4 test_alexnet.py -s --epochs 6 --num_ranks_in_server 4 --master_addr localhost
