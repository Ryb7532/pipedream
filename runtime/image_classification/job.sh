#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N nccl_test
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn/7.4 nccl/2.4.2

MASTER_ADDR=$(hostname -i | awk '{print $1}')

echo mpirun -x LD_LIBRARY_PATH -x PATH -n 1 -npernode 1 --bind-to none ./cpp/alexnet_1_4_theory_hybrid $MASTER_ADDR
export NCCL_DEBUG=INFO 
mpirun -x LD_LIBRARY_PATH -x PATH -n 1 -npernode 1 --bind-to none ./cpp/alexnet_1_4_theory_hybrid $MASTER_ADDR
