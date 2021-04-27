#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:30:00
#$ -N alexnet_4_straight_mp
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn nccl

MASTER_ADDR=$(hostname -i | awk '{print $1}')

echo mpirun -x LD_LIBRARY_PATH -x PATH -n 1 -npernode 1 --bind-to none ./cpp/alexnet_4_straight_mp $MASTER_ADDR
mpirun -x LD_LIBRARY_PATH -x PATH -n 1 -npernode 1 --bind-to none ./cpp/alexnet_4_straight_mp $MASTER_ADDR
