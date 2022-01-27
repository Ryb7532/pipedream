#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N alexnet_8_dp_intra_ib_disable

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn/7.4 nccl/2.4.2

MASTER_ADDR=$(hostname -i | awk '{print $1}')
export NCCL_IB_DISABLE=1

echo mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./cpp/alexnet_8_dp_intra $MASTER_ADDR
#export NCCL_DEBUG=INFO 
mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./cpp/alexnet_8_dp_intra $MASTER_ADDR
