#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N allreduce

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn/7.4 openmpi nccl/2.4.2

export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1
export NCCL_IB_TIMEOUT=18

MASTER_ADDR=$(hostname -i | awk '{print $1}')
proc=4
node=2

mpicxx scatter_node.cpp

echo "nodes: ${node}, npernode: ${proc}"

mpirun -x LD_LIBRARY_PATH -x PATH -n $node -npernode 1 --bind-to none ./a.out $MASTER_ADDR $proc
