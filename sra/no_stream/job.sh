#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:30:00
#$ -N vgg16_8_hybrid
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn/7.4 nccl/2.4.2

MASTER_ADDR=$(hostname -i | awk '{print $1}')

echo mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./cpp/vgg16_8_hybrid $MASTER_ADDR
mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./cpp/vgg16_8_hybrid $MASTER_ADDR
