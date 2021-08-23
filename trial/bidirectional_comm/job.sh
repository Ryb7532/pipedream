#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N bidirec_comm_test
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load cuda/10.2.89 openmpi nccl


#NCCL_DEBUG=INFO 
mpirun -x LD_LIBRARY_PATH -x PATH -n 4 -npernode 4 --bind-to none ./a.out
