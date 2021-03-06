#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N p2p_gloo
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn openmpi nccl

MASTER_ADDR=$(hostname -i | awk '{print $1}')

python point_to_point.py --backend gloo --master_addr localhost --rank 0 &
python point_to_point.py --backend gloo --master_addr localhost --rank 1
#mpicxx run_p2p_test.cpp -o run_p2p_test
#mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./run_p2p_test $MASTER_ADDR
