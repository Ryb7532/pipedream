#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:03:00
#$ -N test_gloo
#$ -v GPU_COMPUTE_MODE=1

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn openmpi nccl

MASTER_ADDR=$(hostname -i | awk '{print $1}')

python gloo_communication_handler.py --master_addr localhost --rank 0 &
python gloo_communication_handler.py --master_addr localhost --rank 1
#mpicxx run_gloo_test.cpp -o run_gloo_test
#mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./run_gloo_test $MASTER_ADDR
