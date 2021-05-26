#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N comm__
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn/7.4 openmpi nccl/2.4.2

MASTER_ADDR=$(hostname -i | awk '{print $1}')

python nccl_communication_handler.py --master_addr localhost --rank 0 --broadcast &
python nccl_communication_handler.py --master_addr localhost --rank 1 --broadcast

#mpicxx run_test.cpp -o run_test
#mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./run_test $MASTER_ADDR
