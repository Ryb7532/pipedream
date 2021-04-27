#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:03:00
#$ -N allreduce_gloo
#$ -v GPU_COMPUTE_MODE=1

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn openmpi nccl

MASTER_ADDR=$(hostname -i | awk '{print $1}')

python all_to_all.py --backend gloo --master_addr localhost --rank 0 --local_rank 0 --world_size 2 &
python all_to_all.py --backend gloo --master_addr localhost --rank 1 --local_rank 1 --world_size 2
mpicxx run_allreduce.cpp -o run_allreduce
mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./run_allreduce $MASTER_ADDR
