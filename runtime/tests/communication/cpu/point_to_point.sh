#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00
#$ -N cpu_gloo

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn/7.4 openmpi nccl/2.4.2

MASTER_ADDR=$(hostname -i | awk '{print $1}')

python point_to_point.py --backend gloo --master_addr localhost --rank 0 &
python point_to_point.py --backend gloo --master_addr localhost --rank 1 
mpicxx run_cpu_test.cpp -o run_cpu_test
mpirun -x LD_LIBRARY_PATH -x PATH -n 2 -npernode 1 --bind-to none ./run_cpu_test $MASTER_ADDR
