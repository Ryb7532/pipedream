#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N test_comm_comp
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn/7.4 openmpi nccl/2.4.2

MASTER_ADDR=$(hostname -i | awk '{print $1}')

export NCCL_DEBUG=info
python comm_comp.py --master_addr localhost --rank 0 &
python comm_comp.py --master_addr localhost --rank 1 &
python comm_comp.py --master_addr localhost --rank 2 &
python comm_comp.py --master_addr localhost --rank 3
