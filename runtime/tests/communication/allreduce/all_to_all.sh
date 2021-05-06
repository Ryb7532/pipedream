#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=00:10:00
#$ -N allreduce_test
#$ -v GPU_COMPUTE_MODE=0

. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 cudnn/7.4 openmpi nccl/2.4.2

MASTER_ADDR=$(hostname -i | awk '{print $1}')

#nvidia-smi topo -m

<< COMMENTOUT
echo "world size: 2"
python all_to_all.py --backend nccl --master_addr localhost --rank 0 --local_rank 0 --world_size 2 &
python all_to_all.py --backend nccl --master_addr localhost --rank 1 --local_rank 1 --world_size 2

echo "world size: 3"
python all_to_all.py --backend nccl --master_addr localhost --rank 0 --local_rank 0 --world_size 3 &
python all_to_all.py --backend nccl --master_addr localhost --rank 1 --local_rank 1 --world_size 3 &
python all_to_all.py --backend nccl --master_addr localhost --rank 2 --local_rank 2 --world_size 3

COMMENTOUT

echo "world size: 4"
python all_to_all.py --backend nccl --master_addr localhost --rank 0 --local_rank 0 --world_size 4 &
python all_to_all.py --backend nccl --master_addr localhost --rank 1 --local_rank 1 --world_size 4 &
python all_to_all.py --backend nccl --master_addr localhost --rank 2 --local_rank 2 --world_size 4 &
python all_to_all.py --backend nccl --master_addr localhost --rank 3 --local_rank 3 --world_size 4
#COMMENTOUT

<< COMMENTOUT
mpicxx run_allreduce.cpp -o run_allreduce
for i in 2
do
    echo "world size: ${i}"
    mpirun -x LD_LIBRARY_PATH -x PATH -n $i -npernode 1 --bind-to none ./run_allreduce $MASTER_ADDR
done
COMMENTOUT
