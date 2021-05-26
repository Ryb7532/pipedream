#!/bin/sh
#$ -cwd
#$ -l s_gpu=1
#$ -l h_rt=00:10:00
#$ -N resnet50_128
#$ -v GPU_COMPUTE_MODE=0
. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn/7.4 nccl/2.4.2 texlive
CUDA_VISIBLE_DEVICES=0 python main.py -a resnet50 -b 128 -s

