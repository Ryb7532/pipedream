#!/bin/bash

printf "which model?\n"
read model
if [ "$model" = "" ]; then
    model="alexnet"
elif [ "$model" = "-h" -o "$model" = "-H" -o "$model" = "-help" ]; then
    printf "
alexnet | densenet121 | densenet161 | densenet169 | densenet201 |
inception_v3 | resnet101 | resnet152 | resnet18 | resnet34 | resnet50
| squeezenet1_0 | squeezenet1_1 | vgg11 | vgg11_bn | vgg13 | vgg13_bn
| vgg16 | vgg16_bn | vgg19 | vgg19_bn | mobilenet | nasnetalarge
| nasnetamobile | resnext101 | resnext152 | resnext18 | resnext50
\n"
    printf "which model?\n"
    read model
fi

printf "how much batch size?\n"
read bs
if [ "$bs" = "" ]; then
    bs=256
fi


data_dir="--data_dir /gs/hs1/tga-sssml/17B13541/images/imagenet"
echo "#!/bin/sh
#$ -cwd
#$ -l s_gpu=1
#$ -l h_rt=00:10:00
#$ -N ${model}_${bs}
#$ -v GPU_COMPUTE_MODE=0
. /etc/profile.d/modules.sh
module load python/3.6.5 cuda/10.1.105 openmpi cudnn/7.4 nccl/2.4.2 texlive
CUDA_VISIBLE_DEVICES=0 python main.py -a ${model} -b ${bs} -s
" > job.sh

qsub job.sh
