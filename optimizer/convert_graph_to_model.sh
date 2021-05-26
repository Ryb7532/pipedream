#!/bin/bash

#printf "which model?\n"
#read model
model=vgg16

printf "how many GPUs?\n"
read gpus

printf "allocate stage to ranks?\n"
read stnr
if [ "${stnr}" = "" ];then
    stnr="0:1"
    for ((i=1;i<$gpus;i++))
    do
	stnr=$stnr",${i}:1"
    done
fi

echo $stnr

printf "file name?\n"
read file

python convert_graph_to_model.py -f partitioned/$model/gpus=$gpus.txt -n ${model}Partitioned -a $model -o gpus=$file --stage_to_num_ranks $stnr
cp -r gpus=$file ../runtime/image_classification/models/$model/.
cp -r gpus=$file ../improve/image_classification/models/$model/.
cp -r gpus=$file ../sendandrecv/image_classification/models/$model/.
rm -rf gpus=$file
