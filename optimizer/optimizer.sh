#!/bin/bash

#printf "model?\n"
#read model
model=alexnet
printf "node gpu?\n"
read node gpu
gpus=`expr $node \* $gpu`
printf "1: theoretical, 2: actual, 3: actual(node), 4: without comm\n"
read ans
if [ $ans == "1" ];then
    machines="${gpu} ${node}"
    bw="20000000000 12500000000"
    label=theory
elif [ $ans = "2" ];then
    machines=$gpus
    bw="1800000000"
    label=actual
elif [ $ans = "3" ];then
    machines="${gpu} ${node}"
    bw="1800000000 1800000000"
    label=actual_by_node
else
    machines=$gpus
    bw="100000000000"
    label=""
fi

python optimizer_graph_hierarchical.py -f ../profiler/image_classification/profiles/$model/graph.txt -n $machines -b $bw -o partitioned/$model --use_memory_constraint -s 15000000000 --activation_compression_ratio 1.0 > tmp.txt --straight_pipeline #--use_fewer_machines

cat tmp.txt

mv tmp.txt info/${model}/"${node}_${gpu}_${label}.txt"
