#!/bin/bash

for ((i=0; i<$2; i++))
do
    echo $i >> out.txt
    for pass in Forward Backward
    do
	echo $pass >> out.txt
	for label in rec comp send total
	do
	    grep ms $1 | grep "Rank ${i}," | grep $label | grep $pass | grep -v "Epoch 0" | sed s/$'\t'//g | sed -e 's/(Epoch //g' -e 's/Rank //g' > tmp.txt
	    ./analysis.o >> out.txt
	done
    done
    for label in _queue _sync
    do
	grep ms $1 | grep "Rank ${i} " | grep $label > tmp0.txt
	total=$(cat tmp0.txt | wc -l)
	n=`expr ${total} / 6 \* 5`
	cat tmp0.txt | tail -n $n > tmp.txt
	./analysis.o >> out.txt
    done
    grep Ave $1 | grep "Rank ${i} " >> out.txt
done

mv out.txt "${1}.txt"
rm tmp*
