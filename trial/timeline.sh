#!/bin/sh

echo $@ > timeline.txt

grep " : " $@ >> timeline.txt
mv timeline.txt $HOME/t3workspace/profile/timeline.txt

cd $HOME/t3workspace/profile/
./div.sh
