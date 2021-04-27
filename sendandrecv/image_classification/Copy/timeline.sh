#!/bin/sh

echo $@ > timeline.txt

grep " : " $@ >> timeline.txt
cp timeline.txt $HOME/t3workspace/profile/timeline.txt

cd $HOME/t3workspace/profile/
./div.sh
