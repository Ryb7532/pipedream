#!/bin/sh

grep Epoch_ $1 | grep -v Epoch_0 | sed 's/ seconds//g' | sed -e 's/^.*: //g'
