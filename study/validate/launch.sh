#!/bin/bash

for i in 32,0.9
do 
    for nx in 1120
    do  
        for eqn in euler heat
        IFS=","; 
        set -- $i; 
        export PYSWEEP_SHARE=$2
        export PYSWEEP_BLOCK=$1
        export PYSWEEP_ARRSIZE=$nx
        export PYSWEEP_EQN=$eqn
        sbatch -J "val"$PYSWEEP_EQN$1$2$nx validate.sh
    done 
done
