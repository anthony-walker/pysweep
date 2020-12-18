#!/bin/bash
export PYSWEEP_EQN=euler

for i in 16,0.2 #20,0.5 16,0.3; 
do 
    for nx in 960 #160 320 480 640 800 960 1120
    do
        IFS=","; 
        set -- $i; 
        export PYSWEEP_SHARE=$2
        export PYSWEEP_BLOCK=$1
        export PYSWEEP_ARRSIZE=$nx
        #sbatch -J "newSw"$PYSWEEP_EQN$1$2$nx sweptm.sh
        sbatch -J "newSt"$PYSWEEP_EQN$1$2$nx stdm.sh
    done 
done
