#!/bin/bash
export PYSWEEP_EQN=euler

for i in 24,0.6 #8,0.8 12,0 12,0.1 12,0.2 12,0.3 12,0.4 16,0 24,0; 
do 
    for nx in 1120 #160 320 480 640 800 960 1120
    do  
        IFS=","; 
        set -- $i; 
        export PYSWEEP_SHARE=$2
        export PYSWEEP_BLOCK=$1
        export PYSWEEP_ARRSIZE=$nx
        sbatch -J "oldSw"$PYSWEEP_EQN$1$2$nx sweptm.sh
        sbatch -J "oldSt"$PYSWEEP_EQN$1$2$nx stdm.sh
    done 
done
