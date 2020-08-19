#!/bin/bash
for bs in 8 12 16 20 24 32
do
    for gs in $(seq 0 0.10 1)
    do
        export PYSWEEP_SHARE=$gs
        export PYSWEEP_BLOCK=$bs
        sbatch -J "more"$PYSWEEP_EQN$bs$gs more.sh
    done
done

export PYSWEEP_SHARE=0.5
export PYSWEEP_BLOCK=20
export PYSWEEP_ARR=4800
sbatch -J "time"$PYSWEEP_EQN$bs$gs time.sh