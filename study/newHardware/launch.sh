#!/bin/bash
for bs in 20 #8 12 16 20 24 32
do
    for gs in $(seq 0 0.10 1)
    do
        export PYSWEEP_SHARE=$gs
        export PYSWEEP_BLOCK=$bs
        sbatch -J "new"$bs$gs new.sh
    done
done