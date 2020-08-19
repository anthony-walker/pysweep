#!/bin/bash
a=1
for name in one two three four five six seven 
do
    export PYSWEEP_NODES=$a
    export PYSWEEP_FILE=$name
    sbatch -N $a --nodefile $name scalability.sh
    a=$((a+1))
done