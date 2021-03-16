#!/bin/bash
a=1
for name in one two three four
do
    for eqn in heat euler
    do
        export PYSWEEP_NODES=$a
        export PYSWEEP_FILE=$name
        sbatch -N $a --nodefile ./hosts/$name scalability.sh
    done
    a=$((a+1))
done