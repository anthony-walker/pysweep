#!/bin/bash
a=1
for name in one two three four five six seven 
do
    for i in 32,1 8,0 #8,0.8 12,0 12,0.1 12,0.2 12,0.3 12,0.4 16,0 24,0; 
    do 
    for eqn in heat euler
    do
        for 
    export PYSWEEP_NODES=$a
    export PYSWEEP_FILE=$name

    sbatch -N $a --nodefile ./hosts/$name scalability.sh
    a=$((a+1))
        done
    done
done