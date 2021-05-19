#!/bin/bash

for eqn in heat euler
do
    export a=1
    for name in one two three four
    do
        
            export PYSWEEP_NODES=$a
            export PYSWEEP_FILE=$name
            export PYSWEEP_EQN=$eqn
            sbatch -N $a --nodefile ./hosts/$name scalability.sh
        
        a=$((a+1))
    done
done
