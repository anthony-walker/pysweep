#!/bin/bash
a=4
for name in four five #one two three
do
    # for eqn in heat euler
    # do
        export PYSWEEP_NODES=$a
        export PYSWEEP_FILE=$name
        export PYSWEEP_EQN=heat #$eqn
        sbatch -N $a --nodefile ./hosts/$name scalability.sh
    # done
    a=$((a+1))
done