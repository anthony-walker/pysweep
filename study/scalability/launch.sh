#!/bin/bash
a=3
for name in three four #one two three four five
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