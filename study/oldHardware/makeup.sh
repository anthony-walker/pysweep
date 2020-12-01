#!/bin/bash

#SBATCH --get-user-env                      #Use user env

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p dgxs								# name of partition or queue

#SBATCH --gres=gpu:1

#SBATCH -F ./new-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --time=1-00:00:00

#SBATCH -o dgxsSweepOne.out					# name of output file for this submission script

#SBATCH -e dgxsSweepOne.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

export PYSWEEP_EQN=euler

for i in 8,0.8 12,0 12,0.1 12,0.2 12,0.3 12,0.4 16,0 24,0; 
do 
    for nx in 160 320 480 640 800 960 1120
    do  
        IFS=","; 
        set -- $i; 
        export PYSWEEP_SHARE=$2
        export PYSWEEP_BLOCK=$1
        export PYSWEEP_ARRSIZE=$nx
        sbatch -J "oldSw"$PYSWEEP_EQN$1$2 sweptm.sh
        sbatch -J "oldSt"$PYSWEEP_EQN$1$2 stdm.sh
    done 
done