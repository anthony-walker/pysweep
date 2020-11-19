#!/bin/bash

#SBATCH --get-user-env                      #Use user env

#SBATCH -A nrg					# name of my sponsored account, e.g. class or research group

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

echo $SLURM_JOB_ID

export RANKS_PER_NODE=1

mpiexec -n 32 --hostfile ./new-nodes pysweep -f $PYSWEEP_EQN -nx $PYSWEEP_ARRSIZE -nt 500 -b $PYSWEEP_BLOCK -s $PYSWEEP_SHARE --verbose --ignore --clean