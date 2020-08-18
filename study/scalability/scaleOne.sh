#!/bin/bash

#SBATCH -J scalSweepOne						# name of job

#SBATCH --get-user-env                      #Use user env

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p dgxs								# name of partition or queue

#SBATCH --gres=gpu:1

#SBATCH -N 1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --time=2-00:00:00

#SBATCH --mail-type=END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

echo $SLURM_JOB_ID

export RANKS_PER_NODE=1


mpiexec -n 32  pysweep -f euler -nx 960 -nt 500 -b 16 -s 0.8 --swept --verbose --ignore --clean

mpiexec -n 32  pysweep -f euler -nx 960 -nt 500 -b 16 -s 0.8 --verbose --ignore --clean

mpiexec -n 32  pysweep -f heat -nx 960 -nt 500 -b 16 -s 0.8 --swept --verbose --ignore --clean

mpiexec -n 32  pysweep -f heat -nx 960 -nt 500 -b 16 -s 0.8 --verbose --ignore --clean
