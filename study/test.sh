#!/bin/bash

#SBATCH -J gtxSweepOne						# name of job

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p preempt								# name of partition or queue

#SBATCH -F ./old-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20

#SBATCH --time=7-00:00:00

#SBATCH -o gtxSweepOne.out					# name of output file for this submission script

#SBATCH -e gtxSweepOne.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

echo $SLURM_JOB_ID

# mpiexec -n 40 --hostfile ./nrg-nodes pysweep -f euler -nx 1344 -nt 100 -b 16 -s 0.5 --swept --verbose --ignore

mpiexec -n 1 --hostfile $HPC_PATH/pysweep-git/nrg-nodes pysweep -f heat -nx 1344 -nt 100 -b 16 -s 1 --swept --verbose --ignore