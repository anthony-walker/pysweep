#!/bin/bash

#SBATCH -J testPysweep						# name of job

#SBATCH --get-user-env                      #Use user env

#SBATCH -A niemeyek 

#SBATCH -p preempt

#SBATCH --gres=gpu:1

#SBATCH -F ./test-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20

#SBATCH --time=7-00:00:00

#SBATCH -o testPysweep.out					# name of output file for this submission script

#SBATCH -e testPysweep.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

echo $SLURM_JOB_ID

mpiexec -n 40 --hostfile=./old-nodes pysweep -f euler -nx 1344 -nt 500 -b 16 -s 0.5 --swept --verbose --ignore

mpiexec -n 40 --hostfile=./old-nodes pysweep -f euler -nx 1344 -nt 500 -b 16 -s 0.5 --verbose --ignore

mv log.yaml test.yaml
