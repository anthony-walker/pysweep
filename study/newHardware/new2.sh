#!/bin/bash

#SBATCH -J test2						# name of job

#SBATCH --get-user-env                      #Use user env

#SBATCH -A niemeyek 

#SBATCH -p dgxs

#SBATCH --gres=gpu:1

#SBATCH -F ./new-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --time=2-00:00:00

#SBATCH -o test2.out					# name of output file for this submission script

#SBATCH -e test2.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

echo $SLURM_JOB_ID

export RANKS_PER_NODE=1

for nt in 200 400 600 800 1000
do
    mpiexec -n 32 --hostfile ./new-nodes pysweep -f euler -nx 640 -nt $nt -b 16 -s 0.5 --swept --verbose --ignore --clean

    mpiexec -n 32 ./new-nodes pysweep -f euler -nx 640 -nt $nt -b 16 -s 0.5 --verbose --ignore -- clean

    mpiexec -n 32 ./new-nodes pysweep -f heat -nx 640 -nt $nt -b 16 -s 0.5 --swept --verbose --ignore --clean

    mpiexec -n 32 ./new-nodes pysweep -f heat -nx 640 -nt $nt -b 16 -s 0.5 --verbose --ignore --clean
done

