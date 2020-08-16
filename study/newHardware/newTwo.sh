#!/bin/bash

#SBATCH -J dgxsSweepTwo						# name of job

#SBATCH --get-user-env                      #Use user env

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p dgxs								# name of partition or queue

#SBATCH --gres=gpu:1

#SBATCH -F ./new-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=20

#SBATCH --time=7-00:00:00

#SBATCH -o dgxsSweepTwo.out					# name of output file for this submission script

#SBATCH -e dgxsSweepTwo.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

echo $SLURM_JOB_ID

for eq in heat euler
do
    for bs in 8 12 16 24 32
    do
        for gs in $(seq 0 0.10 1)
        do
            for nx in 160 320 480 640 800 960 1120
            do
                    mpiexec -n 40 --hostfile ./new-nodes pysweep -f $eq -nx $nx -nt 500 -b $bs -s $gs --swept --verbose --ignore --clean

                    mpiexec -n 40 --hostfile ./new-nodes pysweep -f $eq -nx $nx -nt 500 -b $bs -s $gs --verbose --ignore --clean
            done
        done
    done
done
