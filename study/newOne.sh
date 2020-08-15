#!/bin/bash

#SBATCH -J hrf						# name of job

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p mime4								# name of partition or queue

#SBATCH -F ./nrg-nodes

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=6

#SBATCH --time=7-00:00:00

#SBATCH -o hrf.out					# name of output file for this submission script

#SBATCH -e hrf.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

echo $SLURM_JOB_ID

mpiexec -n 12 python cluster.py
for eq in heat euler
do
    for bs in 8 12 16 24 32
    do
        for gs in $(seq 0 0.10 1)
        do
            for nx in  1344 2624 3904 5184 6464
            do

                    mpiexec -n 40 --hostfile ./nrg-nodes pysweep -f $eq -nx $nx -nt 1000 -b $bs -s $gs --swept --verbose --ignore --clean
            done
        done
    done
done