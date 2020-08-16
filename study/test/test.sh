#!/bin/bash

#SBATCH -J testPysweep						# name of job

#SBATCH --get-user-env                      #Use user env

#SBATCH -A niemeyek 

#SBATCH -p dgx

#SBATCH --gres=gpu:1

#SBATCH -N 2

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --time=2-00:00:00

#SBATCH -o testPysweep.out					# name of output file for this submission script

#SBATCH -e testPysweep.err					# name of error file for this submission script

#SBATCH --mail-type=BEGIN,END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

echo $SLURM_JOB_ID

mpiexec -n 32 pysweep -f euler -nx 640 -nt 500 -b 16 -s 0.5 --swept --verbose --ignore

mv eulerOutput.hdf5 /nfs/hpc/share/nrg/walkanth/eulerOutputSwept.hdf5

mpiexec -n 32 pysweep -f euler -nx 640 -nt 500 -b 16 -s 0.5 --verbose --ignore

mv eulerOutput.hdf5 /nfs/hpc/share/nrg/walkanth/eulerOutputStandard.hdf5

mpiexec -n 32 pysweep -f heat -nx 640 -nt 500 -b 16 -s 0.5 --swept --verbose --ignore

mv heatOutput.hdf5 /nfs/hpc/share/nrg/walkanth/heatOutputSwept.hdf5

mpiexec -n 32 pysweep -f heat -nx 640 -nt 500 -b 16 -s 0.5 --verbose --ignore

mv heatOutput.hdf5 /nfs/hpc/share/nrg/walkanth/heatOutputStandard.hdf5

