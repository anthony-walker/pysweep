#!/bin/bash

#SBATCH -J scalability						# name of job

#SBATCH --get-user-env                      #Use user env

#SBATCH -A niemeyek						# name of my sponsored account, e.g. class or research group

#SBATCH -p preempt								# name of partition or queue

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --time=2-00:00:00

#SBATCH --mail-type=END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

#SBATCH -F ./hosts/$PYSWEEP_FILE
# load any software environment module required for app

# run my jobs

#run me with sbatch --nodes=N scalability.sh
echo $SLURM_JOB_ID

SCALE_NPROC=$(($PYSWEEP_NODES*32))
SCALE_ARR=$(($PYSWEEP_NODES*960))

echo $SCALE_NPROC $SCALE_ARR

mpiexec -n $SCALE_NPROC --hostfile ./hosts/$PYSWEEP_FILE  pysweep -f $PYSWEEP_EQN -nx $SCALE_ARR -nt 500 -b 16 -s 0.8 --swept --verbose --ignore --clean

mpiexec -n $SCALE_NPROC --hostfile ./hosts/$PYSWEEP_FILE  pysweep -f $PYSWEEP_EQN -nx $SCALE_ARR -nt 500 -b 16 -s 0.8 --verbose --ignore --clean

