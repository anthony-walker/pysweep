#!/bin/bash

#SBATCH -J scalability						# name of job

#SBATCH --get-user-env                      #Use user env

#SBATCH -A nrg						# name of my sponsored account, e.g. class or research group

#SBATCH -p preempt								# name of partition or queue

#SBATCH --gres=gpu:1

#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=16

#SBATCH --time=2-00:00:00

#SBATCH --mail-type=END,FAIL				# send email when job begins, ends or aborts

#SBATCH --mail-user=walkanth@oregonstate.edu		# send email to this address

# load any software environment module required for app

# run my jobs

#run me with sbatch --nodes=N scalability.sh
SCALE_NPROC=$(($PYSWEEP_NODES*32))
if [ $PYSWEEP_NODES -eq 1 ]; then 
    SCALE_ARR=1424
elif [ $PYSWEEP_NODES -eq 2 ]; then
    SCALE_ARR=2000
elif [ $PYSWEEP_NODES -eq 3 ]; then
    SCALE_ARR=2448
elif [ $PYSWEEP_NODES -eq 4 ]; then
    SCALE_ARR=2832
elif [ $PYSWEEP_NODES -eq 5 ]; then
    SCALE_ARR=3162
elif [ $PYSWEEP_NODES -eq 6 ]; then
    SCALE_ARR=3464
else
    SCALE_ARR=3742
fi

echo $SLURM_JOB_ID $PYSWEEP_NODES $SCALE_NPROC $SCALE_ARR $PYSWEEP_EQN

mpiexec -n $SCALE_NPROC --hostfile ./hosts/$PYSWEEP_FILE  pysweep -f $PYSWEEP_EQN -nx $SCALE_ARR -nt 500 -b 16 -s 0.9 --swept --verbose --ignore --clean

mpiexec -n $SCALE_NPROC --hostfile ./hosts/$PYSWEEP_FILE  pysweep -f $PYSWEEP_EQN -nx $SCALE_ARR -nt 500 -b 16 -s 0.9 --verbose --ignore --clean

