#!/usr/bin/python
import math, os
batch = """#!/bin/tcsh
    ### myBatchSubmit
    ### LSF syntax
    #BSUB -nnodes {:d}                   #number of nodes
    #BSUB -W 72:00                       #walltime in hours:minutes
    #BSUB -G orsu                     #account
    #BSUB -e pysweep-log-{:d}.txt              #stderr
    #BSUB -o pysweep-out-{:d}.txt              #stdout
    #BSUB -J pysweep-{:d}               #name of job
    #BSUB -q pbatch                   #queue to use

    ### Shell scripting
    hostname
    echo -n 'JobID is '; echo $LSB_JOBID
    
    #Set env
    conda activate pysweep-dev
    
    #Set num gpus
    export GPUS_PER_NODE=1

    ### Launch parallel executable
    echo 'Launching executable...'
    echo 'Time at computation start:'
    date
    echo ' '
    jsrun -n{:d} -r1 -a40 -c40 -g1 pysweep -f euler -nx {:d} -nt 2000 -b 16 -s 1 --swept --verbose --ignore --clean

    jsrun -n{:d} -r1 -a40 -c40 -g1 pysweep -f euler -nx {:d} -nt 2000 -b 16 -s 1 --verbose --ignore --clean

    jsrun -n{:d} -r1 -a40 -c40 -g1 pysweep -f heat -nx {:d} -nt 2000 -b 16 -s 1 --swept --verbose --ignore --clean

    jsrun -n{:d} -r1 -a40 -c40 -g1 pysweep -f heat -nx {:d} -nt 2000 -b 16 -s 1 --verbose --ignore --clean
    echo 'Time at computation finish:'
    date
    echo 'Done'
    
    """

files = []
node_numbers = [1,]+[i for i in range(2,36,2)]
for i in node_numbers:
    npts = int(math.sqrt(2000000*i))
    fname =  'pysweep-batch-{:d}'.format(i)
    files.append(fname)   
    with open(fname,'w') as f:
        f.write(batch.format(i,i,i,i,i,npts,i,npts,i,npts,i,npts))

for f in files:
    os.system("bsub<"+fname)