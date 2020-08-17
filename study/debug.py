import numpy, mpi4py.MPI as MPI, warnings

clusterComm = MPI.COMM_WORLD
rank = clusterComm.Get_rank()  #current rank

GPURows = 1
numberOfGPUs = 10
numberOfGPUsList = [numberOfGPUs,numberOfGPUs]
share = 0.1
totalGPUs = numberOfGPUs+numberOfGPUs
gpuRank = [i for i in range(numberOfGPUs)]

if  share > 0 and totalGPUs > GPURows:
    warnings.warn("Not enough rows for the number of GPUS({}), add more GPU rows({}), increase share({}), or exclude GPUs. Attempting to adjust GPU ranks to continue run.".format(numberOfGPUs,GPURows,share))
    removeGPUs = totalGPUs-GPURows
    removeList = [1 for i in range(removeGPUs)]
    nodeRemoveGPUs = numpy.array_split(removeList,len(numberOfGPUsList))
    nodeRemoveGPUs = clusterComm.scatter(nodeRemoveGPUs[::-1]) #needs solver.
    #Adjusting ranks
    for i in range(len(nodeRemoveGPUs)):
        gpuRank.pop()
    #Getting updated GPU variables
    numberOfGPUs = len(gpuRank)
    numberOfGPUsList = clusterComm.allgather(numberOfGPUs)
    totalGPUs = numpy.sum(numberOfGPUsList) #getting total number of GPUs
    #Check that adjustment worked
    assert totalGPUs <= GPURows if share > 0 else True, "GPU rank adjustment failed." 