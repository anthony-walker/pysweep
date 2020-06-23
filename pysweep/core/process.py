import numpy,time, mpi4py.MPI as MPI
import pysweep.core.GPUtil as GPUtil
import pysweep.core.io as io
import socket #TEMPORARY

def pseudoCluster(rank):
    """This is a temporary function to rename processors based on rank."""
    if rank < 2:
        return "node1"
    elif rank < 4:
        return "node2"
    elif rank < 6:
        return "node3"
    elif rank < 8:
        return "node4"


def pseudoGPU(gpuRank,rank):
    """This is a temporary function to assign gpu's based on rank."""
    if rank < 2:
        return gpuRank
    elif rank < 4:
        return gpuRank
    elif rank < 6:
         return list()
    elif rank < 8:
         return list()

def cleanupProcesses(solver,start,stop):
    """Use this function to wrap up the code and produce log file."""
    # Clean Up - Pop Cuda Contexts and Close Pool
    if solver.gpuBool:
        solver.cuda_context.pop()
    solver.comm.Barrier()
    clocktime = stop-start
    solver.clocktime[0] = clocktime
    solver.hdf5.close()
    #Removing input file.
    if solver.clusterMasterBool:
        io.updateLogFile(solver,clocktime)

def MinorSplit(solver,numberOfGPUs,gpuRank):
    """Use this function to split data amongst nodes."""
    if solver.nodeMasterBool:
        start,stop,gpuMagnitude,cpuMagnitude = solver.nodeInfo
        gstop = start+gpuMagnitude
        g = numpy.linspace(start,gstop,numberOfGPUs+1,dtype=numpy.intc)
        if solver.simulation: #Swept rule if true else standard
            gbs = [slice(int(g[i]*solver.blocksize[0]),int(g[i+1]*solver.blocksize[0]),1) for i in range(numberOfGPUs)]+[slice(int(gstop*solver.blocksize[0]),int(stop*solver.blocksize[0]),1)]
        else: #Standard
            gbs = [slice(int(g[i]*solver.blocksize[0]+solver.operating),int(g[i+1]*solver.blocksize[0]+solver.operating),1) for i in range(numberOfGPUs)]+[slice(int(gstop*solver.blocksize[0]+solver.operating),int(stop*solver.blocksize[0]+solver.operating),1)]
        gbs = numpy.array_split(gbs,solver.nodeComm.Get_size()-numberOfGPUs)

        print(solver.rank,gbs)
    #     gpuRank = list(zip(gpuRank,gbs))
    # solver.gpuRank,solver.blocks = solver.nodeComm.scatter(gpuRank)
    # print(solver.blocks)
    # solver.gpuBool = True if gpuRank is not None else False

def destroyUnecessaryProcesses(solver,nodeRanks,removeRanks,allRanks):
    """Use this to destory unwanted mpi processes."""
    removeList = []
    for x in solver.comm.allgather(removeRanks):
        if x:
            removeList+=x
    [allRanks.remove(x) for x in set(removeList)]
    solver.comm.Barrier()
    #Remaking comms
    if solver.rank in allRanks:
        #New Global comm
        ncg = solver.comm.group.Incl(allRanks)
        solver.comm = solver.comm.Create_group(ncg)
        # New node comm
        nodeGroup = solver.comm.group.Incl(nodeRanks)
        solver.nodeComm = solver.comm.Create_group(nodeGroup)
    else: #Ending unnecs
        MPI.Finalize()
        exit(0)
    solver.comm.Barrier()

def findRanksToDestroy(nodeRanks,share,numberOfGPUs,CPURows):
    """Use this function to find ranks that need removed."""
    removeRanks = list()
    # nodeRanks = [nodeRanks[0]] if solver.share == 0 else nodeRanks
    try:
        assert CPURows > 0 if share < 1 else True, "AssertionError: CPURows!>0, Affinity specified forces number of rows for the cpu to be zero, CPU process will be destroyed. Adjusting affintiy to 1."
        while len(nodeRanks) > numberOfGPUs+(1-int(share)):
            removeRanks.append(nodeRanks.pop())
    except Exception as e:
        print(str(e))
        while len(nodeRanks) > numberOfGPUs:
            removeRanks.append(nodeRanks.pop())
    return removeRanks

def MajorSplit(solver,nodeID):
    """
    This function splits the total number of number of rows (determined from blocksize) amongst a global group for the GPU and CPU.

    The blocks are then further subdivided on the node level. It then determines block boundaries for each node based on this premise. 

    The block boundaries are used to determine the array sections in which a node will handle.

    """
    numOfNodes = solver.clusterComm.Get_size() #number of nodes in cluster
    ranksPerNode = solver.nodeComm.Get_size() #number of ranks for each node

    #Getting gpus if share is greater than 0
    if solver.share>0:  
        lim = ranksPerNode if solver.share==1 else ranksPerNode-1
        gpuRank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=solver.exid,limit=lim) #getting devices by load
        
        gpuRank = pseudoGPU(gpuRank,solver.rank) #TEMPORARY REMOVE ME

        numberOfGPUs = len(gpuRank)
    
    else:
        gpuRank = []
        numberOfGPUs = 0


    #Assert that the total number of blocks is an integer
    assert (numpy.prod(solver.arrayShape[1:])/numpy.prod(solver.blocksize)).is_integer(), "Provided array dimensions is not divisible by the specified block size."
   
    #Determining number of GPU rows to number of CPU rows
    numberOfGPUsList = solver.clusterComm.allgather(numberOfGPUs)
    totalGPUs = numpy.sum(numberOfGPUsList) #getting total number of GPUs
    numberOfRows = solver.arrayShape[1]/solver.blocksize[0] #total number of rows
    i = [0]+[numberOfGPUsList[i]+sum(numberOfGPUsList[:i]) for i in range(len(numberOfGPUsList))] #GPU indexs
    j = numpy.arange(0,numOfNodes+1,1,dtype=numpy.intc) if solver.share < 1 else numpy.zeros(numOfNodes+1,dtype=numpy.intc) #CPU indexs
    GPURows = numpy.ceil(numberOfRows*solver.share) #round up for GPU rows
    CPURows = numberOfRows-GPURows #remaining rows to CPUs

    #GPU decomposition
    k = (GPURows-GPURows%totalGPUs)/totalGPUs if totalGPUs!=0 else 0.0
    n = GPURows%totalGPUs if totalGPUs!=0 else 0.0
    m = totalGPUs-n
    assert (k+1)*n+k*m == GPURows, "GPU: Problem with decomposition."
    getBoundGPU = lambda x: (k+1)*n+k*(x-n) if x > n else (k+1)*x

    #CPU decomposition
    kc = (CPURows-CPURows%numOfNodes)/numOfNodes
    nc = CPURows%numOfNodes
    mc = numOfNodes-nc
    assert (kc+1)*nc+kc*mc == CPURows, "CPU: Problem with decomposition."
    getBoundCPU = lambda y: (kc)*mc+(kc+1)*(y-mc) if y > mc else (kc)*y

    #Getting boundaries of blocks
    gpuLowerBound = getBoundGPU(i[nodeID-1])
    gpuUpperBound = getBoundGPU(i[nodeID])
    cpuLowerBound = getBoundCPU(j[nodeID-1])
    cpuUpperBound = getBoundCPU(j[nodeID])

    #Compiling info into ranges and magnitudes
    nodeInfo = gpuLowerBound+cpuLowerBound,gpuUpperBound+cpuUpperBound,gpuUpperBound-gpuLowerBound,cpuUpperBound-cpuLowerBound #low range, high range, gpu magnitude, cpu magnitude
    return gpuRank,totalGPUs,numberOfGPUs,nodeInfo,GPURows,CPURows


def getNeighborRanks(solver,nodeID):
    #Get communicating ranks
    ranks = [None,]+solver.clusterComm.allgather(solver.rank)
    ranks[0] = ranks[-1] #Set first rank to last rank
    ranks.append(ranks[1]) #Add first rank to end
    solver.neighbors = (ranks[nodeID-1],ranks[nodeID+1]) #communicating neighbors

def setupCommunicators(solver):
    """Use this function to create MPI communicators for all ranks, node ranks, and cluster ranks
    args:
    solver - solver object
    """
    solver.comm = MPI.COMM_WORLD
    solver.rank = solver.comm.Get_rank()  #current rank
    nodeProcessor = MPI.Get_processor_name()
    nodeProcessor = pseudoCluster(solver.rank) #TEMPORARY
    processors = list(set(solver.comm.allgather(nodeProcessor))) #temporarily replace processor with hostname to treat vc as different nodes
    processors.sort()
    colors = {proc:i for i,proc in enumerate(processors)}
    solver.nodeComm =solver.comm.Split(colors[nodeProcessor])
    nodeRanks = solver.nodeComm.allgather(solver.rank) #Temporary
    allRanks = solver.comm.allgather(solver.rank) #All ranks in simulation
    solver.nodeMaster = nodeRanks[0]
    solver.nodeMasterBool = solver.rank == solver.nodeMaster
    #Create cluster comm
    clusterRanks = list(set(solver.comm.allgather(solver.nodeMaster)))
    clusterRanks.sort() #sets are unsorted, so list needs to be sorted
    clusterMaster = clusterRanks[0]
    cluster_group = solver.comm.group.Incl(clusterRanks)
    solver.clusterComm = solver.comm.Create_group(cluster_group)
    solver.clusterMasterBool = solver.rank == clusterMaster
    return allRanks,nodeRanks

def setupMPI(solver):
    #-------------MPI SETUP----------------------------#
    allRanks,nodeRanks = setupCommunicators(solver) #This function creates necessary variables for MPI to use
    io.verbosePrint(solver,"Setting up MPI...\n")
    #Assumes all nodes have the same number of cores in CPU
    if solver.nodeMasterBool:
        #Giving each node an id
        if solver.clusterMasterBool:
            nodeID = numpy.arange(1,solver.clusterComm.Get_size()+1,1,dtype=numpy.intc)
        else:
            nodeID = None
        nodeID = solver.clusterComm.scatter(nodeID)
        #Getting GPU information
        gpuRank,totalGPUs, numberOfGPUs, nodeInfo, GPURows, CPURows = MajorSplit(solver,nodeID)

        getNeighborRanks(solver,nodeID) #Getting neighboring ranks for communication
        # removeRanks = findRanksToDestroy(nodeRanks,solver.share,numberOfGPUs,CPURows)
        [gpuRank.append(None) for i in range(len(nodeRanks)-len(gpuRank))]
        #Testing ranks and number of gpus to ensure simulation is viable
        assert totalGPUs < solver.comm.Get_size() if solver.share < 1 else True,"The affinity specifies use of heterogeneous system but number of GPUs exceeds number of specified ranks."
        assert totalGPUs > 0 if solver.share > 0 else True, "There are no avaliable GPUs"
        assert totalGPUs <= GPURows if solver.share > 0 else True, "Not enough rows for the number of GPUS, add more GPU rows, increase affinity, or exclude GPUs."
    else:
        totalGPUs,nodeInfo,gpuRank,nodeID,numberOfGPUs = None,None,None,None,None
        removeRanks = []
    #Broadcasting gpu information
    totalGPUs = solver.comm.bcast(totalGPUs)
    nodeRanks = solver.nodeComm.bcast(nodeRanks)
    solver.nodeInfo = solver.nodeComm.bcast(nodeInfo)
    # #----------------------__Removing Unwanted MPI Processes------------------------#
    # destroyUnecessaryProcesses(solver,nodeRanks,removeRanks,allRanks)
    MinorSplit(solver,numberOfGPUs,gpuRank)
    # Checking to ensure that there are enough
    assert totalGPUs >= solver.nodeComm.Get_size() if solver.share == 1 else True,"Not enough GPUs for ranks"
