import numpy,time, warnings, mpi4py.MPI as MPI
import pysweep.core.GPUtil as GPUtil
import pysweep.core.io as io

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

def getGPUInfo(solver):
    """Use this function to return info about the GPUs on the system."""
    #Getting gpus if share is greater than 0
    ranksPerNode = solver.nodeComm.Get_size() #number of ranks for each node
    if solver.share>0:  
        lim = ranksPerNode if solver.share==1 else ranksPerNode-1
        gpuRank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=solver.exid,limit=lim) #getting devices by load
        
        gpuRank = pseudoGPU(gpuRank,solver.rank) #TEMPORARY REMOVE ME

        numberOfGPUs = len(gpuRank)
    
    else:
        gpuRank = []
        numberOfGPUs = 0
    return gpuRank,numberOfGPUs

def getBlockBoundaries(Rows,Devices,nodeID,deviceType,multiplier):
    """
    Use this function to get block boundaries on each node.
    args:
    Rows: number of global rows for the device type
    Devices: number of devices (e.g. nodes or GPUs)
    nodeID: node identifier
    deviceType: a string with "CPU" or "GPU" for error assertion
    multiplier: a variable used for determining boundary
    """
    #GPU decomposition
    k = (Rows-Rows%Devices)/Devices if Devices!=0 else 0.0
    n = Rows%Devices if Devices!=0 else 0.0
    m = Devices-n
    assert (k+1)*n+k*m == Rows, "{}: Problem with decomposition.".format(deviceType)

    if deviceType == "GPU": #Different boundary functions for GPU and CPU
        getBound = lambda x: (k+1)*n+k*(x-n) if x > n else (k+1)*x #GPU
    else:
        getBound = lambda y: (k)*m+(k+1)*(y-m) if y > m else (k)*y #CPU

    return getBound(multiplier[nodeID-1]),getBound(multiplier[nodeID])


def MinorSplit(solver,nodeInfo,gpuRank):
    """
        This function splits the total number of number of rows (determined from blocksize) amongst a global group for the GPU and CPU.

        The blocks are then further subdivided on the node level. It then determines block boundaries for each node based on this premise. 

        The block boundaries are used to determine the array sections in which a node will handle.
    """

    
    ranksPerNode = solver.nodeComm.Get_size() #getting number of ranks per node
    start, stop, gpuSize, cpuSize = nodeInfo
    start = int(start*solver.blocksize[0]) #Convert to actual array size
    intersection = int((stop-cpuSize)*solver.blocksize[0]) #gpu-cpu intersection on a node
    stop = int(stop*solver.blocksize[0]) #Convert to actual array size
    gpuBlock = slice(start,intersection,1),slice(0,solver.arrayShape[-1],1) #total GPU slice of shared array
    cpuBlock = slice(intersection,stop,1),slice(0,solver.arrayShape[-1],1) #total CPU slice of shared array
    individualBlocks = list() #For CPU 
    #MAY NEED TO ADD IF STATEMENT FOR DECOMP
    for j in range(0,solver.arrayShape[-1],solver.blocksize[1]): #column for loop
        for i in range(intersection,stop,solver.blocksize[0]):
            currBlock = slice(i,i+solver.blocksize[0],1),slice(j,j+solver.blocksize[0],1) #Getting current block
            individualBlocks.append(currBlock) #appending to individual blocks
    dividedBlocks = numpy.array_split(individualBlocks,ranksPerNode) #-len(gpuRank) #Potentially add use some nodes for both CPU and GPU

    while len(gpuRank)<len(dividedBlocks):
        gpuRank.append(None)

    return gpuBlock,cpuBlock,dividedBlocks,gpuRank

def MajorSplit(solver,nodeID):
    """
    This function splits the total number of number of rows (determined from blocksize) amongst a global group for the GPU and CPU.

    The blocks are then further subdivided on the node level. It then determines block boundaries for each node based on this premise. 

    The block boundaries are used to determine the array sections in which a node will handle.

    """
    numOfNodes = solver.clusterComm.Get_size() #number of nodes in cluster
    gpuRank,numberOfGPUs = getGPUInfo(solver) #getting ranks with gpus and number
    #Assert that the total number of blocks is an integer
    assert (numpy.prod(solver.arrayShape[1:])/numpy.prod(solver.blocksize)).is_integer(), "Provided array dimensions is not divisible by the specified block size."
    #Determining number of GPU rows to number of CPU rows
    numberOfGPUsList = solver.clusterComm.allgather(numberOfGPUs)
    totalGPUs = numpy.sum(numberOfGPUsList) #getting total number of GPUs
    numberOfRows = solver.arrayShape[1]/solver.blocksize[0] #total number of rows
    gpuMult = [0]+[numberOfGPUsList[i]+sum(numberOfGPUsList[:i]) for i in range(len(numberOfGPUsList))] #GPU multipliers
    cpuMult = numpy.arange(0,numOfNodes+1,1,dtype=numpy.intc) if solver.share < 1 else numpy.zeros(numOfNodes+1,dtype=numpy.intc) #CPU multipliers
    GPURows = numpy.ceil(numberOfRows*solver.share) #round up for GPU rows
    CPURows = numberOfRows-GPURows #remaining rows to CPUs
    #Get gpu boundaries
    gpuLowerBound,gpuUpperBound = getBlockBoundaries(GPURows,totalGPUs,nodeID,"GPU",gpuMult)
    #Get cpu boundaries
    cpuLowerBound,cpuUpperBound = getBlockBoundaries(CPURows,numOfNodes,nodeID,"CPU",cpuMult)
    #Compiling info into ranges and magnitudes
    nodeInfo = gpuLowerBound+cpuLowerBound,gpuUpperBound+cpuUpperBound,gpuUpperBound-gpuLowerBound,cpuUpperBound-cpuLowerBound #low range, high range, gpu magnitude, cpu magnitude
    
    #Testing ranks and number of gpus to ensure simulation is viable
    assert totalGPUs < solver.comm.Get_size() if solver.share < 1 else True,"The affinity specifies use of heterogeneous system but number of GPUs exceeds number of specified ranks."
    assert totalGPUs > 0 if solver.share > 0 else True, "There are no avaliable GPUs"
    assert totalGPUs <= GPURows if solver.share > 0 else True, "Not enough rows for the number of GPUS, add more GPU rows, increase affinity, or exclude GPUs."
    assert totalGPUs >= solver.nodeComm.Get_size() if solver.share == 1 else True,"Not enough GPUs for ranks"

    return MinorSplit(solver,nodeInfo,gpuRank) #Splitting up blocks at node level


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
    nodeRanks = solver.nodeComm.allgather(solver.rank) 
    solver.nodeMaster = nodeRanks[0]
    solver.nodeMasterBool = solver.rank == solver.nodeMaster
    #Create cluster comm
    clusterRanks = list(set(solver.comm.allgather(solver.nodeMaster)))
    clusterRanks.sort() #sets are unsorted, so list needs to be sorted
    clusterMaster = clusterRanks[0]
    cluster_group = solver.comm.group.Incl(clusterRanks)
    solver.clusterComm = solver.comm.Create_group(cluster_group)
    solver.clusterMasterBool = solver.rank == clusterMaster


def setupMPI(solver):
    #-------------MPI SETUP----------------------------#
    setupCommunicators(solver) #This function creates necessary variables for MPI to use
    # io.verbosePrint(solver,"Setting up MPI...\n")
    #Assumes all nodes have the same number of cores in CPU
    if solver.nodeMasterBool:
        #Giving each node an id
        if solver.clusterMasterBool:
            nodeID = numpy.arange(1,solver.clusterComm.Get_size()+1,1,dtype=numpy.intc)
        else:
            nodeID = None
        nodeID = solver.clusterComm.scatter(nodeID)
        #Splitting data across cluster (Major)
        gpuBlock,cpuBlock,blocks,gpuRank =  MajorSplit(solver,nodeID)
        #Getting neighboring ranks for communication
        getNeighborRanks(solver,nodeID) 
    else:
       gpuRank,gpuBlock,cpuBlock,nodeID,blocks = None,None,None,None,None
    #Broadcasting gpu information
    solver.blocks = solver.nodeComm.scatter(blocks)
    gpuRank = solver.nodeComm.scatter(gpuRank)
    solver.gpuBlock = solver.nodeComm.bcast(gpuBlock) #total gpu block in shared array
    solver.cpuBlock = solver.nodeComm.bcast(cpuBlock) #total cpu block in shared array
    solver.gpuBool = True if gpuRank is not None else False
    #----------------------Warning Empty MPI Processes------------------------#
    if len(solver.blocks)==0:
        warnings.warn('rank {} was not given any blocks to solve.'.format(solver.rank))
