import numpy,GPUtil, os, time, warnings, mpi4py.MPI as MPI,traceback
import pysweep.core.io as io
import socket

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
        return list() #gpuRank
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
    ranksPerNode = os.getenv('RANKS_PER_NODE') #make an environment variable available
    ranksPerNode = solver.nodeComm.Get_size() if ranksPerNode is None else int(ranksPerNode) #number of ranks for each node
    if solver.share>0:  
        gpuRank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=solver.exid,limit=ranksPerNode) #getting devices by load
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
    #Decomposition
    # print(Rows,Devices,nodeID,deviceType,multiplier)
    k = (Rows-Rows%Devices)/Devices if Devices!=0 else 0.0
    n = Rows%Devices if Devices!=0 else 0.0
    m = Devices-n
    # print(k,n,m,(k+1)*n+k*m,Rows)
    assert (k+1)*n+k*m == Rows, "{}: Problem with decomposition.".format(deviceType)

    if deviceType == "GPU": #Different boundary functions for GPU and CPU
        getBound = lambda x: (k+1)*n+k*(x-n) if x > n else (k+1)*x #GPU
    else:
        getBound = lambda y: (k)*m+(k+1)*(y-m) if y > m else (k)*y #CPU

    return getBound(multiplier[nodeID-1]),getBound(multiplier[nodeID])


def MinorSplit(solver,nodeInfo,gpuRank,adjustment):
    """
        This function splits the total number of blocks up on the node level 
        args:
        solver: solver object containing simulation data
        nodeInfo: information about the nodes starting and stopping points
        gpuRank: list containing gpu device info
        adjustment: used to adjust standard solver for boundary condition purposes

    """
    ranksPerNode = solver.nodeComm.Get_size() #getting number of ranks per node
    gpuSize, cpuSize = nodeInfo
    variableSlice = slice(0,solver.arrayShape[0],1)
    start = adjustment #Convert to actual array size
    intersection = int(gpuSize*solver.blocksize[0]+adjustment) #gpu-cpu intersection on a node
    stop = int((gpuSize+cpuSize)*solver.blocksize[0]+adjustment) #Convert to actual array size
    #Creating shared shape based on swept or not
    solver.sharedShape = solver.arrayShape[0],stop+adjustment,solver.arrayShape[2]+2*adjustment 
    #total GPU slice of shared array
    gpuBlock = variableSlice,slice(start,intersection,1),slice(adjustment,solver.arrayShape[-1]+adjustment,1) 
    individualBlocks = list() #For CPU 
    #Getting blocks
    for j in range(adjustment,solver.arrayShape[-1]+adjustment,solver.blocksize[1]): #column for loop
        for i in range(intersection,stop,solver.blocksize[0]):
            currBlock = (variableSlice,slice(i,i+solver.blocksize[0],1),slice(j,j+solver.blocksize[0],1)) #Getting current block
            individualBlocks.append(currBlock) #appending to individual blocks
    #Splitting blocks
    dividedBlocks = numpy.array_split(individualBlocks,ranksPerNode) #-len(gpuRank) #Potentially add use some nodes for both CPU and GPU
    while len(gpuRank)<len(dividedBlocks):
        gpuRank.append(None)
    return gpuBlock,dividedBlocks,gpuRank

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

    #Getting number of rows
    numberOfRows = solver.arrayShape[1]/solver.blocksize[0] #total number of rows
    GPURows = numpy.ceil(numberOfRows*solver.share)  #round up for GPU rows
    CPURows = numberOfRows-GPURows #remaining rows to CPUs
    solver.share=GPURows/numberOfRows #update effective share based on rounding

    #Determining number of GPU rows to number of CPU rows
    numberOfGPUsList = solver.clusterComm.allgather(numberOfGPUs)
    totalGPUs = numpy.sum(numberOfGPUsList) #getting total number of GPUs
    print(numberOfGPUs,numberOfGPUsList,totalGPUs,gpuRank,socket.gethostname())
    # if totalGPUs <= GPURows:
        
    assert totalGPUs <= GPURows if solver.share > 0 else True, "Not enough rows for the number of GPUS({}), add more GPU rows({}), increase share({}), or exclude GPUs.".format(numberOfGPUs,GPURows,solver.share)

    #multipliers for for boundaries
    gpuMult = [0]+[numberOfGPUsList[i]+sum(numberOfGPUsList[:i]) for i in range(len(numberOfGPUsList))] #GPU multipliers
    cpuMult = numpy.arange(0,numOfNodes+1,1,dtype=numpy.intc) if solver.share < 1 else numpy.zeros(numOfNodes+1,dtype=numpy.intc) #CPU multipliers
    #Get gpu boundaries
    gpuLowerBound,gpuUpperBound = getBlockBoundaries(GPURows,totalGPUs,nodeID,"GPU",gpuMult)
    #Get cpu boundaries
    cpuLowerBound,cpuUpperBound = getBlockBoundaries(CPURows,numOfNodes,nodeID,"CPU",cpuMult)
    #Compiling info into ranges and magnitudes
    start = int((gpuLowerBound+cpuLowerBound)*solver.blocksize[0])
    stop = int((gpuUpperBound+cpuUpperBound)*solver.blocksize[0])
    solver.globalBlock = slice(0,solver.arrayShape[0],1),slice(start,stop,1),slice(0,solver.arrayShape[-1],1) #part of initial conditions for this node
    nodeInfo = gpuUpperBound-gpuLowerBound,cpuUpperBound-cpuLowerBound #low range, high range, gpu magnitude, cpu magnitude
    #Testing ranks and number of gpus to ensure simulation is viable
    assert totalGPUs > 0 if solver.share > 0 else True, "There are no avaliable GPUs"
    try:
        assert totalGPUs >= solver.nodeComm.Get_size() if solver.share == 1 else True,"Not enough GPUs for ranks"
    except Exception as e:
        if solver.nodeMasterBool:
            tb = traceback.format_exc()
            addOn = "There are more processes than GPUs with a share of 1, {} processes will not contribute to the computation.".format(solver.nodeComm.Get_size()-totalGPUs)
            warnings.warn(tb+addOn)
    #Splitting up blocks at node level and returning results
    adjustment = 0 if solver.simulation else solver.operating
    return MinorSplit(solver,nodeInfo,gpuRank,adjustment) 

def getNeighborRanks(solver,nodeID):
    #Get communicating ranks
    ranks = [None,]+solver.clusterComm.allgather(solver.rank)
    ranks[0] = ranks[-1] #Set first rank to last rank
    ranks.append(ranks[1]) #Add first rank to end
    return (ranks[nodeID-1],ranks[nodeID+1]) #communicating neighbors

def setupCommunicators(solver):
    """Use this function to create MPI communicators for all ranks, node ranks, and cluster ranks
    args:
    solver - solver object
    """
    solver.comm = MPI.COMM_WORLD
    solver.rank = solver.comm.Get_rank()  #current rank
    nodeProcessor = MPI.Get_processor_name()
    # nodeProcessor = pseudoCluster(solver.rank) #TEMPORARY
    processors = list(set(solver.comm.allgather(nodeProcessor))) #temporarily replace processor with hostname to treat vc as different nodes
    processors.sort()
    colors = {proc:i for i,proc in enumerate(processors)}
    solver.nodeComm =solver.comm.Split(colors[nodeProcessor])
    nodeRanks = solver.nodeComm.allgather(solver.rank) 
    nodeMaster = nodeRanks[0]
    solver.nodeMasterBool = solver.rank == nodeMaster
    #Create cluster comm
    clusterRanks = list(set(solver.comm.allgather(nodeMaster)))
    clusterRanks.sort() #sets are unsorted, so list needs to be sorted
    clusterMaster = clusterRanks[0]
    cluster_group = solver.comm.group.Incl(clusterRanks)
    solver.clusterComm = solver.comm.Create_group(cluster_group)
    solver.clusterMasterBool = solver.rank == clusterMaster


def setupProcesses(solver):
    #Assumes all nodes have the same number of cores in CPU
    if solver.nodeMasterBool:
        #Giving each node an id
        if solver.clusterMasterBool:
            nodeID = numpy.arange(1,solver.clusterComm.Get_size()+1,1,dtype=numpy.intc)
        else:
            nodeID = None
        nodeID = solver.clusterComm.scatter(nodeID)
        #Splitting data across cluster (Major)
        gpuBlock,blocks,gpuRank =  MajorSplit(solver,nodeID)
        #Getting neighboring ranks for communication
        solver.neighbors = getNeighborRanks(solver,nodeID) 
    else:
       gpuRank,gpuBlock,solver.globalBlock,blocks,solver.sharedShape,solver.neighbors = None,None,None,None,None,None
    #Broadcasting gpu information
    solver.blocks = solver.nodeComm.scatter(blocks)
    solver.gpuRank = solver.nodeComm.scatter(gpuRank)
    solver.gpuBlock = solver.nodeComm.bcast(gpuBlock) #total gpu block in shared array
    solver.share = solver.nodeComm.bcast(solver.share) #update effective share
    solver.globalBlock = solver.nodeComm.bcast(solver.globalBlock) #total cpu block in shared array
    solver.sharedShape = solver.nodeComm.bcast(solver.sharedShape) #shape of node shared array
    solver.gpuBool = True if solver.gpuRank is not None else False
    
    #----------------------Warning Empty MPI Processes------------------------#
    if len(solver.blocks)==0 and not solver.gpuBool:
        warnings.warn('rank {} was not given any CPU blocks to solve and does not have any GPUs available.'.format(solver.rank))
