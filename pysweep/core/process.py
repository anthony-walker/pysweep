import numpy,time, mpi4py.MPI as MPI
import pysweep.core.GPUtil as GPUtil
import pysweep.core.io as io
import socket #TEMPORARY

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

def splitNodeBlocks(solver,numberOfGPUs,gpuRank):
    """Use this function to split data amongst nodes."""
    if solver.nodeMasterBool:
        start,stop,gm,cm = solver.nodeInfo
        gstop = start+gm
        g = numpy.linspace(start,gstop,numberOfGPUs+1,dtype=numpy.intc)
        if solver.simulation: #Swept rule
            gbs = [slice(int(g[i]*solver.blocksize[0]),int(g[i+1]*solver.blocksize[0]),1) for i in range(numberOfGPUs)]+[slice(int(gstop*solver.blocksize[0]),int(stop*solver.blocksize[0]),1)]
        else: #Standard
            ops = solver.operating
            gbs = [slice(int(g[i]*solver.blocksize[0]+ops),int(g[i+1]*solver.blocksize[0]+ops),1) for i in range(numberOfGPUs)]+[slice(int(gstop*solver.blocksize[0]+ops),int(stop*solver.blocksize[0]+ops),1)]
        # gbs = numpy.array_split(gbs,solver.nodeComm.Get_size()-numberOfGPUs)
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

def getGPUInfo(solver,nodeID):
    """Use this function to split data amongst nodes."""

    tnc = solver.clusterComm.Get_size()
    ns = solver.nodeComm.Get_size()
    if solver.share>0:
        lim = ns if solver.share==1 else ns-1
        gpuRank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=solver.exid,limit=lim) #getting devices by load
        numberOfGPUs = len(gpuRank)
    else:
        gpuRank = []
        numberOfGPUs = 0
    j = numpy.arange(0,tnc+1,1,dtype=numpy.intc) if solver.share < 1 else numpy.zeros(tnc+1,dtype=numpy.intc)
    ranks,numberOfGPUsList = zip(*[(None,0)]+solver.clusterComm.allgather((solver.rank,numberOfGPUs)))
    ranks = list(ranks)
    ranks[0] = ranks[-1] #Set first rank to last rank
    ranks.append(ranks[1]) #Add first rank to end
    i = [0]+[numberOfGPUsList[i]+sum(numberOfGPUsList[:i]) for i in range(1,len(numberOfGPUsList))]
    totalGPUs = numpy.sum(numberOfGPUsList)
    numberOfRows = solver.arrayShape[1]/solver.blocksize[0]
    #Assert that the total number of blocks is an integer
    assert (numpy.prod(solver.arrayShape[1:])/numpy.prod(solver.blocksize)).is_integer(), "Provided array dimensions is not divisible by the specified block size."
    GPURows = numpy.ceil(numberOfRows*solver.share)
    CPURows = numberOfRows-GPURows
    k = (GPURows-GPURows%totalGPUs)/totalGPUs if totalGPUs!=0 else 0.0
    n = GPURows%totalGPUs if totalGPUs!=0 else 0.0
    m = totalGPUs-n
    assert (k+1)*n+k*m == GPURows, "GPU: Problem with decomposition."
    kc = (CPURows-CPURows%tnc)/tnc
    nc = CPURows%tnc
    mc = tnc-nc
    assert (kc+1)*nc+kc*mc == CPURows, "CPU: Problem with decomposition."
    getBoundGPU = lambda x: (k+1)*n+k*(x-n) if x > n else (k+1)*x
    getBoundCPU = lambda y: (kc)*mc+(kc+1)*(y-mc) if y > mc else (kc)*y
    gpuLowerBound = getBoundGPU(i[nodeID-1])
    gpuUpperBound = getBoundGPU(i[nodeID])
    cpuLowerBound = getBoundCPU(j[nodeID-1])
    cpuUpperBound = getBoundCPU(j[nodeID])
    nodeInfo = gpuLowerBound+cpuLowerBound,gpuUpperBound+cpuUpperBound,gpuUpperBound-gpuLowerBound,cpuUpperBound-cpuLowerBound #low range, high range, gpu magnitude, cpu magnitude
    commRanks = (ranks[nodeID-1],ranks[nodeID+1]) #communicating neighbors
    return gpuRank,totalGPUs,numberOfGPUs,nodeInfo,commRanks,GPURows,CPURows


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

def separateNodeAndCluster(solver):
    """Use this function to create MPI variables and separate node and cluster variables
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
    myNode = solver.nodeComm.allgather(solver.rank)
    print(myNode,solver.rank)
    allRanks = solver.comm.allgather(solver.rank) #All ranks in simulation
   
    # print(processor,solver.rank)
    # #Create individual node comm
    # nodes_processors = solver.comm.allgather((solver.rank,processor))
    # processors = tuple(zip(*nodes_processors))[1]
    # nodeRanks = [n for n,p in nodes_processors if p==processor]
    # #Group node ranks and make node comm
    # nodeGroup = solver.comm.group.Incl(nodeRanks)
    # solver.nodeComm = solver.comm.Create_group(nodeGroup)
    # solver.nodeMaster = nodeRanks[0]
    # solver.nodeMasterBool = solver.rank == solver.nodeMaster
    # #Create cluster comm
    # clusterRanks = list(set(solver.comm.allgather(solver.nodeMaster)))
    # solver.clusterMaster = clusterRanks[0]
    # cluster_group = solver.comm.group.Incl(clusterRanks)
    # solver.clusterComm = solver.comm.Create_group(cluster_group)
    # solver.clusterMasterBool = solver.rank == solver.clusterMaster
    # return allRanks,nodeRanks

def setupMPI(solver):
    #-------------MPI SETUP----------------------------#
    separateNodeAndCluster(solver) #This function creates necessary variables for MPI to use
    # io.verbosePrint(solver,"Setting up MPI...\n")
    #Assumes all nodes have the same number of cores in CPU
    # if solver.nodeMasterBool:
    #     #Giving each node an id
    #     if solver.clusterMasterBool:
    #         nodeID = numpy.arange(1,solver.clusterComm.Get_size()+1,1,dtype=numpy.intc)
    #     else:
    #         nodeID = None
    #     nodeID = solver.clusterComm.scatter(nodeID)
    #     #Getting GPU information
    #     gpuRank,totalGPUs, numberOfGPUs, nodeInfo, communicationRanks, GPURows,CPURows = getGPUInfo(solver,nodeID)
    #     [gpuRank.append(None) for i in range(len(nodeRanks)-len(gpuRank))]
    #     #Testing ranks and number of gpus to ensure simulation is viable
    #     assert totalGPUs < solver.comm.Get_size() if solver.share < 1 else True,"The affinity specifies use of heterogeneous system but number of GPUs exceeds number of specified ranks."
    #     assert totalGPUs > 0 if solver.share > 0 else True, "There are no avaliable GPUs"
    #     assert totalGPUs <= GPURows if solver.share > 0 else True, "Not enough rows for the number of GPUS, add more GPU rows, increase affinity, or exclude GPUs."
    # else:
    #     totalGPUs,nodeInfo,gpuRank,nodeID,numberOfGPUs,comranks = None,None,None,None,None,None
    #     removeRanks = []
    # #Broadcasting gpu information
    # totalGPUs = solver.comm.bcast(totalGPUs)
    # nodeRanks = solver.nodeComm.bcast(nodeRanks)
    # solver.nodeInfo = solver.nodeComm.bcast(nodeInfo)
    # #----------------------__Removing Unwanted MPI Processes------------------------#
    # splitNodeBlocks(solver,numberOfGPUs,gpuRank)
    # # Checking to ensure that there are enough
    # assert totalGPUs >= solver.nodeComm.Get_size() if solver.share == 1 else True,"Not enough GPUs for ranks"
