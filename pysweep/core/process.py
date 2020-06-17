import numpy, mpi4py.MPI as MPI
import pysweep.core.GPUtil as GPUtil

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
        gpuRank,solver.blocks = solver.nodeComm.scatter(list(zip(gpuRank,gbs)))
    solver.gpuBool = True if gpuRank is not None else False

def destroyUnecessaryProcesses(solver,nodeRanks,removeRanks,allRanks):
    """Use this to destory unwanted mpi processes."""
    [allRanks.remove(x) for x in set([x[0] for x in solver.comm.allgather(removeRanks) if x])]
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
    #Assert that the total number of blocks is an integer
    tnc = solver.cluster_comm.Get_size()
    ns = solver.nodeComm.Get_size()
    if solver.share>0:
        lim = ns if solver.share==1 else ns-1
        gpuRank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=solver.exid,limit=lim) #getting devices by load
        numberOfGPUs = len(gpuRank)
    else:
        gpuRank = []
        numberOfGPUs = 0
    j = numpy.arange(0,tnc+1,1,dtype=numpy.intc) if solver.share < 1 else numpy.zeros(tnc+1,dtype=numpy.intc)
    ranks,numberOfGPUsList = zip(*[(None,0)]+solver.cluster_comm.allgather((solver.rank,numberOfGPUs)))
    ranks = list(ranks)
    ranks[0] = ranks[-1] #Set first rank to last rank
    ranks.append(ranks[1]) #Add first rank to end
    i = [0]+[numberOfGPUsList[i]+sum(numberOfGPUsList[:i]) for i in range(1,len(numberOfGPUsList))]
    totalGPUs = numpy.sum(numberOfGPUsList)
    numberOfRows = solver.arrayshape[2]/solver.blocksize[0]
    assert (numpy.prod(solver.arrayshape[1:])/numpy.prod(solver.blocksize)).is_integer(), "Provided array dimensions is not divisible by the specified block size."
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


def separateNodeAndCluster(solver):
    """Use this function to create MPI variables and separate node and cluster variables
    args:
    solver - solver object
    """
    solver.comm = MPI.COMM_WORLD
    processor = MPI.Get_processor_name()
    solver.rank = solver.comm.Get_rank()  #current rank
    allRanks = solver.comm.allgather(solver.rank) #All ranks in simulation
    #Create individual node comm
    nodes_processors = solver.comm.allgather((solver.rank,processor))
    processors = tuple(zip(*nodes_processors))[1]
    nodeRanks = [n for n,p in nodes_processors if p==processor]
    nodeGroup = solver.comm.group.Incl(nodeRanks)
    solver.nodeComm = solver.comm.Create_group(nodeGroup)
    solver.nodeMaster = nodeRanks[0]
    solver.nodeMasterBool = solver.rank == solver.nodeMaster
    #Create cluster comm
    clusterRanks = list(set(solver.comm.allgather(solver.nodeMaster)))
    solver.cluster_master = clusterRanks[0]
    cluster_group = solver.comm.group.Incl(clusterRanks)
    solver.cluster_comm = solver.comm.Create_group(cluster_group)
    solver.clusterMasterBool = solver.rank == solver.cluster_master
    return allRanks,nodeRanks


def setupMPI(solver):
    #-------------MPI SETUP----------------------------#
    allRanks,nodeRanks = separateNodeAndCluster(solver) #This function creates necessary variables for MPI to use
     #Assumes all nodes have the same number of cores in CPU
    if solver.nodeMasterBool:
        #Giving each node an id
        if solver.clusterMasterBool:
            nodeID = numpy.arange(1,solver.cluster_comm.Get_size()+1,1,dtype=numpy.intc)
        else:
            nodeID = None
        nodeID = solver.cluster_comm.scatter(nodeID)
        #Getting GPU information
        gpuRank,totalGPUs, numberOfGPUs, nodeInfo, communicationRanks, GPURows,CPURows = getGPUInfo(solver,nodeID)
        removeRanks = findRanksToDestroy(nodeRanks,solver.share,numberOfGPUs,CPURows)
        [gpuRank.append(None) for i in range(len(nodeRanks)-len(gpuRank))]
        #Testing ranks and number of gpus to ensure simulation is viable
        assert totalGPUs < solver.comm.Get_size() if solver.share < 1 else True,"The affinity specifies use of heterogeneous system but number of GPUs exceeds number of specified ranks."
        assert totalGPUs > 0 if solver.share > 0 else True, "There are no avaliable GPUs"
        assert totalGPUs <= GPURows if solver.share > 0 else True, "Not enough rows for the number of GPUS, add more GPU rows, increase affinity, or exclude GPUs."
    else:
        totalGPUs,nodeInfo,gpuRank,nodeID,numberOfGPUs,comranks = None,None,None,None,None,None
        removeRanks = []
    #Broadcasting gpu information
    totalGPUs = solver.comm.bcast(totalGPUs)
    nodeRanks = solver.nodeComm.bcast(nodeRanks)
    solver.nodeInfo = solver.nodeComm.bcast(nodeInfo)
    #----------------------__Removing Unwanted MPI Processes------------------------#
    destroyUnecessaryProcesses(solver,nodeRanks,removeRanks,allRanks)
    splitNodeBlocks(solver,numberOfGPUs,gpuRank)
    # Checking to ensure that there are enough
    assert totalGPUs >= solver.nodeComm.Get_size() if solver.share == 1 else True,"Not enough GPUs for ranks"
