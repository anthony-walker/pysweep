def standardCore():
    # ------------------- Operations specifically for GPus and CPUs------------------------#
    if GRB:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank)
        cuda_context = cuda_device.make_context()
        DecompObj,garr = dcore.gpu_core(blocks,BS,OPS,gargs,GRB,ITS)
        mpi_pool= None
    else:
        garr = None
        DecompObj,blocks= dcore.cpu_core(sarr,blocks,shared_shape,OPS,BS,GRB,ITS,gargs)
        pool_size = min(len(blocks), os.cpu_count()-node_comm.Get_size()+1)
        mpi_pool = mp.Pool(pool_size)
    functions.send_edges(sarr,NMB,GRB,node_comm,cluster_comm,comranks,OPS,garr,DecompObj)
    comm.Barrier() #Ensure all processes are prepared to solve

def standard_cpu_core(sarr,totalCPUBlock,shared_shape,OPS,BS,GRB,ITS,gargs):
    """Use this function to execute core cpu only processes"""
    xslice = slice(shared_shape[2]-OPS-(totalCPUBlock[2].stop-totalCPUBlock[2].start),shared_shape[2]-OPS,1)
    swb = (totalCPUBlock[0],totalCPUBlock[1],xslice,totalCPUBlock[3])
    blocks,totalCPUBlock = decomp.create_cpu_blocks(totalCPUBlock,BS,shared_shape,OPS)
    cpu.set_globals(GRB,*gargs,source_mod=None)
    #Creating sets for cpu calculation
    DecompObj = functions.DecompCPU(0,[(x+OPS,y+OPS) for x,y in numpy.ndindex((BS[0],BS[1]))],ITS-1,cwrite=swb,cread=totalCPUBlock)
    sgs.CPUArray = decomp.createSharedPoolArray(sarr[totalCPUBlock].shape)
    return DecompObj,blocks

def standard_gpu_core(blocks,BS,OPS,gargs,GRB,ITS):
    """Use this function to execute core gpu only processes"""
    block_shape = [i.stop-i.start for i in blocks[:2]]+[i.stop-i.start+2*OPS for i in blocks[2:]]
    gwrite = blocks
    gread = blocks[:2]
    gread+=tuple([slice(i.start-OPS,i.stop+OPS,1) for i in blocks[2:]])
    # Creating local GPU array with split
    GRD = (int((block_shape[2]-2*OPS)/BS[0]),int((block_shape[3]-2*OPS)/BS[1]))   #Grid size
    #Creating constants
    NV = block_shape[1]
    SX = block_shape[2]
    SY = block_shape[3]
    VARS =  block_shape[2]*(block_shape[3])
    TIMES = VARS*NV
    const_dict = ({"NV":NV,"VARS":VARS,"TIMES":TIMES,"OPS":OPS,"ITS":ITS,"SX":SX,"SY":SY})
    garr = decomp.createLocalGPUArray(block_shape)
    #Building CUDA source code
    SM = source.buildGPUSource(os.path.dirname(__file__))
    source.decomp_constant_copy(SM,const_dict)
    cpu.set_globals(GRB,*gargs,source_mod=SM)
    DecompObj  = functions.DecompGPU(0,SM.get_function("Decomp"),GRD,BS,gread,gwrite,garr)
    return DecompObj,garr

def create_standard_cpu_blocks(totalCPUBlock,BS,shared_shape,ops):
    """Use this function to create blocks."""
    row_range = numpy.arange(ops,totalCPUBlock[2].stop-totalCPUBlock[2].start+ops,BS[1],dtype=numpy.intc)
    column_range = numpy.arange(ops,totalCPUBlock[3].stop-totalCPUBlock[3].start+ops,BS[1],dtype=numpy.intc)
    xslice = slice(shared_shape[2]-2*ops-(totalCPUBlock[2].stop-totalCPUBlock[2].start),shared_shape[2],1)
    yslice = slice(totalCPUBlock[3].start-ops,totalCPUBlock[3].stop+ops,1)
    #There is an issue with total cpu block for decomp on cluster
    ntcb = (totalCPUBlock[0],totalCPUBlock[1],xslice,yslice)
    return [(totalCPUBlock[0],totalCPUBlock[1],slice(x-ops,x+BS[0]+ops,1),slice(y-ops,y+BS[1]+ops,1)) for x,y in product(row_range,column_range)],ntcb


def standardBlock(solver):
    """This is the entry point for the standard portion of the block module."""
    #---------------------------Creating and Filling Shared Array-------------#
    shared_shape = (ITS+1,arr0.shape[0],int(sum(solver.nodeInfo[2:])*BS[0]+2*OPS),arr0.shape[2]+2*OPS)
    sarr = decomp.create_CPU_sarray(node_comm,shared_shape,dType,numpy.zeros(shared_shape).nbytes)
    #Making blocks match array other dimensions
    bsls = [slice(0,i,1) for i in shared_shape]
    blocks = (bsls[0],bsls[1],blocks,slice(bsls[3].start+OPS,bsls[3].stop-OPS,1))
    GRB = True if gpu_rank is not None else False
    #Filling shared array
    if NMB:
        gsc = (slice(0,arr0.shape[0],1),slice(int(solver.nodeInfo[0]*BS[0]),int(solver.nodeInfo[1]*BS[0]),1),slice(0,arr0.shape[2],1))
        i1,i2,i3 = gsc
        #ITS STEP and BOUNDARIES
        sarr[ITS-1,:,OPS:-OPS,OPS:-OPS] =  arr0[gsc]
        sarr[ITS-1,:,OPS:-OPS,:OPS] = arr0[gsc[0],gsc[1],-OPS-1:-1]
        sarr[ITS-1,:,OPS:-OPS,-OPS:] = arr0[gsc[0],gsc[1],1:OPS+1]
        #INITIAL STEP AND BOUNDARIES
        sarr[0,:,OPS:-OPS,OPS:-OPS] =  arr0[gsc]
        sarr[0,:,OPS:-OPS,:OPS] = arr0[gsc[0],gsc[1],-OPS-1:-1]
        sarr[0,:,OPS:-OPS,-OPS:] = arr0[gsc[0],gsc[1],1:OPS+1]
    else:
        gsc = None


# FUNCTIONS
def Decomposition(GRB,OPS,sarr,garr,blocks,mpi_pool,DecompObj):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Splitting between cpu and gpu
    if solver.gpuBool:
        DecompObj(garr)
        sarr[DecompObj.gwrite]=garr[:,:,OPS:-OPS,OPS:-OPS]
    else:   #CPUs do this
        mpi_pool.map(DecompObj,blocks)
        #Copy result to MPI shared process array
        sarr[DecompObj.cwrite] = sgs.CPUArray[:,:,OPS:-OPS,OPS:-OPS]
    DecompObj.globalTimeStep+=1 #Update global time step

def send_edges(sarr,NMB,GRB,nodeComm,cluster_comm,comranks,ops,garr,DecompObj):
    """Use this function to communicate data between nodes"""
    if NMB:
        sarr[:,:,ops:-ops,:ops] = sarr[:,:,ops:-ops,-2*ops-1:-ops-1]
        sarr[:,:,ops:-ops,-ops:] = sarr[:,:,ops:-ops,1:ops+1]
        bufffor = numpy.copy(sarr[:,:,-2*ops-1:-ops-1,:])
        buffback = numpy.copy(sarr[:,:,ops+1:2*ops+1,:])
        bufferf = cluster_comm.sendrecv(sendobj=bufffor,dest=comranks[1],source=comranks[0])
        bufferb = cluster_comm.sendrecv(sendobj=buffback,dest=comranks[0],source=comranks[1])
        cluster_comm.Barrier()
        sarr[:,:,:ops,:] = bufferf[:,:,:,:]
        sarr[:,:,-ops:,:] = bufferb[:,:,:,:]
    solver.nodeComm.Barrier()

    if solver.gpuBool:
        garr[:,:,:,:] = sarr[DecompObj.gread]
    else:
        sgs.CPUArray[:,:,:,:] = sarr[DecompObj.cread]

