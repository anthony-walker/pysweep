
import numpy,mpi4py.MPI as MPI
import pysweep.core.functions as functions

def createCPUSharedArray(solver,arrayBytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(solver.dtype.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arrayBytes, itemsize, comm=solver.nodeComm)
    sharedBuffer, itemsize = win.Shared_query(0)
    solver.sharedArray = numpy.ndarray(buffer=sharedBuffer, dtype=solver.dtype.type, shape=solver.arrayShape)

def createSweptSharedArray(solver):
    """Use this function to create and fill the shared array for the swept solver."""
    #---------------------------Creating and Filling Shared Array-------------#
    createCPUSharedArray(solver,numpy.zeros(solver.arrayShape,dtype=solver.dtype).nbytes)
    #Making blocks match array other dimensions
    bsls = [slice(0,i,1) for i in solver.arrayShape]
    solver.blocks = (bsls[0],bsls[1],solver.blocks,bsls[3])
    #Filling shared array
    if solver.nodeMasterBool:
        gsc = (slice(0,solver.arrayShape[2],1),slice(int(solver.nodeInfo[0]*solver.blocksize[0]),int(solver.nodeInfo[1]*solver.blocksize[0]),1),slice(0,solver.arrayShape[3],1))
        solver.sharedArray[solver.intermediate-1,:,:,:] =  solver.initialConditions[gsc]
        solver.sharedArray[0,:,:,:] =  solver.initialConditions[gsc]
    else:
        gsc = None

def sweptBlock(solver):
    """This is the entry point for the swept portion of the block module."""
    createSweptSharedArray(solver)

def standardBlock(solver):
    """This is the entry point for the standard portion of the block module."""
    #---------------------------Creating and Filling Shared Array-------------#
    shared_shape = (TSO+1,arr0.shape[0],int(sum(solver.nodeInfo[2:])*BS[0]+2*OPS),arr0.shape[2]+2*OPS)
    sarr = decomp.create_CPU_sarray(node_comm,shared_shape,dType,numpy.zeros(shared_shape).nbytes)
    #Making blocks match array other dimensions
    bsls = [slice(0,i,1) for i in shared_shape]
    blocks = (bsls[0],bsls[1],blocks,slice(bsls[3].start+OPS,bsls[3].stop-OPS,1))
    GRB = True if gpu_rank is not None else False
    #Filling shared array
    if NMB:
        gsc = (slice(0,arr0.shape[0],1),slice(int(solver.nodeInfo[0]*BS[0]),int(solver.nodeInfo[1]*BS[0]),1),slice(0,arr0.shape[2],1))
        i1,i2,i3 = gsc
        #TSO STEP and BOUNDARIES
        sarr[TSO-1,:,OPS:-OPS,OPS:-OPS] =  arr0[gsc]
        sarr[TSO-1,:,OPS:-OPS,:OPS] = arr0[gsc[0],gsc[1],-OPS-1:-1]
        sarr[TSO-1,:,OPS:-OPS,-OPS:] = arr0[gsc[0],gsc[1],1:OPS+1]
        #INITIAL STEP AND BOUNDARIES
        sarr[0,:,OPS:-OPS,OPS:-OPS] =  arr0[gsc]
        sarr[0,:,OPS:-OPS,:OPS] = arr0[gsc[0],gsc[1],-OPS-1:-1]
        sarr[0,:,OPS:-OPS,-OPS:] = arr0[gsc[0],gsc[1],1:OPS+1]
    else:
        gsc = None

def sweptCore():
    # ------------------- Operations specifically for GPus and CPUs------------------------#
    if GRB:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank)
        cuda_context = cuda_device.make_context()
        SM,garr,Up,Down,Oct,Xb,Yb  = dcore.gpu_core(blocks,BS,OPS,gargs,GRB,MPSS,MOSS,TSO)
        mpiPool,total_cpu_block = None,None
    else:
        SM,garr = None,None
        blocks,total_cpu_block,Up,Down,Oct,Xb,Yb = dcore.cpu_core(sarr,blocks,shared_shape,OPS,BS,gargs,MPSS,TSO)
        poolSize = min(len(blocks[0]), os.cpu_count()-node_comm.Get_size()+1)
        mpiPool = mp.Pool(poolSize)
    comm.Barrier() #Ensure all processes are prepared to solve

def standardCore():
    # ------------------- Operations specifically for GPus and CPUs------------------------#
    if GRB:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(gpu_rank)
        cuda_context = cuda_device.make_context()
        DecompObj,garr = dcore.gpu_core(blocks,BS,OPS,gargs,GRB,TSO)
        mpi_pool= None
    else:
        garr = None
        DecompObj,blocks= dcore.cpu_core(sarr,blocks,shared_shape,OPS,BS,GRB,TSO,gargs)
        pool_size = min(len(blocks), os.cpu_count()-node_comm.Get_size()+1)
        mpi_pool = mp.Pool(pool_size)
    functions.send_edges(sarr,NMB,GRB,node_comm,cluster_comm,comranks,OPS,garr,DecompObj)
    comm.Barrier() #Ensure all processes are prepared to solve


def swept_cpu_core(sarr,total_cpu_block,shared_shape,OPS,BS,gargs,MPSS,TSO):
    """Use this function to execute core cpu only processes"""
    blocks,total_cpu_block = decomp.create_cpu_blocks(total_cpu_block,BS,shared_shape)
    blocks = decomp.create_escpu_blocks(blocks,shared_shape,BS)
    cpu.set_globals(False,*gargs)
    #Creating sets for cpu calculation
    up_sets = block.create_dist_up_sets(BS,OPS)
    down_sets = block.create_dist_down_sets(BS,OPS)
    oct_sets = down_sets+up_sets
    y_sets,x_sets = block.create_dist_bridge_sets(BS,OPS,MPSS)
    #Shared array
    sgs.carr = decomp.create_shared_pool_array(sarr[total_cpu_block].shape)
    sgs.carr[:,:,:,:] = sarr[total_cpu_block]
    #Create function objects
    Up = functions.GeometryCPU(0,up_sets,TSO,MPSS,TSO-1)
    Down = functions.GeometryCPU(0,down_sets,TSO,MPSS,TSO-1)
    Xb = functions.GeometryCPU(0,x_sets,TSO,MPSS,TSO-1)
    Yb = functions.GeometryCPU(0,y_sets,TSO,MPSS,TSO-1)
    Oct = functions.GeometryCPU(0,oct_sets,TSO,MPSS,TSO-1)
    return blocks,total_cpu_block,Up,Down,Oct,Xb,Yb

def swept_gpu_core(blocks,BS,OPS,gargs,GRB,MPSS,MOSS,TSO):
    """Use this function to execute core gpu only processes"""
    block_shape = [i.stop-i.start for i in blocks]
    block_shape[-1] += int(2*BS[0]) #Adding 2 blocks in the column direction
    # Creating local GPU array with split
    GRD = (int((block_shape[2])/BS[0]),int((block_shape[3])/BS[1]))   #Grid size
    #Creating constants
    NV = block_shape[1]
    SX = block_shape[2]
    SY = block_shape[3]
    VARS =  int(SX*SY)
    TIMES = VARS*NV
    const_dict = ({"NV":NV,"VARS":VARS,"TIMES":TIMES,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,'SX':SX,'SY':SY})
    garr = decomp.create_local_gpu_array(block_shape)
    garr = cuda.mem_alloc(garr.nbytes)
    #Building CUDA source code
    SM = source.build_gpu_source(os.path.dirname(__file__))
    source.swept_constant_copy(SM,const_dict)
    cpu.set_globals(GRB,*gargs,source_mod=SM)
    # Make GPU geometry
    Up = functions.GeometryGPU(0,SM.get_function("UpPyramid"),BS,GRD,MPSS,block_shape)
    Down = functions.GeometryGPU(0,SM.get_function("DownPyramid"),BS,(GRD[0],GRD[1]-1),MPSS,block_shape)
    Yb = functions.GeometryGPU(0,SM.get_function("YBridge"),BS,(GRD[0],GRD[1]-1),MPSS,block_shape)
    Xb = functions.GeometryGPU(0,SM.get_function("XBridge"),BS,GRD,MPSS,block_shape)
    Oct = functions.GeometryGPU(0,SM.get_function("Octahedron"),BS,(GRD[0],GRD[1]-1),MPSS,block_shape)
    return SM,garr,Up,Down,Oct,Xb,Yb

def standard_cpu_core(sarr,total_cpu_block,shared_shape,OPS,BS,GRB,TSO,gargs):
    """Use this function to execute core cpu only processes"""
    xslice = slice(shared_shape[2]-OPS-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2]-OPS,1)
    swb = (total_cpu_block[0],total_cpu_block[1],xslice,total_cpu_block[3])
    blocks,total_cpu_block = decomp.create_cpu_blocks(total_cpu_block,BS,shared_shape,OPS)
    cpu.set_globals(GRB,*gargs,source_mod=None)
    #Creating sets for cpu calculation
    DecompObj = functions.DecompCPU(0,[(x+OPS,y+OPS) for x,y in numpy.ndindex((BS[0],BS[1]))],TSO-1,cwrite=swb,cread=total_cpu_block)
    sgs.carr = decomp.create_shared_pool_array(sarr[total_cpu_block].shape)
    return DecompObj,blocks

def standard_gpu_core(blocks,BS,OPS,gargs,GRB,TSO):
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
    const_dict = ({"NV":NV,"VARS":VARS,"TIMES":TIMES,"OPS":OPS,"TSO":TSO,"SX":SX,"SY":SY})
    garr = decomp.create_local_gpu_array(block_shape)
    #Building CUDA source code
    SM = source.build_gpu_source(os.path.dirname(__file__))
    source.decomp_constant_copy(SM,const_dict)
    cpu.set_globals(GRB,*gargs,source_mod=SM)
    DecompObj  = functions.DecompGPU(0,SM.get_function("Decomp"),GRD,BS,gread,gwrite,garr)
    return DecompObj,garr

def create_local_gpu_array(block_shape):
    """Use this function to create the local gpu array"""
    arr = numpy.zeros(block_shape)
    arr = arr.astype(numpy.float64)
    arr = numpy.ascontiguousarray(arr)
    return arr



def get_local_extend_array(sarr,arr,blocks,BS):
    """Use this function to copy the shared array to the local array and extend each end by one block."""
    i1,i2,i3,i4 = blocks
    arr[:,:,:,BS[0]:-BS[0]] = sarr[blocks]
    arr[:,:,:,0:BS[0]] = sarr[i1,i2,i3,-BS[0]:]
    arr[:,:,:,-BS[0]:] = sarr[i1,i2,i3,:BS[0]]
    return arr

def create_shared_pool_array(shared_shape):
    """Use this function to create the shared array for the process pool."""
    sarr_base = mp.Array(ctypes.c_double, int(numpy.prod(shared_shape)),lock=False)
    sarr = numpy.ctypeslib.as_array(sarr_base)
    sarr = sarr.reshape(shared_shape)
    return sarr

def create_swept_cpu_blocks(total_cpu_block,BS,shared_shape):
    """Use this function to create blocks."""
    row_range = numpy.arange(0,total_cpu_block[2].stop-total_cpu_block[2].start,BS[1],dtype=numpy.intc)
    column_range = numpy.arange(total_cpu_block[3].start,total_cpu_block[3].stop,BS[1],dtype=numpy.intc)
    xslice = slice(shared_shape[2]-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2],1)
    ntcb = (total_cpu_block[0],total_cpu_block[1],xslice,total_cpu_block[3])
    return [(total_cpu_block[0],total_cpu_block[1],slice(x,x+BS[0],1),slice(y,y+BS[1],1)) for x,y in product(row_range,column_range)],ntcb

def create_standard_cpu_blocks(total_cpu_block,BS,shared_shape,ops):
    """Use this function to create blocks."""
    row_range = numpy.arange(ops,total_cpu_block[2].stop-total_cpu_block[2].start+ops,BS[1],dtype=numpy.intc)
    column_range = numpy.arange(ops,total_cpu_block[3].stop-total_cpu_block[3].start+ops,BS[1],dtype=numpy.intc)
    xslice = slice(shared_shape[2]-2*ops-(total_cpu_block[2].stop-total_cpu_block[2].start),shared_shape[2],1)
    yslice = slice(total_cpu_block[3].start-ops,total_cpu_block[3].stop+ops,1)
    #There is an issue with total cpu block for decomp on cluster
    ntcb = (total_cpu_block[0],total_cpu_block[1],xslice,yslice)
    return [(total_cpu_block[0],total_cpu_block[1],slice(x-ops,x+BS[0]+ops,1),slice(y-ops,y+BS[1]+ops,1)) for x,y in product(row_range,column_range)],ntcb

def create_swept_escpu_blocks(cpu_blocks,shared_shape,BS):
    """Use this function to create shift blocks and edge blocks."""
    #This handles edge blocks in the y direction
    shift = int(BS[1]/2)
    #Creating shift blocks and edges
    shift_blocks = [(block[0],block[1],block[2],slice(block[3].start+shift,block[3].stop+shift,1)) for block in cpu_blocks]
    for i,block in enumerate(shift_blocks):
        if block[3].start < 0:
            new_block = (block[0],block[0],block[2],numpy.arange(block[3].start,block[3].stop))
            shift_blocks[i] = new_block
        elif block[3].stop > shared_shape[3]:
            ny = numpy.concatenate((numpy.arange(block[3].start,block[3].stop-shift),numpy.arange(0,shift,1)))
            new_block = (block[0],block[0],block[2],ny)
            shift_blocks[i] = new_block
    return cpu_blocks,shift_blocks


def create_dist_up_sets(block_size,ops):
    """This function creates up sets for the dsweep."""
    sets = tuple()
    ul = block_size[0]-ops
    dl = ops
    while ul > dl:
        r = np.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl+=ops
        ul-=ops
    return sets

def create_dist_down_sets(block_size,ops):
    """Use this function to create the down pyramid sets from up sets."""
    bsx = int(block_size[0]/2)
    bsy = int(block_size[1]/2)
    dl = int((bsy)-ops); #lower y
    ul = int((bsy)+ops); #upper y
    sets = tuple()
    while dl > 0:
        r = np.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl-=ops
        ul+=ops
    return sets

def create_dist_bridge_sets(block_size,ops,MPSS):
    """Use this function to create the iidx sets for bridges."""
    sets = tuple()
    xul = block_size[0]-ops
    xdl = ops
    yul = int(block_size[0]/2+ops)
    ydl = int(block_size[0]/2-ops)
    xts = xul
    xbs = xdl
    for i in range(MPSS):
        sets+=(tuple(product(np.arange(xdl,xul,1),np.arange(ydl,yul,1))),)
        xdl+=ops
        xul-=ops
        ydl-=ops
        yul+=ops
    return sets,sets[::-1]
