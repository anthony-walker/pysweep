
import numpy,mpi4py.MPI as MPI,ctypes,os
import pysweep.core.geometry as geometry
import pysweep.core.io as io
from itertools import product

try:
    import pycuda.driver as cuda
except Exception as e:
    # print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")
    pass

def createCPUSharedArray(solver,arrayBytes):
    """Use this function to create shared memory arrays for node communication."""
    itemsize = int(solver.dtype.itemsize)
    #Creating MPI Window for shared memory
    win = MPI.Win.Allocate_shared(arrayBytes, itemsize, comm=solver.nodeComm)
    sharedBuffer, itemsize = win.Shared_query(0)
    solver.sharedArray = numpy.ndarray(buffer=sharedBuffer, dtype=solver.dtype.type, shape=solver.sharedShape)

#-----------------------------Swept functions----------------------------------------#

def sweptBlock(solver):
    """This is the entry point for the swept portion of the block module."""
    #Create and fill shared array
    createCPUSharedArray(solver,numpy.zeros(solver.sharedShape,dtype=solver.dtype).nbytes)
    for i in range(solver.intermediate):
        solver.sharedArray[i,:,:,:] =  solver.initialConditions[solver.globalBlock]
    #Create phase objects
    solver.Up = geometry.sweptGeometry() 
    solver.Down = geometry.sweptGeometry() 
    solver.Xb = geometry.sweptGeometry()
    solver.Yb = geometry.sweptGeometry()
    solver.Oct = geometry.sweptGeometry() 

    if solver.gpuBool:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(solver.gpuRank)
        solver.cuda_context = cuda_device.make_context()
        setupGPUSwept(solver)

    setupCPUSwept(solver)
    solver.comm.Barrier() #Ensure all processes are


def setupCPUSwept(solver):
    """Use this function to execute core cpu only proceysses"""
    timeSlice = slice(0,solver.sharedShape[0],1)
    solver.blocks = [(timeSlice,)+tuple(block) for block in solver.blocks]
    solver.edgeblocks = makeEdgeBlocksSwept(solver.blocks,solver.arrayShape,solver.blocksize)
    solver.cpu.set_globals(False,*solver.globals)
    #Creating sets for cpu calculation
    up_sets = createUpPyramidSets(solver.blocksize,solver.operating)
    down_sets = createDownPyramidSets(solver.blocksize,solver.operating)
    oct_sets = down_sets+up_sets
    y_sets,x_sets = createBridgeSets(solver.blocksize,solver.operating,solver.maxPyramidSize)
    #Create function objects
    solver.Up.initializeCPU(solver.cpu,up_sets,solver.intermediate,solver.intermediate-1) 
    solver.Down.initializeCPU(solver.cpu,down_sets,solver.intermediate,solver.intermediate-1)
    solver.Xb.initializeCPU(solver.cpu,x_sets,solver.intermediate,solver.intermediate-1)
    solver.Yb.initializeCPU(solver.cpu,y_sets,solver.intermediate,solver.intermediate-1)
    solver.Oct.initializeCPU(solver.cpu,oct_sets,solver.intermediate,solver.intermediate-1)

def setupGPUSwept(solver):
    """Use this function to execute core gpu only processes"""
    solver.gpuBlock = (slice(0,solver.sharedShape[0],1),)+solver.gpuBlock
    blockShape =[element.stop for element in solver.gpuBlock]
    blockShape[-1] += int(2*solver.blocksize[0]) #Adding 2 blocks in the column direction
    # Creating local GPU array with split
    grid = (int((blockShape[2])/solver.blocksize[0]),int((blockShape[3])/solver.blocksize[1]))   #Grid size
    #Creating constants
    bsp = lambda x: int(numpy.prod(blockShape[x:])) #block shape product returned as an integer
    const_dict = ({"NV":blockShape[1],'SX':blockShape[2],'SY':blockShape[3],"VARS":bsp(2),"TIMES":bsp(1),"MPSS":solver.maxPyramidSize,"MOSS":solver.maxOctSize,"OPS":solver.operating,"ITS":solver.intermediate})
    solver.GPUArray = createLocalGPUArray(blockShape)
    solver.GPUArray = cuda.mem_alloc(solver.GPUArray.nbytes)
    #Building CUDA source code
    solver.gpu = io.buildGPUSource(solver.gpu)
    io.copySweptConstants(solver.gpu,const_dict) #This copys swept constants not global constants
    solver.cpu.set_globals(solver.gpuBool,*solver.globals,source_mod=solver.gpu)
    # Make GPU geometry
    solver.Up.initializeGPU(solver.gpu.get_function("UpPyramid"),solver.blocksize,grid,blockShape)
    solver.Down.initializeGPU(solver.gpu.get_function("DownPyramid"),solver.blocksize,grid,blockShape)
    solver.Yb.initializeGPU(solver.gpu.get_function("YBridge"),solver.blocksize,(grid[0],grid[1]-1),blockShape)
    solver.Xb.initializeGPU(solver.gpu.get_function("XBridge"),solver.blocksize,(grid[0],grid[1]-1),blockShape)
    solver.Oct.initializeGPU(solver.gpu.get_function("Octahedron"),solver.blocksize,(grid),blockShape)

def createLocalGPUArray(blockShape):
    """Use this function to create the local gpu array"""
    arr = numpy.zeros(blockShape)
    arr = arr.astype(numpy.float64)
    arr = numpy.ascontiguousarray(arr)
    return arr

def makeEdgeBlocksSwept(cpuBlocks,sharedShape,blocksize):
    """Use this function to create shift blocks and edge blocks."""
    #This handles edge blocks in the y direction
    shift = int(blocksize[1]/2)
    #Creating shift blocks and edges
    shiftBlocks = [(block[0],block[1],block[2],slice(block[3].start+shift,block[3].stop+shift,1)) for block in cpuBlocks]
    for i,block in enumerate(shiftBlocks):
        if block[3].start < 0:
            newBlock = (block[0],block[0],block[2],numpy.arange(block[3].start,block[3].stop))
            shiftBlocks[i] = newBlock
        elif block[3].stop > sharedShape[3]:
            ny = numpy.concatenate((numpy.arange(block[3].start,block[3].stop-shift),numpy.arange(0,shift,1)))
            newBlock = (block[0],block[1],block[2],ny) #Changed 1 from 0 here, if bug hunting later
            shiftBlocks[i] = newBlock
    return shiftBlocks


def createUpPyramidSets(blocksize,operating):
    """This function creates up sets for the dsweep."""
    sets = tuple()
    ul = blocksize[0]-operating
    dl = operating
    while ul > dl:
        r = numpy.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl+=operating
        ul-=operating
    return sets

def createDownPyramidSets(blocksize,operating):
    """Use this function to create the down pyramid sets from up sets."""
    bsx = int(blocksize[0]/2)
    bsy = int(blocksize[1]/2)
    dl = int((bsy)-operating); #lower y
    ul = int((bsy)+operating); #upper y
    sets = tuple()
    while dl > 0:
        r = numpy.arange(dl,ul,1)
        sets+=(tuple(product(r,r)),)
        dl-=operating
        ul+=operating
    return sets

def createBridgeSets(blocksize,operating,MPSS):
    """Use this function to create the iidx sets for bridges."""
    sets = tuple()
    xul = blocksize[0]-operating
    xdl = operating
    yul = int(blocksize[0]/2+operating)
    ydl = int(blocksize[0]/2-operating)
    xts = xul
    xbs = xdl
    for i in range(MPSS):
        sets+=(tuple(product(numpy.arange(xdl,xul,1),numpy.arange(ydl,yul,1))),)
        xdl+=operating
        xul-=operating
        ydl-=operating
        yul+=operating
    return sets,sets[::-1]

#-----------------------------Standard functions----------------------------------------#
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

def setupCPUStandard(sarr,totalCPUBlock,shared_shape,OPS,BS,GRB,ITS,gargs):
    """Use this function to execute core cpu only processes"""
    xslice = slice(shared_shape[2]-OPS-(totalCPUBlock[2].stop-totalCPUBlock[2].start),shared_shape[2]-OPS,1)
    swb = (totalCPUBlock[0],totalCPUBlock[1],xslice,totalCPUBlock[3])
    blocks,totalCPUBlock = decomp.create_cpu_blocks(totalCPUBlock,BS,shared_shape,OPS)
    cpu.set_globals(GRB,*gargs,source_mod=None)
    #Creating sets for cpu calculation
    DecompObj = functions.DecompCPU(0,[(x+OPS,y+OPS) for x,y in numpy.ndindex((BS[0],BS[1]))],ITS-1,cwrite=swb,cread=totalCPUBlock)
    sgs.CPUArray = decomp.createSharedPoolArray(sarr[totalCPUBlock].shape)
    return DecompObj,blocks

def create_standard_cpu_blocks(totalCPUBlock,BS,shared_shape,ops):
    """Use this function to create blocks."""
    row_range = numpy.arange(ops,totalCPUBlock[2].stop-totalCPUBlock[2].start+ops,BS[1],dtype=numpy.intc)
    column_range = numpy.arange(ops,totalCPUBlock[3].stop-totalCPUBlock[3].start+ops,BS[1],dtype=numpy.intc)
    xslice = slice(shared_shape[2]-2*ops-(totalCPUBlock[2].stop-totalCPUBlock[2].start),shared_shape[2],1)
    yslice = slice(totalCPUBlock[3].start-ops,totalCPUBlock[3].stop+ops,1)
    #There is an issue with total cpu block for decomp on cluster
    ntcb = (totalCPUBlock[0],totalCPUBlock[1],xslice,yslice)
    return [(totalCPUBlock[0],totalCPUBlock[1],slice(x-ops,x+BS[0]+ops,1),slice(y-ops,y+BS[1]+ops,1)) for x,y in product(row_range,column_range)],ntcb


def setupGPUStandard(solver):
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


def standardBlock(solver):
    """This is the entry point for the standard portion of the block module."""
    #Create and fill shared array
    createCPUSharedArray(solver,numpy.zeros(solver.sharedShape,dtype=solver.dtype).nbytes)
    for i in range(solver.intermediate):
        solver.sharedArray[i,:,solver.operating:-solver.operating,solver.operating:-solver.operating] =  solver.initialConditions[solver.globalBlock]
        solver.sharedArray[i,:,solver.operating:-solver.operating,:solver.operating] = solver.initialConditions[solver.globalBlock[0],solver.globalBlock[1],-solver.operating-1:-1]
        solver.sharedArray[i,:,solver.operating:-solver.operating,-solver.operating:] = solver.initialConditions[solver.globalBlock[0],solver.globalBlock[1],1:solver.operating+1]
        

    #Create phase objects
    # solver.Up = geometry.standardGeometry() 
    # solver.Down = geometry.standardGeometry() 
    # solver.Xb = geometry.standardGeometry()
    # solver.Yb = geometry.standardGeometry()
    # solver.Oct = geometry.standardGeometry() 

    if solver.gpuBool:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(solver.gpuRank)
        solver.cuda_context = cuda_device.make_context()
        setupGPUStandard(solver)

    setupCPUStandard(solver)
    solver.comm.Barrier() #Ensure all processes are

