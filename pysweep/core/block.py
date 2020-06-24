
import numpy,mpi4py.MPI as MPI,ctypes,os
import pysweep.core.functions as functions
import pysweep.core.io as io
import pysweep.core.sgs as sgs
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
    solver.sharedArray[solver.intermediate-1,:,:,:] =  solver.initialConditions[solver.globalBlock]
    solver.sharedArray[0,:,:,:] =  solver.initialConditions[solver.globalBlock]

    if solver.gpuBool:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(solver.gpuRank)
        solver.cuda_context = cuda_device.make_context()
        setupGPUSwept(solver)
        solver.mpiPool,solver.totalCPUBlock = None,None
    else:
        solver.gpu,solver.GPUArray = None,None

    
    # setupCPUSwept(solver)
    solver.comm.Barrier() #Ensure all processes are


def setupCPUSwept(solver):
    """Use this function to execute core cpu only processes"""
    solver.blocks,solver.totalCPUBlock = makeCPUBlocksSwept(solver.blocks,solver.blocksize,solver.arrayShape)
    solver.blocks = makeECPUBlocksSwept(solver.blocks,solver.arrayShape,solver.blocksize)
    solver.cpu.set_globals(False,*solver.globals)
    #Creating sets for cpu calculation
    up_sets = createUpPyramidSets(solver.blocksize,solver.operating)
    down_sets = createDownPyramidSets(solver.blocksize,solver.operating)
    oct_sets = down_sets+up_sets
    y_sets,x_sets = createBridgeSets(solver.blocksize,solver.operating,solver.maxPyramidSize)
    #Shared array
    sgs.CPUArray = createSharedPoolArray(solver.sharedArray[solver.totalCPUBlock].shape)
    sgs.CPUArray[:,:,:,:] = solver.sharedArray[solver.totalCPUBlock]
    sgs.CPUArray[:,:,:,:] = solver.sharedArray[solver.totalCPUBlock]
    #Create function objects
    solver.Up = functions.GeometryCPU(0,up_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)
    solver.Down = functions.GeometryCPU(0,down_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)
    solver.Xb = functions.GeometryCPU(0,x_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)
    solver.Yb = functions.GeometryCPU(0,y_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)
    solver.Oct = functions.GeometryCPU(0,oct_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)

def setupGPUSwept(solver):
    """Use this function to execute core gpu only processes"""
    #PICK UP HERE
    # block_shape = [i.stop-i.start for i in solver.blocks]
    # block_shape[-1] += int(2*solver.blocksize[0]) #Adding 2 blocks in the column direction
    # # Creating local GPU array with split
    # solver.grid = (int((block_shape[2])/solver.blocksize[0]),int((block_shape[3])/solver.blocksize[1]))   #Grid size
    #Creating constants
    solver.NV = block_shape[1]
    solver.SX = block_shape[2]
    solver.SY = block_shape[3]
    solver.VARS =  int(solver.SX*solver.SY)
    solver.TIMES =solver.VARS*solver.NV
    const_dict = ({"NV":solver.NV,"VARS":solver.VARS,"TIMES":solver.TIMES,"MPSS":solver.maxPyramidSize,"MOSS":solver.maxOctSize,"OPS":solver.operating,"TSO":solver.intermediate,'SX':solver.SX,'SY':solver.SY})
    solver.GPUArray = createLocalGPUArray(block_shape)
    solver.GPUArray = cuda.mem_alloc(solver.GPUArray.nbytes)
    #Building CUDA source code
    solver.gpu = io.buildGPUSource(solver.gpu)
    io.copySweptConstants(solver.gpu,const_dict) #This copys swept constants not global constants
    solver.cpu.set_globals(solver.gpuBool,*solver.globals,source_mod=solver.gpu)
    # Make GPU geometry
    solver.Up = functions.GeometryGPU(0,solver.gpu.get_function("UpPyramid"),solver.blocksize,solver.grid,solver.maxPyramidSize,block_shape)
    solver.Down = functions.GeometryGPU(0,solver.gpu.get_function("DownPyramid"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),solver.maxPyramidSize,block_shape)
    solver.Yb = functions.GeometryGPU(0,solver.gpu.get_function("YBridge"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),solver.maxPyramidSize,block_shape)
    solver.Xb = functions.GeometryGPU(0,solver.gpu.get_function("XBridge"),solver.blocksize,solver.grid,solver.maxPyramidSize,block_shape)
    solver.Oct = functions.GeometryGPU(0,solver.gpu.get_function("Octahedron"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),solver.maxPyramidSize,block_shape)

def createLocalGPUArray(block_shape):
    """Use this function to create the local gpu array"""
    arr = numpy.zeros(block_shape)
    arr = arr.astype(numpy.float64)
    arr = numpy.ascontiguousarray(arr)
    return arr

def makeCPUBlocksSwept(totalCPUBlock,BS,shared_shape):
    """Use this function to create blocks."""
    row_range = numpy.arange(0,totalCPUBlock[2].stop-totalCPUBlock[2].start,BS[1],dtype=numpy.intc)
    column_range = numpy.arange(totalCPUBlock[3].start,totalCPUBlock[3].stop,BS[1],dtype=numpy.intc)
    xslice = slice(shared_shape[2]-(totalCPUBlock[2].stop-totalCPUBlock[2].start),shared_shape[2],1)
    ntcb = (totalCPUBlock[0],totalCPUBlock[1],xslice,totalCPUBlock[3])
    return [(totalCPUBlock[0],totalCPUBlock[1],slice(x,x+BS[0],1),slice(y,y+BS[1],1)) for x,y in product(row_range,column_range)],ntcb


def makeECPUBlocksSwept(cpu_blocks,shared_shape,BS):
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
