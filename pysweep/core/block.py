
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
    #Create phase objects
    solver.Up = functions.Geometry() 
    solver.Down = functions.Geometry() 
    solver.Xb = functions.Geometry()
    solver.Yb = functions.Geometry()
    solver.Oct = functions.Geometry() 

    if solver.gpuBool:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(solver.gpuRank)
        solver.cuda_context = cuda_device.make_context()
        setupGPUSwept(solver)
    else:
        solver.gpu,solver.GPUArray = None,None

    setupCPUSwept(solver)
    solver.comm.Barrier() #Ensure all processes are


def setupCPUSwept(solver):
    """Use this function to execute core cpu only processes"""
    solver.edgeblocks = makeEdgeBlocksSwept(solver.blocks,solver.arrayShape,solver.blocksize)
    solver.cpu.set_globals(False,*solver.globals)
    #Creating sets for cpu calculation
    up_sets = createUpPyramidSets(solver.blocksize,solver.operating)
    down_sets = createDownPyramidSets(solver.blocksize,solver.operating)
    oct_sets = down_sets+up_sets
    y_sets,x_sets = createBridgeSets(solver.blocksize,solver.operating,solver.maxPyramidSize)
    #Create function objects
    solver.Up.initializeCPU(up_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1) 
    solver.Down.initializeCPU(down_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)
    solver.Xb.initializeCPU(x_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)
    solver.Yb.initializeCPU(y_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)
    solver.Oct.initializeCPU(oct_sets,solver.intermediate,solver.maxPyramidSize,solver.intermediate-1)

def setupGPUSwept(solver):
    """Use this function to execute core gpu only processes"""
    blockShape =[solver.sharedShape[0],]+[element.stop for element in solver.gpuBlock]
    blockShape[-1] += int(2*solver.blocksize[0]) #Adding 2 blocks in the column direction
    # Creating local GPU array with split
    solver.grid = (int((blockShape[2])/solver.blocksize[0]),int((blockShape[3])/solver.blocksize[1]))   #Grid size
    #Creating constants
    solver.NV = blockShape[1]
    solver.SX = blockShape[2]
    solver.SY = blockShape[3]
    solver.VARS =  int(solver.SX*solver.SY)
    solver.TIMES =solver.VARS*solver.NV
    const_dict = ({"NV":solver.NV,"VARS":solver.VARS,"TIMES":solver.TIMES,"MPSS":solver.maxPyramidSize,"MOSS":solver.maxOctSize,"OPS":solver.operating,"TSO":solver.intermediate,'SX':solver.SX,'SY':solver.SY})
    solver.GPUArray = createLocalGPUArray(blockShape)
    solver.GPUArray = cuda.mem_alloc(solver.GPUArray.nbytes)
    #Building CUDA source code
    solver.gpu = io.buildGPUSource(solver.gpu)
    io.copySweptConstants(solver.gpu,const_dict) #This copys swept constants not global constants
    solver.cpu.set_globals(solver.gpuBool,*solver.globals,source_mod=solver.gpu)
    # Make GPU geometry
    solver.Up.initializeGPU(solver.gpu.get_function("UpPyramid"),solver.blocksize,solver.grid,solver.maxPyramidSize,blockShape)
    solver.Down.initializeGPU(solver.gpu.get_function("DownPyramid"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),solver.maxPyramidSize,blockShape)
    solver.Yb.initializeGPU(solver.gpu.get_function("YBridge"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),solver.maxPyramidSize,blockShape)
    solver.Xb.initializeGPU(solver.gpu.get_function("XBridge"),solver.blocksize,solver.grid,solver.maxPyramidSize,blockShape)
    solver.Oct.initializeGPU(solver.gpu.get_function("Octahedron"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),solver.maxPyramidSize,blockShape)

def createLocalGPUArray(blockShape):
    """Use this function to create the local gpu array"""
    arr = numpy.zeros(blockShape)
    arr = arr.astype(numpy.float64)
    arr = numpy.ascontiguousarray(arr)
    return arr

def makeEdgeBlocksSwept(cpu_blocks,shared_shape,BS):
    """Use this function to create shift blocks and edge blocks."""
    #This handles edge blocks in the y direction
    shift = int(BS[1]/2)
    #Creating shift blocks and edges
    shift_blocks = [(block[0],block[1],slice(block[2].start+shift,block[2].stop+shift,1)) for block in cpu_blocks]
    for i,block in enumerate(shift_blocks):
        if block[2].start < 0:
            new_block = (block[0],block[1],numpy.arange(block[2].start,block[2].stop))
            shift_blocks[i] = new_block
        elif block[2].stop > shared_shape[3]:
            ny = numpy.concatenate((numpy.arange(block[2].start,block[2].stop-shift),numpy.arange(0,shift,1)))
            new_block = (block[0],block[1],ny)
            shift_blocks[i] = new_block
    return shift_blocks


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
