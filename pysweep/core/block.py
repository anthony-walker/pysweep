
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
    solver.sharedArray[solver.intermediate-1,:,:,:] =  solver.initialConditions[solver.globalBlock]
    solver.sharedArray[0,:,:,:] =  solver.initialConditions[solver.globalBlock]
    #Create phase objects
    solver.Up = geometry.Geometry() 
    solver.Down = geometry.Geometry() 
    solver.Xb = geometry.Geometry()
    solver.Yb = geometry.Geometry()
    solver.Oct = geometry.Geometry() 

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
    solver.Up.initializeGPU(solver.gpu.get_function("UpPyramid"),solver.blocksize,solver.grid,blockShape)
    solver.Down.initializeGPU(solver.gpu.get_function("DownPyramid"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),blockShape)
    solver.Yb.initializeGPU(solver.gpu.get_function("YBridge"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),blockShape)
    solver.Xb.initializeGPU(solver.gpu.get_function("XBridge"),solver.blocksize,solver.grid,blockShape)
    solver.Oct.initializeGPU(solver.gpu.get_function("Octahedron"),solver.blocksize,(solver.grid[0],solver.grid[1]-1),blockShape)

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
