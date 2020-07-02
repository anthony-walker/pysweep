
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
    #Setting up CPU
    setupCPUSwept(solver)
    solver.comm.Barrier() #Ensure all processes are


def setupCPUSwept(solver):
    """Use this function to execute core cpu only proceysses"""
    timeSlice = slice(0,solver.sharedShape[0],1)
    solver.blocks = [(timeSlice,)+tuple(block) for block in solver.blocks]
    solver.edgeblocks = makeEdgeBlocksSwept(solver.blocks,solver.arrayShape,solver.blocksize)
    solver.cpu.set_globals(*solver.globals)
    #Creating sets for cpu calculation
    up_sets = createUpPyramidSets(solver.blocksize,solver.operating)
    down_sets = createDownPyramidSets(solver.blocksize,solver.operating)
    oct_sets = down_sets+up_sets
    y_sets,x_sets = createBridgeSets(solver.blocksize,solver.operating,solver.maxPyramidSize)
    cshape = solver.sharedArray[solver.blocks[0]].shape  if solver.blocks else (0,)
    #Initializing CPU portion of Geometry
    solver.Up.initializeCPU(solver.cpu,up_sets,solver.intermediate-1,cshape) 
    solver.Down.initializeCPU(solver.cpu,down_sets,solver.intermediate-1,cshape)
    solver.Xb.initializeCPU(solver.cpu,x_sets,solver.intermediate-1,cshape)
    solver.Yb.initializeCPU(solver.cpu,y_sets,solver.intermediate-1,cshape)
    solver.Oct.initializeCPU(solver.cpu,oct_sets,solver.intermediate-1,cshape)
    #Setting starting swept step
    solver.Up.setSweptStep(solver.intermediate-1)
    solver.Down.setSweptStep(solver.intermediate-1)
    solver.Xb.setSweptStep(solver.intermediate-1)
    solver.Yb.setSweptStep(solver.intermediate-1)
    solver.Oct.setSweptStep(solver.intermediate-1)

def getGPUReadBlockSwept(solver):
    """Use this function to create the GPU read block."""
    solver.gpuReadBlock = solver.gpuBlock[:-1]
    yshape = solver.sharedShape[-1]
    startblock = numpy.arange(yshape-solver.blocksize[0],yshape,1)
    centerblock = numpy.arange(0,yshape,1)
    endblock = numpy.arange(0,solver.blocksize[0],1)
    extendedBlock = (numpy.concatenate((startblock,centerblock,endblock)),)   
    solver.gpuReadBlock = solver.gpuReadBlock+extendedBlock
 
def setupGPUSwept(solver):
    """Use this function to execute core gpu only processes"""
    solver.gpuBlock = (slice(0,solver.sharedShape[0],1),)+solver.gpuBlock
    getGPUReadBlockSwept(solver) #Finish creating gpuReadBlock here
    blockShape =[element.stop for element in solver.gpuBlock]
    blockShape[-1] += int(2*solver.blocksize[0]) #Adding 2 blocks in the column direction
    # Creating local GPU array with split
    grid = (int((blockShape[2])/solver.blocksize[0]),int((blockShape[3])/solver.blocksize[1]))   #Grid size
    #Creating constants
    bsp = lambda x: int(numpy.prod(blockShape[x:])) #block shape product returned as an integer
    const_dict = ({"NV":blockShape[1],'SX':blockShape[2],'SY':blockShape[3],"VARS":bsp(2),"TIMES":bsp(1),"MPSS":solver.maxPyramidSize,"MOSS":solver.maxOctSize,"OPS":solver.operating,"ITS":solver.intermediate})
    solver.GPUArray = mallocGPUArray(blockShape) #Allocated GPU
    solver.localGPUArray = numpy.zeros(blockShape)
    #Building CUDA source code
    solver.gpu = io.buildGPUSource(solver.gpu)
    io.copyConstants(solver.gpu,const_dict) #This copys cpu constants not global constants
    solver.cpu.set_globals(*solver.globals,source_mod=solver.gpu)
    # Make GPU geometry
    solver.Up.initializeGPU(solver.gpu.get_function("UpPyramid"),solver.blocksize,(grid[0],grid[1]-1))
    solver.Oct.initializeGPU(solver.gpu.get_function("Octahedron"),solver.blocksize,(grid[0],grid[1]-1))
    solver.Down.initializeGPU(solver.gpu.get_function("DownPyramid"),solver.blocksize,(grid[0],grid[1]-1))
    solver.Yb.initializeGPU(solver.gpu.get_function("YBridge"),solver.blocksize,grid)
    solver.Xb.initializeGPU(solver.gpu.get_function("XBridge"),solver.blocksize,grid)
    

def mallocGPUArray(blockShape):
    """Use this function to create the local gpu array"""
    arr = numpy.zeros(blockShape)
    arr = arr.astype(numpy.float64)
    arr = numpy.ascontiguousarray(arr)
    return cuda.mem_alloc(arr.nbytes)

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

def setupCPUStandard(solver):
    """Use this function to execute core cpu only processes"""
    #Adjust blocks for boundary conditions
    makeReadBlocksStandard(solver,solver.operating)
    solver.cpu.set_globals(*solver.globals)
    #Creating sets for cpu calculation
    standardSet = [(x+solver.operating,y+solver.operating) for x,y in numpy.ndindex(solver.blocksize[:-1])]
    #Initializing CPU on standard
    cshape = solver.sharedArray[solver.blocks[0][1]].shape if solver.blocks else (0,)
    solver.standard.initializeCPU(solver.cpu,standardSet,solver.intermediate-1,cshape) 


def makeReadBlocksStandard(solver, adjustment):
    """Use this function to create blocks."""
    #Correct CPU block
    timeSlice = slice(0,solver.sharedShape[0],1)
    solver.blocks = [(timeSlice,)+tuple(block) for block in solver.blocks]
    #Make CPU read blocks
    if solver.blocks: #Just in case blocks are empty
        readBlocks = list()
        for block in solver.blocks:
            readBlocks.append(block[:2]+tuple([slice(curr.start-adjustment,curr.stop+adjustment,1) for curr in block[2:]]))
        solver.blocks = list(zip(solver.blocks,readBlocks))


def setupGPUStandard(solver):
    """Use this function to execute core gpu only processes"""
    #Correct GPU block
    solver.gpuBlock = (slice(0,solver.sharedShape[0],1),)+solver.gpuBlock
    solver.gpuReadBlock = solver.gpuBlock[:2]+tuple([slice(cs.start-solver.operating,cs.stop+solver.operating,1) for cs in solver.gpuBlock[2:]])
    blockShape =[element.stop-element.start for element in solver.gpuBlock]
    blockShape[-1] += int(2*solver.operating) #Adding 2 blocks in the column direction
    blockShape[-2] += int(2*solver.operating) #Adding 2 blocks in the column direction
    #Creating local GPU array with split
    grid = (int((blockShape[2]-2*solver.operating)/solver.blocksize[0]),int((blockShape[3]-2*solver.operating)/solver.blocksize[1]))   #Grid size
    #Creating constants
    bsp = lambda x: int(numpy.prod(blockShape[x:])) #block shape product returned as an integer
    const_dict = ({"NV":blockShape[1],'SX':blockShape[2],'SY':blockShape[3],"VARS":bsp(2),"TIMES":bsp(1),"OPS":solver.operating,"ITS":solver.intermediate})
    solver.GPUArray = mallocGPUArray(blockShape)
    solver.localGPUArray = numpy.zeros(blockShape)
    #Building CUDA source code
    solver.gpu = io.buildGPUSource(solver.gpu)
    io.copyConstants(solver.gpu,const_dict) #This copys gpu constants not global constants
    solver.cpu.set_globals(*solver.globals,source_mod=solver.gpu)
    #Make GPU geometry
    solver.standard.initializeGPU(solver.gpu.get_function("Standard"),solver.blocksize,grid)
    
def standardBlock(solver):
    """This is the entry point for the standard portion of the block module."""
    #Create and fill shared array
    createCPUSharedArray(solver,numpy.zeros(solver.sharedShape,dtype=solver.dtype).nbytes)
    for i in range(solver.intermediate):
        solver.sharedArray[i,:,solver.operating:-solver.operating,solver.operating:-solver.operating] =  solver.initialConditions[solver.globalBlock]
        solver.sharedArray[i,:,solver.operating:-solver.operating,:solver.operating] = solver.initialConditions[solver.globalBlock[0],solver.globalBlock[1],-solver.operating-1:-1]
        solver.sharedArray[i,:,solver.operating:-solver.operating,-solver.operating:] = solver.initialConditions[solver.globalBlock[0],solver.globalBlock[1],1:solver.operating+1]
    #Create phase objects
    solver.standard = geometry.Geometry() 
    solver.standard.setAdjustment(solver.operating)
    #Setting up GPU
    if solver.gpuBool:
        # Creating cuda device and context
        cuda.init()
        cuda_device = cuda.Device(solver.gpuRank)
        solver.cuda_context = cuda_device.make_context()
        setupGPUStandard(solver)
    #Setup CPU
    setupCPUStandard(solver)
    solver.comm.Barrier() #Ensure all processes are


