#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
#------------------------------Decomp Functions----------------------------------
import numpy
try:
    import pysweep.core.io as io
    import pycuda.driver as cuda
except Exception as e:
    pass
#------------------------------Swept Functions----------------------------------

def FirstPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Start GPU Operations
    if solver.gpuBool:
        solver.localGPUArray[:,:,:,:] = solver.sharedArray[solver.gpuReadBlock]
        cuda.memcpy_htod(solver.GPUArray,solver.localGPUArray)
        solver.Up.callGPU(solver.GPUArray,solver.globalTimeStep)
        solver.Yb.callGPU(solver.GPUArray,solver.globalTimeStep)
    #Do CPU Operations
    solver.Up.callCPU(solver.sharedArray,solver.edgeblocks,solver.globalTimeStep)
    solver.nodeComm.Barrier() #need barrier to make sure all data is in place for next step
    solver.Yb.callCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(solver.localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=solver.localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
    solver.Yb.startAdd(solver.maxPyramidSize) #Change starting location for UpPrism
    solver.nodeComm.Barrier() #Node barrier here before the write

def UpPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Start GPU Operations
    if solver.gpuBool:
        solver.localGPUArray[:,:,:,:] = solver.sharedArray[solver.gpuReadBlock]
        cuda.memcpy_htod(solver.GPUArray,solver.localGPUArray)  
        solver.Xb.callGPU(solver.GPUArray,solver.globalTimeStep)
        solver.Oct.callGPU(solver.GPUArray,solver.globalTimeStep)
        solver.Yb.callGPU(solver.GPUArray,solver.globalTimeStep+solver.maxPyramidSize)
    #Do CPU  Operations
    solver.Xb.callCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    solver.nodeComm.Barrier() #need barrier to make sure all data is in place for next step
    solver.Oct.callCPU(solver.sharedArray,solver.edgeblocks,solver.globalTimeStep)
    solver.nodeComm.Barrier() #need barrier to make sure all data is in place for next step
    solver.globalTimeStep+=solver.maxPyramidSize #Add pyramid to global time step so ybridge starts at appropriate step for intermediate schemes
    solver.Yb.callCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(solver.localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=solver.localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
    solver.nodeComm.Barrier() #Node barrier here before the write

def LastPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Start GPU Operations
    if solver.gpuBool:
        solver.localGPUArray[:,:,:,:] = solver.sharedArray[solver.gpuReadBlock]
        cuda.memcpy_htod(solver.GPUArray,solver.localGPUArray)  
        solver.Xb.callGPU(solver.GPUArray,solver.globalTimeStep)
        solver.Down.callGPU(solver.GPUArray,solver.globalTimeStep)
    #Do CPU Operations
    solver.Xb.callCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    solver.nodeComm.Barrier() #need barrier to make sure all data is in place for next step
    solver.Down.callCPU(solver.sharedArray,solver.edgeblocks,solver.globalTimeStep)
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(solver.localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=solver.localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
    solver.globalTimeStep+=solver.maxPyramidSize #Need this for the write of intermediate steps
    solver.nodeComm.Barrier() #Node barrier here before the write

def firstForward(solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        buff = numpy.copy(solver.sharedArray[:,:,-solver.splitx:,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[1],source=solver.neighbors[0])
        solver.clusterComm.Barrier() #Barrier to make sure all buffers are copied before writing
        solver.sharedArray[:,:,solver.splitx:,:] = solver.sharedArray[:,:,:-solver.splitx,:] #Shift solver.sharedArray data forward by solver.splitx
        solver.sharedArray[:,:,:solver.splitx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier() #Wait for copy before calculating again


def sendForward(cwt,solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        cwt = io.sweptWrite(cwt,solver)
        buff = numpy.copy(solver.sharedArray[:,:,-solver.splitx:,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[1],source=solver.neighbors[0])
        solver.clusterComm.Barrier() #Barrier to make sure all buffers are copied before writing
        solver.sharedArray[:,:,solver.splitx:,:] = solver.sharedArray[:,:,:-solver.splitx,:] #Shift solver.sharedArray data forward by solver.splitx
        solver.sharedArray[:,:,:solver.splitx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier() #Wait for copy before calculating again
    return cwt

def sendBackward(cwt,solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        buff = numpy.copy(solver.sharedArray[:,:,:solver.splitx,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[0],source=solver.neighbors[1])
        solver.clusterComm.Barrier() #Barrier to make sure all buffers are copied before writing
        solver.sharedArray[:,:,:-solver.splitx,:] = solver.sharedArray[:,:,solver.splitx:,:] #Shift solver.sharedArray backward data by solver.splitx
        solver.sharedArray[:,:,-solver.splitx:,:] = buffer[:,:,:,:]
        cwt = io.sweptWrite(cwt,solver)
    solver.nodeComm.Barrier() #Wait for copy before calculating again
    return cwt

#----------------------------------Standard Functions----------------------------------------------#
def StandardFunction(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Calling Standard GPU
    if solver.gpuBool:
        solver.localGPUArray[:,:,:,:] = solver.sharedArray[solver.gpuReadBlock]
        cuda.memcpy_htod(solver.GPUArray,solver.localGPUArray)  
        solver.standard.callStandardGPU(solver.GPUArray,solver.globalTimeStep)      
    #Calling standard CPU 
    solver.standard.callStandardCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    #Clean up GPU processes
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(solver.localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=solver.localGPUArray[:,:,solver.operating:-solver.operating,solver.operating:-solver.operating]
    solver.globalTimeStep+=1 #Update global time step
    solver.nodeComm.Barrier() #Node barrier here before the write

def sendEdges(solver):
    """Use this function to communicate data between nodes"""
    ops = solver.operating
    if solver.nodeMasterBool:
        solver.sharedArray[:,:,ops:-ops,:ops] = solver.sharedArray[:,:,ops:-ops,-2*ops:-ops] #Copy y boundaries
        solver.sharedArray[:,:,ops:-ops,-ops:] = solver.sharedArray[:,:,ops:-ops,ops:2*ops]
        bufffor = numpy.copy(solver.sharedArray[:,:,-2*ops:-ops,:])
        buffback = numpy.copy(solver.sharedArray[:,:,ops:2*ops,:])
        bufferf = solver.clusterComm.sendrecv(sendobj=bufffor,dest=solver.neighbors[1],source=solver.neighbors[0])
        bufferb = solver.clusterComm.sendrecv(sendobj=buffback,dest=solver.neighbors[0],source=solver.neighbors[1])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,:ops,:] = bufferf[:,:,:,:]
        solver.sharedArray[:,:,-ops:,:] = bufferb[:,:,:,:]
    solver.nodeComm.Barrier() #Wait for copy before calculating again

