#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
#------------------------------Decomp Functions----------------------------------
import numpy


try:
    import pycuda.driver as cuda
except Exception as e:
    # print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")
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
        localGPUArray = getLocalExtendedArray(solver.sharedArray,numpy.zeros(solver.Up.shape),solver.gpuBlock,solver.Up.blocksize)
        cuda.memcpy_htod(solver.GPUArray,localGPUArray)
        solver.Up.callGPU(solver.GPUArray)
        solver.Yb.callGPU(solver.GPUArray)
    #Do CPU Operations
    solver.Up.callCPU(solver.sharedArray,solver.blocks)
    solver.Yb.callCPU(solver.sharedArray,solver.edgeblocks)
    #Add to global time step after operations
    solver.Up.gts+=solver.Up.maxPyramidSteps
    solver.Yb.gts+=solver.Yb.maxPyramidSteps
    solver.Yb.start += solver.Yb.maxPyramidSteps #Change starting location for UpPrism
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
        solver.Yb.gfunction = solver.gpu.get_function("YBT") #Change to new ybridge function

def UpPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Start GPU Operations
    if solver.gpuBool:
        localGPUArray = getLocalExtendedArray(solver.sharedArray,numpy.zeros(solver.Up.shape),solver.gpuBlock,solver.Up.blocksize)
        cuda.memcpy_htod(solver.GPUArray,localGPUArray)  
        solver.Xb.callGPU(solver.GPUArray)
        solver.Oct.callGPU(solver.GPUArray)
        solver.Yb.callGPU(solver.GPUArray)    
    #Do CPU  Operations
    solver.Xb.callCPU(solver.sharedArray,solver.blocks)
    solver.Oct.callCPU(solver.sharedArray,solver.edgeblocks)
    solver.Yb.callCPU(solver.sharedArray,solver.blocks)  
    #Add to global time step after operations
    solver.Xb.gts+=solver.maxPyramidSteps
    solver.Oct.gts+=solver.maxPyramidSteps
    solver.Yb.gts+=solver.maxPyramidSteps
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
        solver.Yb.gfunction = solver.gpu.get_function("YBT") #Change to new ybridge function


def LastPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Start GPU Operations
    if solver.gpuBool:
        localGPUArray = getLocalExtendedArray(solver.sharedArray,numpy.zeros(solver.Up.shape),solver.gpuBlock,solver.Up.blocksize)
        cuda.memcpy_htod(solver.GPUArray,localGPUArray)  
        solver.Xb.callGPU(solver.GPUArray)
        solver.Down.callGPU(solver.GPUArray)
    #Do CPU Operations
    solver.Xb.callCPU(solver.sharedArray,solver.blocks)
    solver.Down.callCPU(solver.sharedArray,solver.edgeblocks)
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
        solver.Yb.gfunction = solver.gpu.get_function("YBT") #Change to new ybridge function


def firstForward(solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        buff = numpy.copy(solver.sharedArray[:,:,-solver.splitx:,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[1],source=solver.neighbors[0])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,solver.splitx:,:] = solver.sharedArray[:,:,:-solver.splitx,:] #Shift solver.sharedArray data forward by solver.splitx
        solver.sharedArray[:,:,:solver.splitx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier()


def sendForward(solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        cwt = decomp.swept_write(cwt,solver.sharedArray,hdf5_data,gsc,gts,TSO,maxPyramidSteps)
        buff = numpy.copy(solver.sharedArray[:,:,-solver.splitx:,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[1],source=solver.neighbors[0])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,solver.splitx:,:] = solver.sharedArray[:,:,:-solver.splitx,:] #Shift solver.sharedArray data forward by solver.splitx
        solver.sharedArray[:,:,:solver.splitx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier()
    return cwt

def sendBackward(solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        buff = numpy.copy(solver.sharedArray[:,:,:solver.splitx,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[0],source=solver.neighbors[1])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,:-solver.splitx,:] = solver.sharedArray[:,:,solver.splitx:,:] #Shift solver.sharedArray backward data by solver.splitx
        solver.sharedArray[:,:,-solver.splitx:,:] = buffer[:,:,:,:]
        cwt = decomp.swept_write(cwt,solver.sharedArray,hdf5_data,gsc,gts,TSO,MPSS)
    solver.nodeComm.Barrier()
    return cwt

def getLocalExtendedArray(sharedArray,arr,blocks,blocksize):
    """Use this function to copy the shared array to the local array and extend each end by one block."""
    it,iv,ix,iy = blocks
    arr[:,:,:,blocksize[0]:-blocksize[0]] = sharedArray[:,iv,ix,iy]
    arr[:,:,:,0:blocksize[0]] = sharedArray[:,iv,ix,-blocksize[0]:]
    arr[:,:,:,-blocksize[0]:] = sharedArray[:,iv,ix,:blocksize[0]]
    return arr

    