#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
#------------------------------Decomp Functions----------------------------------
import numpy
import pysweep.core.io as io
#Try to import pycuda
try:
    import pycuda.driver as cuda
except Exception as e:
    print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")
    
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
        fillLocalExtendedArray(solver)
        cuda.memcpy_htod(solver.GPUArray,solver.localGPUArray)
        solver.Up.callGPU(solver.GPUArray,solver.globalTimeStep)
        solver.Yb.callGPU(solver.GPUArray,solver.globalTimeStep)
    #Do CPU Operations
    solver.Up.callCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    solver.nodeComm.Barrier() #need barrier to make sure all data is in place for next step
    solver.Yb.callCPU(solver.sharedArray,solver.edgeblocks,solver.globalTimeStep)
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(solver.localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=solver.localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
        solver.Yb.setSweptStep(solver.maxPyramidSize)
    #Add to global time step after operations
    solver.globalTimeStep+=solver.maxPyramidSize
    solver.Yb.start += solver.maxPyramidSize #Change starting location for UpPrism
    

def UpPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    
    #Start GPU Operations
    if solver.gpuBool:
        fillLocalExtendedArray(solver)
        cuda.memcpy_htod(solver.GPUArray,solver.localGPUArray)  
        solver.Xb.callGPU(solver.GPUArray,solver.globalTimeStep)
        solver.Oct.callGPU(solver.GPUArray,solver.globalTimeStep)
        solver.Yb.callGPU(solver.GPUArray,solver.globalTimeStep)
    #Do CPU  Operations
    solver.Xb.callCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    solver.nodeComm.Barrier() #need barrier to make sure all data is in place for next step
    solver.Oct.callCPU(solver.sharedArray,solver.edgeblocks,solver.globalTimeStep)
    solver.nodeComm.Barrier() #need barrier to make sure all data is in place for next step
    solver.Yb.callCPU(solver.sharedArray,solver.blocks,solver.globalTimeStep)
    #Cleanup GPU Operations
    if solver.gpuBool:
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(solver.localGPUArray,solver.GPUArray)
        solver.sharedArray[solver.gpuBlock]=solver.localGPUArray[:,:,:,solver.blocksize[0]:-solver.blocksize[0]]
    #Add to global time step after operations
    solver.globalTimeStep+=solver.maxPyramidSize


def LastPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Start GPU Operations
    if solver.gpuBool:
        fillLocalExtendedArray(solver)
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

def firstForward(solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        buff = numpy.copy(solver.sharedArray[:,:,-solver.splitx:,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[1],source=solver.neighbors[0])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,solver.splitx:,:] = solver.sharedArray[:,:,:-solver.splitx,:] #Shift solver.sharedArray data forward by solver.splitx
        solver.sharedArray[:,:,:solver.splitx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier()


def sendForward(cwt,solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        cwt = io.sweptWrite(cwt,solver)
        buff = numpy.copy(solver.sharedArray[:,:,-solver.splitx:,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[1],source=solver.neighbors[0])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,solver.splitx:,:] = solver.sharedArray[:,:,:-solver.splitx,:] #Shift solver.sharedArray data forward by solver.splitx
        solver.sharedArray[:,:,:solver.splitx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier()
    return cwt

def sendBackward(cwt,solver):
    """Use this function to communicate data between nodes"""
    if solver.nodeMasterBool:
        buff = numpy.copy(solver.sharedArray[:,:,:solver.splitx,:])
        buffer = solver.clusterComm.sendrecv(sendobj=buff,dest=solver.neighbors[0],source=solver.neighbors[1])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,:-solver.splitx,:] = solver.sharedArray[:,:,solver.splitx:,:] #Shift solver.sharedArray backward data by solver.splitx
        solver.sharedArray[:,:,-solver.splitx:,:] = buffer[:,:,:,:]
        cwt = io.sweptWrite(cwt,solver)
    solver.nodeComm.Barrier()
    return cwt

def fillLocalExtendedArray(solver):
    """Use this function to copy the shared array to the local array and extend each end by one block."""
    it,iv,ix,iy = solver.gpuBlock
    adj = solver.blocksize[0]
    solver.localGPUArray[:,:,:,adj:-adj] = solver.sharedArray[:,iv,ix,iy]
    solver.localGPUArray[:,:,:,0:adj] = solver.sharedArray[:,iv,ix,-adj:]
    solver.localGPUArray[:,:,:,-adj:] = solver.sharedArray[:,iv,ix,:adj]

def fillLocalExtendedArrayStandard(sharedArray,arr,blocks,adjustment):
    """Use this function to copy the shared array to the local array and extend each end by one block."""
    it,iv,ix,iy = blocks
    arr[:,:,adjustment:-adjustment,adjustment:-adjustment] = sharedArray[:,iv,ix,iy]
    arr[:,:,adjustment:-adjustment,0:adjustment] = sharedArray[:,iv,ix,-adjustment:]
    arr[:,:,adjustment:-adjustment,-adjustment:] = sharedArray[:,iv,ix,:adjustment]
    arr[:,:,0:adjustment,adjustment:-adjustment] = sharedArray[:,iv,-adjustment:,iy]
    arr[:,:,-adjustment:,adjustment:-adjustment] = sharedArray[:,iv,:adjustment,iy]


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
        fillLocalExtendedArrayStandard(solver.sharedArray,solver.localGPUArray,solver.gpuBlock,solver.operating)
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

def sendEdges(solver):
    """Use this function to communicate data between nodes"""
    ops = solver.operating
    if solver.nodeMasterBool:
        solver.sharedArray[:,:,ops:-ops,:ops] = solver.sharedArray[:,:,ops:-ops,-2*ops-1:-ops-1]
        solver.sharedArray[:,:,ops:-ops,-ops:] = solver.sharedArray[:,:,ops:-ops,1:ops+1]
        bufffor = numpy.copy(solver.sharedArray[:,:,-2*ops-1:-ops-1,:])
        buffback = numpy.copy(solver.sharedArray[:,:,ops+1:2*ops+1,:])
        bufferf = solver.clusterComm.sendrecv(sendobj=bufffor,dest=solver.neighbors[1],source=solver.neighbors[0])
        bufferb = solver.clusterComm.sendrecv(sendobj=buffback,dest=solver.neighbors[0],source=solver.neighbors[1])
        solver.clusterComm.Barrier()
        solver.sharedArray[:,:,:ops,:] = bufferf[:,:,:,:]
        solver.sharedArray[:,:,-ops:,:] = bufferb[:,:,:,:]
    solver.nodeComm.Barrier()

