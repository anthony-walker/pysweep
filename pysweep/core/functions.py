#Programmer: Anthony Walker
#This file contains all of the necessary functions for implementing the swept rule.
#------------------------------Decomp Functions----------------------------------
import numpy


try:
    import pycuda.driver as cuda
except Exception as e:
    # print(str(e)+": Importing pycuda failed, execution will continue but is most likely to fail unless the affinity is 0.")
    pass


def Decomposition(GRB,OPS,sarr,garr,blocks,mpi_pool,DecompObj):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    #Splitting between cpu and gpu
    if GRB:
        DecompObj(garr)
        sarr[DecompObj.gwrite]=garr[:,:,OPS:-OPS,OPS:-OPS]
    else:   #CPUs do this
        mpi_pool.map(DecompObj,blocks)
        #Copy result to MPI shared process array
        sarr[DecompObj.cwrite] = sgs.CPUArray[:,:,OPS:-OPS,OPS:-OPS]
    DecompObj.gts+=1 #Update global time step

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

    if GRB:
        garr[:,:,:,:] = sarr[DecompObj.gread]
    else:
        sgs.CPUArray[:,:,:,:] = sarr[DecompObj.cread]


class DecompCPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args,cwrite=None,cread=None):
        """Use this function to create the initial function object"""
        self.gts,self.set,self.start = args
        self.cwrite=cwrite
        self.cread=cread

    def __call__(self,block):
        """Use this function to build the Up Pyramid."""
        sgs.CPUArray[block] = cpu.step(sgs.CPUArray[block],self.set,self.start,self.gts)

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (DecompCPU,(self.gts,self.set,self.start))

class DecompGPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args):
        """Use this function to create the initial function object"""
        self.gts,self.gfunction,self.grid,self.blocksize,self.gread,self.gwrite,garr = args
        self.arr_gpu = cuda.mem_alloc(garr.nbytes)


    def __call__(self,garr):
        """Use this function to build the Up Pyramid."""
        cuda.memcpy_htod(self.arr_gpu,garr)
        self.gfunction(self.arr_gpu,numpy.int32(self.gts),grid=self.grid, block=self.blocksize)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,self.arr_gpu)
        return garr

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (DecompGPU,(self.gts,self.set))

#------------------------------Swept Functions----------------------------------

def getLocalExtendedArray(sarr,arr,blocks,blocksize):
    """Use this function to copy the shared array to the local array and extend each end by one block."""
    iv,ix,iy = blocks
    arr[:,:,:,blocksize[0]:-blocksize[0]] = sarr[:,iv,ix,iy]
    arr[:,:,:,0:blocksize[0]] = sarr[:,iv,ix,-blocksize[0]:]
    arr[:,:,:,-blocksize[0]:] = sarr[:,iv,ix,:blocksize[0]]
    return arr


def FirstPrism(solver):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """

    #Splitting between cpu and gpu
    if solver.gpuBool:
        localGPUArray = getLocalExtendedArray(solver.sharedArray,numpy.zeros(solver.Up.shape),solver.gpuBlock,solver.Up.blocksize)
        cuda.memcpy_htod(solver.GPUArray,localGPUArray)
        solver.Up.callGPU(solver.GPUArray)
        solver.Yb.callGPU(solver.GPUArray)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(localGPUArray,solver.GPUArray)
        # solver.sharedArray[solver.blocks]=localGPUArray[:,:,:,solver.Up.blocksize[0]:-solver.Up.blocksize[0]]
        # solver.Up.gts+=solver.Up.maxPyramidSteps
        # solver.Yb.gts+=solver.Yb.maxPyramidSteps
        # solver.Yb.gfunction = solver.gpu.get_function("YBT") #Change to new ybridge function
    # else:   #CPUs do this
    #     solver.mpiPool.map(solver.Up,solver.blocks[0])
    #     solver.mpiPool.map(solver.Yb,solver.blocks[1])
    #     solver.Up.gts+=solver.Up.maxPyramidSteps
    #     solver.Yb.gts+=solver.Yb.maxPyramidSteps
    #     solver.Yb.start += solver.Yb.maxPyramidSteps #Change starting location for UpPrism
    #     solver.sharedArray[solver.totalCPUBlock] = sgs.CPUArray[:,:,:,:]

def UpPrism(*args):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    GRB,Xb,Yb,Oct,mpiPool,blocks,sarr,garr,totalCPUBlock = args
    #Splitting between cpu and gpu
    if GRB:
        lgarr = decomp.getLocalExtendedArray(sarr,numpy.zeros(Oct.shape),blocks,Oct.blocksize)
        cuda.memcpy_htod(garr,lgarr)
        Xb(garr)
        Oct(garr)
        Yb(garr)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(lgarr,garr)
        sarr[blocks]=lgarr[:,:,:,Oct.blocksize[0]:-Oct.blocksize[0]]
    else:   #CPUs do this
        mpiPool.map(Xb,blocks[0])
        mpiPool.map(Oct,blocks[1])
        mpiPool.map(Yb,blocks[0])
        sarr[totalCPUBlock] = sgs.CPUArray[:,:,:,:]
    Xb.gts+=Xb.maxPyramidSteps
    Oct.gts+=Oct.maxPyramidSteps
    Yb.gts+=Yb.maxPyramidSteps

def LastPrism(*args):
    """
    This is the starting pyramid for the 2D heterogeneous swept rule cpu portion.
    arr-the array that will be solved (t,v,x,y)
    fcn - the function that solves the problem in question
    OPS -  the number of atomic operations
    """
    GRB,Xb,Down,mpiPool,blocks,sarr,garr,totalCPUBlock = args
    #Splitting between cpu and gpu
    if GRB:
        lgarr = decomp.getLocalExtendedArray(sarr,numpy.zeros(Down.shape),blocks,Down.blocksize)
        cuda.memcpy_htod(garr,lgarr)
        Xb(garr)
        Down(garr)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(lgarr,garr)
        sarr[blocks]=lgarr[:,:,:,Down.blocksize[0]:-Down.blocksize[0]]
    else:   #CPUs do this
        mpiPool.map(Xb,blocks[0])
        mpiPool.map(Down,blocks[1])
        sarr[totalCPUBlock] = sgs.CPUArray[:,:,:,:]

def first_forward(NMB,GRB,nodeComm,cluster_comm,comranks,sarr,spx,totalCPUBlock):
    """Use this function to communicate data between nodes"""
    if NMB:
        buff = numpy.copy(sarr[:,:,-spx:,:])
        buffer = cluster_comm.sendrecv(sendobj=buff,dest=comranks[1],source=comranks[0])
        cluster_comm.Barrier()
        sarr[:,:,spx:,:] = sarr[:,:,:-spx,:] #Shift sarr data forward by spx
        sarr[:,:,:spx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier()
    if not GRB:
        sgs.CPUArray[:,:,:,:] = sarr[totalCPUBlock]

def send_forward(cwt,sarr,hdf5_data,gsc,NMB,GRB,nodeComm,cluster_comm,comranks,spx,gts,TSO,maxPyramidSteps,totalCPUBlock):
    """Use this function to communicate data between nodes"""
    if NMB:
        cwt = decomp.swept_write(cwt,sarr,hdf5_data,gsc,gts,TSO,maxPyramidSteps)
        buff = numpy.copy(sarr[:,:,-spx:,:])
        buffer = cluster_comm.sendrecv(sendobj=buff,dest=comranks[1],source=comranks[0])
        cluster_comm.Barrier()
        sarr[:,:,spx:,:] = sarr[:,:,:-spx,:] #Shift sarr data forward by spx
        sarr[:,:,:spx,:] = buffer[:,:,:,:]
    solver.nodeComm.Barrier()
    if not GRB:
        sgs.CPUArray[:,:,:,:] = sarr[totalCPUBlock]
    return cwt

def send_backward(cwt,sarr,hdf5_data,gsc,NMB,GRB,nodeComm,cluster_comm,comranks,spx,gts,TSO,MPSS,totalCPUBlock):
    """Use this function to communicate data between nodes"""
    if NMB:
        buff = numpy.copy(sarr[:,:,:spx,:])
        buffer = cluster_comm.sendrecv(sendobj=buff,dest=comranks[0],source=comranks[1])
        cluster_comm.Barrier()
        sarr[:,:,:-spx,:] = sarr[:,:,spx:,:] #Shift sarr backward data by spx
        sarr[:,:,-spx:,:] = buffer[:,:,:,:]
        cwt = decomp.swept_write(cwt,sarr,hdf5_data,gsc,gts,TSO,MPSS)
    solver.nodeComm.Barrier()
    if not GRB:
        sgs.CPUArray[:,:,:,:] = sarr[totalCPUBlock]
    return cwt


class Geometry(object):
    """Use this class to represent different phases in the swept process."""
    def __init__(self, gts=0):
        self.gts = gts

    def initializeCPU(self,*args):
        """Use this function to initialize CPU arguments."""
        self.sets,self.tso,self.maxPyramidSteps,self.start = args

    def initializeGPU(self,*args):
        """Use this function to initialize GPU arguments."""
        self.gfunction,self.blocksize,self.grid,self.maxPyramidSteps,self.shape = args

    def callCPU(self,block):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        ct = self.gts
        for ts,swept_set in enumerate(self.sets,start=self.start):
            #Calculating Step
            CPUArray[block] = sgs.cpu.step(sgs.CPUArray[block],swept_set,ts,ct)
            ct+=1
        return block

    def callGPU(self,garr):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        self.gfunction(garr,numpy.int32(self.gts),grid=self.grid, block=self.blocksize)

    