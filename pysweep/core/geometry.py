import numpy

class Geometry(object):
    """Use this class to represent different phases in the swept process."""
    def __init__(self):
        super(Geometry, self).__init__()

    def initializeCPU(self,*args):
        """Use this function to initialize CPU arguments."""
        self.cpu,self.sets,self.tso,self.start = args
        self.sweptStep = numpy.int32(self.tso-1)

    def initializeGPU(self,*args):
        """Use this function to initialize GPU arguments."""
        self.gfunction,self.blocksize,self.grid,self.shape = args

    def setSweptStep(self,value):
        """Use this function to set the swept step."""
        self.sweptStep = numpy.int32(value)

    def callCPU(self,sharedArray,blocks,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        for block in blocks:
            ct = globalTimeStep
            CPUArray = numpy.copy(sharedArray[block])
            for ts,sweptSet in enumerate(self.sets,start=self.start):
                #Calculating Step
                CPUArray = self.cpu.step(CPUArray,sweptSet,ts,ct)
                ct+=1
            sharedArray[block] = CPUArray[:,:,:,:]

    def callGPU(self,GPUArray,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        self.gfunction(GPUArray,numpy.int32(globalTimeStep),self.sweptStep,grid=self.grid, block=self.blocksize)

class DecompCPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args,cwrite=None,cread=None):
        """Use this function to create the initial function object"""
        self.globalTimeStep,self.set,self.start = args
        self.cwrite=cwrite
        self.cread=cread

    def __call__(self,block):
        """Use this function to build the Up Pyramid."""
        sgs.CPUArray[block] = cpu.step(sgs.CPUArray[block],self.set,self.start,self.globalTimeStep)

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (DecompCPU,(self.globalTimeStep,self.set,self.start))

class DecompGPU(object):
    """This class computes the octahedron of the swept rule"""

    def __init__(self,*args):
        """Use this function to create the initial function object"""
        self.globalTimeStep,self.gfunction,self.grid,self.blocksize,self.gread,self.gwrite,garr = args
        self.arr_gpu = cuda.mem_alloc(garr.nbytes)


    def __call__(self,garr):
        """Use this function to build the Up Pyramid."""
        cuda.memcpy_htod(self.arr_gpu,garr)
        self.gfunction(self.arr_gpu,numpy.int32(self.globalTimeStep),grid=self.grid, block=self.blocksize)
        cuda.Context.synchronize()
        cuda.memcpy_dtoh(garr,self.arr_gpu)
        return garr

    def __reduce__(self):
        """Use this function to make the object pickleable"""
        return (DecompGPU,(self.globalTimeStep,self.set))
