import numpy

class Geometry(object):
    """Use this class to represent different phases in the swept process."""
    def __init__(self):
        super(Geometry, self).__init__()

    def initializeCPU(self,*args):
        """Use this function to initialize CPU arguments."""
        self.cpu,self.sets,start,cshape = args
        self.start = numpy.int32(start)
        self.CPUArray = numpy.zeros(cshape)

    def initializeGPU(self,*args):
        """Use this function to initialize GPU arguments."""
        self.gfunction,self.blocksize,self.grid = args

    def startAdd(self,value):
        """Use this to add to the existing start parameter."""
        self.start = numpy.int32(value+self.start)
        
    def setAdjustment(self,value):
        """Use this to set the adjustment for standard"""
        self.adj = value

    def callCPU(self,sharedArray,blocks,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        for block in blocks:
            ct = globalTimeStep
            self.CPUArray[:,:,:,:] = sharedArray[block]
            for ts,blockset in enumerate(self.sets,start=self.start):
                #Calculating Step
                self.cpu.step(self.CPUArray,blockset,ts,ct)
                ct+=1
            sharedArray[block] = self.CPUArray[:,:,:,:]

    def callGPU(self,GPUArray,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        self.gfunction(GPUArray,numpy.int32(globalTimeStep),self.start,grid=self.grid, block=self.blocksize)

    def callStandardGPU(self,GPUArray,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        self.gfunction(GPUArray,numpy.int32(globalTimeStep),grid=self.grid, block=self.blocksize)

    def callStandardCPU(self,sharedArray,blocks,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        for block in blocks:
            writeblock,readblock = block
            self.CPUArray[:,:,:,:] = sharedArray[readblock]
            self.cpu.step(self.CPUArray,self.sets,self.start,globalTimeStep)
            sharedArray[writeblock] = self.CPUArray[:,:,self.adj:-self.adj,self.adj:-self.adj]
