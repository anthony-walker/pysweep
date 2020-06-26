import numpy

class Geometry(object):
    """Use this class to represent different phases in the swept process."""
    def __init__(self):
        super(Geometry, self).__init__()

    def initializeCPU(self,*args):
        """Use this function to initialize CPU arguments."""
        self.cpu,self.sets,self.start = args

    def initializeGPU(self,*args):
        """Use this function to initialize GPU arguments."""
        self.gfunction,self.blocksize,self.grid = args

    def setSweptStep(self,value):
        """Use this function to set the swept step."""
        self.sweptStep = numpy.int32(value)

    def setAdjustment(self,value):
        """Use this to set the adjustment for standard"""
        self.adj = value

    def callCPU(self,sharedArray,blocks,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        for block in blocks:
            ct = globalTimeStep
            CPUArray = numpy.copy(sharedArray[block])
            for ts,blockset in enumerate(self.sets,start=self.start):
                #Calculating Step
                CPUArray = self.cpu.step(CPUArray,blockset,ts,ct)
                ct+=1
            sharedArray[block] = CPUArray[:,:,:,:]

    def callGPU(self,GPUArray,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        self.gfunction(GPUArray,numpy.int32(globalTimeStep),grid=self.grid, block=self.blocksize)

    def callStandardGPU(self,GPUArray,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        self.gfunction(GPUArray,numpy.int32(globalTimeStep),grid=self.grid, block=self.blocksize)

    def callStandardCPU(self,sharedArray,blocks,globalTimeStep):
        """Use this function to build the Up Pyramid."""
        #UpPyramid of Swept Step
        for block in blocks:
            writeblock,readblock = block
            CPUArray = numpy.copy(sharedArray[readblock])
            #Calculating Step
            CPUArray = self.cpu.step(CPUArray,self.sets,self.start,globalTimeStep)
            sharedArray[writeblock] = CPUArray[:,:,self.adj:-self.adj,self.adj:-self.adj]