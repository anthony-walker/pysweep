import sys, os, yaml, numpy, warnings, subprocess, traceback, time, h5py
import pysweep.core.GPUtil as GPUtil
import pysweep.core.io as io
import pysweep.core.process as process
import pysweep.core.functions as functions
import pysweep.core.block as block
from itertools import cycle

class Solver(object):
    """docstring for Solver."""
    def __init__(self, initialConditions, yamlFileName=None,sendWarning=True):
        super(Solver, self).__init__()
        self.moments = [time.time(),]
        process.setupCommunicators(self) #This function creates necessary variables for MPI to use
        self.assignInitialConditions(initialConditions)

        if yamlFileName is not None:
            self.yamlFileName = yamlFileName
            #Managing inputs
            io.yamlManager(self)
        else:
            if sendWarning:
                warnings.warn('yaml not specified, requires manual input.')

    def __call__(self,start=0,stop=-1,libname=None,recompile=False):
        """Use this function to spawn processes."""
        #Grabbing start of call time
        self.moments.append(time.time())

        #set up MPI
        io.verbosePrint(self,"Setting up processes...\n")
        process.setupProcesses(self)
        self.moments.append(time.time())

        #Creating time step data
        io.verbosePrint(self,'Creating time step data...\n')
        self.createTimeStepData()
        self.moments.append(time.time())
        
        #Creating simulatneous input and output file
        io.verbosePrint(self,'Creating output file...\n')
        io.createOutputFile(self)
        self.moments.append(time.time())

        # Creating shared array
        io.verbosePrint(self,'Creating shared memory arrays and process functions...\n')
        if self.simulation:
            block.sweptBlock(self)
        else:
            block.standardBlock(self)
        self.moments.append(time.time())
        #Cleaning up unneeded variables
        io.verbosePrint(self,'Cleaning up solver...\n')
        self.solverCleanUp()
        self.moments.append(time.time())
        #Running simulation
        io.verbosePrint(self,'Running simulation...')
        io.verbosePrint(self,io.getSolverPrint(self))
        if self.simulation:
            self.sweptSolve()
        else:
            self.standardSolve()
        self.moments.append(time.time())
        #Process cleanup
        io.verbosePrint(self,'Cleaning up processes...\n')
        process.cleanupProcesses(self,self.moments[start],self.moments[stop])
        io.verbosePrint(self,'Done in {} seconds...\n'.format(self.moments[-1]-self.moments[0]))

    def __str__(self):
        """Use this function to print the object."""
        return io.getSolverPrint(self)

    def assignInitialConditions(self,initialConditions):
        """Use this function to optionally assign initial conditions as an hdf5 file."""
        if type(initialConditions) == str:
            hf =  h5py.File(initialConditions,"r", driver='mpio', comm=self.comm)
            self.initialConditions = hf.get('data')
            self.arrayShape = hf.get('data').shape
            self.hf = hf
        else:
            self.initialConditions = initialConditions
            self.arrayShape = numpy.shape(initialConditions)

    def createTimeStepData(self):
        """Use this function to create timestep data."""
        self.timeSteps = int((self.globals[1]-self.globals[0])/self.globals[2])  #Number of time steps
        if self.simulation:
            self.splitx = self.blocksize[0]//2
            self.splity = self.blocksize[1]//2
            self.maxPyramidSize = self.blocksize[0]//(2*self.operating)-1 #This will need to be adjusted for differing x and y block lengths
            self.maxGlobalSweptStep = int(self.intermediate*(self.timeSteps-self.maxPyramidSize)/(self.maxPyramidSize)+1)  #Global swept step  #THIS ASSUMES THAT timeSteps > MOSS
            self.timeSteps = int(self.maxPyramidSize*(self.maxGlobalSweptStep+1)/self.intermediate+1) #Number of time
            self.maxOctSize = 2*self.maxPyramidSize
            self.sharedShape = (self.maxOctSize+self.intermediate,)+self.sharedShape
            self.arrayShape = (self.timeSteps,)+self.arrayShape
        else:
            self.sharedShape = (self.intermediate+1,)+self.sharedShape
            self.arrayShape = (self.timeSteps+1,)+self.arrayShape

    def solverCleanUp(self):
        """Use this function to remove unvariables not needed for computation."""
        # if self.clusterMasterBool:
        del self.gpuRank
        del self.initialConditions
        del self.yamlFile
        del self.yamlFileName
        del self.rank
        #Close ICS if it is a file.
        try:
            self.hf.close()
        except Exception as e:
            pass

    def debugSimulations(self,arr=None):
        """Use this function to help debug"""
        if self.clusterMasterBool: 
            ci = 0
            while ci != -1:
                if arr is None:
                    print(self.sharedArray[ci,0,:,:])
                else:
                    print(arr[ci,0,:,:])
                ci = int(input())
        self.comm.Barrier()

    def sweptSolve(self):
        """Use this function to begin the simulation."""
        # -------------------------------SWEPT RULE---------------------------------------------#
        #setting global time step to zero
        self.globalTimeStep=0
        # -------------------------------FIRST PRISM AND COMMUNICATION-------------------------------------------#
        functions.FirstPrism(self)
        functions.firstForward(self)
        #Loop variables
        cwt = 1 #Current write time
        del self.Up #Deleting Up object after FirstPrism
        #-------------------------------SWEPT LOOP--------------------------------------------#
        step = cycle([functions.sendBackward,functions.sendForward])
        for i in range(self.maxGlobalSweptStep):
            functions.UpPrism(self)
            cwt = next(step)(cwt,self)
        #Do LastPrism Here then Write all of the remaining data
        functions.LastPrism(self)
        next(step)(cwt,self)

    def standardSolve(self):
        # -------------------------------Standard Decomposition---------------------------------------------#
        #setting global time step to zero
        self.globalTimeStep=0
        #Send Boundary points
        functions.sendEdges(self)
        cwt = 1
        for i in range(self.intermediate*self.timeSteps):
            functions.StandardFunction(self)
            cwt = io.standardWrite(cwt,self)
            #Communicate
            functions.sendEdges(self)
