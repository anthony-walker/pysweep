import sys, os, yaml, numpy, h5py, warnings, subprocess, traceback,  mpi4py.MPI as MPI
import pysweep.core.GPUtil as GPUtil
import pysweep.core.io as io
import pysweep.core.process as process
import pysweep.core.engine as engine
import pysweep.core.functions as functions

class Solver(object):
    """docstring for Solver."""

    def __init__(self, initialConditions, yamlFileName=None,sendWarning=True):
        super(Solver, self).__init__()
        self.initialConditions = initialConditions
        self.arrayshape = numpy.shape(initialConditions)
        self.corepath = os.path.dirname(os.path.abspath(__file__))
        self.libpath = os.path.join(self.corepath,"lib")
        self.kernelpath = os.path.join(self.corepath,"kernels")

        if yamlFileName is not None:
            self.yamlFileName = yamlFileName
            #Managing inputs
            self.yamlManager()
            self.initialConditions = self.initialConditions.astype(self.dtype)

            #Creating time step data
            self.createTimeSteps()
            #set up MPI
            process.setupMPI(self)
            io.verbosePrint(self,"Setting up MPI...\n")

            #Creating simulatneous input and output file
            # io.verbosePrint(self,'Creating output file...\n')
            # self.createOutputFile()

        #     #Setting egine
        #     self.engine = "swept" if self.simulation else "standard"
        #     self.engineGPU = os.path.join(self.kernelpath,self.engine+".cu")
        #     self.engineCPU = os.path.join(self.corepath,self.engine+".py")
        #     io.verbosePrint(self,"Setting engine: {}\n".format(self.engine))
        #
        #     #number of processes
        #     io.verbosePrint(self,"Finding number of processes...\n")
        #     self.numberOfProcesses()
        #
        #     #Compile code
        #     self.cudaCompiler()
        #
        #     #Final statement
        #     io.verbosePrint(self,self)
        #     io.verbosePrint(self,"Ready to run simulation.")
        else:
            if sendWarning:
                warnings.warn('yaml not specified, requires manual input.')

    def __call__(self):
        """Use this function to spawn processes."""
        io.verbosePrint(self,'Running simulation...\n')
        #Starting timer
        start = time.time()
        #Setting global variables
        sgs.init_globals()
        #Local Constants
        ZERO = 0
        QUARTER = 0.25
        HALF = 0.5
        ONE = 1
        TWO = 2


    def __str__(self):
        """Use this function to print the object."""
        returnString = "\nPysweep simulation properties:\n"
        returnString += "\tsimulation type: {}\n".format("swept" if self.simulation else "standard")

        returnString+="\tgpu source: {}\n".format(self.source['gpu'])
        returnString+="\tcpu source: {}\n".format(self.source['cpu'])
        returnString+="\toutput file: {}\n".format(self.output)
        returnString+="\tverbose: {}\n".format(self.verbose)
        returnString+="\tArray shape: {}\n".format(self.arrayshape)
        returnString+="\tdtype: {}\n".format(self.dtype)
        returnString+="\tshare: {}\n".format(self.share)
        returnString+="\tblocksize: {}\n".format(self.blocksize[0])
        returnString+="\tstencil size: {}\n".format(int(self.operating*2+1))
        returnString+="\tintermediate time steps: {}\n".format(self.intermediate)
        returnString+="\ttime data (t0,tf,dt): ({},{},{})\n".format(*self.timeTuple)
        if self.globals:
            returnString+="\tglobals: ["
            for glob in self.globals:
                returnString+=str(glob)+","
            returnString=returnString[:-1]
            returnString+="]\n"
        if self.exid:
            returnString+="\texcluded gpus: ["
            for exid in self.exid:
                returnString+=str(exid)+","
            returnString=returnString[:-1]
            returnString+="]\n"
        return returnString

    def createTimeSteps(self):
        """Use this function to create timestep data."""
        self.time_steps = int((self.timeTuple[1]-self.timeTuple[0])/self.timeTuple[2])  #Number of time steps
        if self.simulation:
            self.maxPyramidSize = int(self.blocksize[0]/(2*self.operating)-1) #This will need to be adjusted for differing x and y block lengths
            self.maxGlobalSweptStep = int(self.intermediate*(self.time_steps-self.maxPyramidSize)/(self.maxPyramidSize)+1)  #Global swept step  #THIS ASSUMES THAT time_steps > MOSS
            self.time_steps = int(self.maxPyramidSize*(self.maxGlobalSweptStep+1)/self.intermediate+1) #Number of time
        self.arrayshape = (self.time_steps,)+self.timeTuple


    def createOutputFile(self):
        """Use this function to create output file which will act as the input file as well."""
        #Create input file
        hdf5_file = h5py.File(self.output, 'w', driver='mpio', comm=self.comm)
        hdf5_file.create_dataset("blocksize",(1,),data=(self.blocksize,))
        hdf5_file.create_dataset("share",(1,),data=(self.share,))
        hdf5_file.create_dataset("exid",(len(self.exid),),data=self.exid)
        hdf5_file.create_dataset("globals",(len(self.globals),),data=self.globals)
        hdf5_file.create_dataset("clocktime",(1,),data=0.0)
        hdf5_data_set = hdf5_file.create_dataset("data",(self.time_steps,)+self.arrayshape,dtype=self.dtype)

        if self.rank == self.master_rank:
            hdf5_data_set[0,:,:,:] = self.initialConditions[:,:,:]
        hdf5_file.close()
        self.comm.Barrier()

    def yamlManager(self):
        """Use this function to manage and assign all specified yaml variables."""
        def yamlGet(key,default):
            """Use this function to get arguments from yaml file."""
            return self.yamlFile[key] if key in self.yamlFile else default
        document = open(self.yamlFileName,'r')
        self.yamlFile = yaml.load(document,Loader=yaml.FullLoader)
        #Setting simulation type - true for swept
        self.simulation = yamlGet('swept',True)
        #Setting verbose flag
        self.verbose = yamlGet('verbose',False)
        #setting intermediate steps
        self.intermediate = yamlGet('intermediate_steps',1)
        #setting operating points
        self.operating = yamlGet('operating_points',1)
        #setting blocksize
        self.blocksize = yamlGet('blocksize',32)
        assert self.blocksize%(2*self.operating)==0, "Invalid blocksize, blocksize must satisfy BS = 2*ops*k and the architectural limit where k is any integer factor."
        self.blocksize = (self.blocksize,self.blocksize,1)
        #setting share
        self.share = yamlGet('share',0.9)
        #setting exid
        self.exid = yamlGet('exid',list())
        #setting time variables
        self.time = yamlGet('time',{'t0': 0, 'tf': 1, 'dt': 0.1})
        #setting globals
        self.globals = yamlGet('globals',list())
        #setting gpu source
        self.source = yamlGet('source',{'cpu': 0, 'gpu': 0})
        try:
            self.source['gpu'] = os.path.abspath(self.source["gpu"])
            self.source['cpu'] = os.path.abspath(self.source["cpu"])
        except Exception as e:
            pass
        #Setting output file
        self.output = yamlGet('filename','output.hdf5')
        #Settings datatype
        self.dtype = numpy.dtype(yamlGet('dtype','float64'))
        #Time tuple
        self.timeTuple = (self.time['t0'],self.time['tf'],self.time['dt'])

    def mpiSetup(self):
        """Use this function to create MPI variables"""
        #-------------MPI Set up----------------------------#
        self.comm = MPI.COMM_WORLD
        self.processor = MPI.Get_processor_name()
        self.proc_mult = 1
        self.rank = self.comm.Get_rank()  #current rank
        self.master_rank = 0

    def cudaCompiler(self,libname=None,recompile=False):
        """Use this function to create compiled lib."""
        if self.rank == self.master_rank:
            self.libname = libname
            if libname is None or recompile:
                io.verbosePrint(self,'Attempting to create shared library...')
                if self.libname is None:
                    basename = os.path.basename(self.source['gpu']).split(".")[0]
                    self.libname = os.path.join(self.libpath,"lib{}.so".format(self.engine+basename))
                nvccStatement = '''nvcc -Xcompiler -fPIC -shared -DPYSWEEP_GPU_SOURCE=\'\"{}\"\' -o {} {}'''.format(self.source['gpu'],self.libname,self.engineGPU)
                os.system(nvccStatement)
            else:
                warnings.warn('Code was not recompiled, set recompile=True to force recompile')
                if not os.path.exists(self.libname):
                    raise UserWarning('{} could not be located'.format(self.libname))
