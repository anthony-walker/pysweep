import sys, os, yaml, numpy, h5py, warnings, subprocess, traceback,  mpi4py.MPI as MPI, pysweep.core.GPUtil as GPUtil

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
            os.environ["PYSWEEP_GPU_SOURCE"] = self.source['gpu']
            os.environ["PYSWEEP_CPU_SOURCE"] = self.source['cpu']

            #set up MPI
            self.mpiSetup()
            self.verbosePrint("Setting up MPI...\n")

            #Creating simulatneous input and output file
            self.verbosePrint('Creating output file...\n')
            self.createOutputFile()

            #Copying source code
            self.verbosePrint('Copying source code...\n')
            # self.sourceCopy()

            #Setting engine
            self.engine = "swept" if self.simulation else "standard"
            self.engineGPU = os.path.join(self.kernelpath,self.engine+".cu")
            self.engineCPU = os.path.join(self.corepath,self.engine+".py")
            self.verbosePrint("Setting engine: {}\n".format(self.engine))

            #number of processes
            self.verbosePrint("Finding number of processes...\n")
            self.numberOfProcesses()

            #Compile code
            self.cudaCompiler()

            #Final statement
            self.verbosePrint(self)
            self.verbosePrint("Ready to run simulation.")
        else:
            if sendWarning:
                warnings.warn('yaml not specified, requires manual input.')

    def __call__(self):
        """Use this function to spawn processes."""
        self.verbosePrint('Running simulation...\n')
        if self.rank == self.master_rank:
            print(self.engineCPU)
            MPI.COMM_SELF.Spawn(sys.executable,args=[self.engineCPU],maxprocs=self.num_procs)
        else:
            MPI.Finalize()
            exit(0)

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
        returnString+="\tblocksize: {}\n".format(self.blocksize)
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
            self.maxPyramidSize = int(self.blocksize/(2*self.operating)-1)
            self.maxGlobalSweptStep = int(self.intermediate*(self.time_steps-self.maxPyramidSize)/(self.maxPyramidSize)+1)  #Global swept step  #THIS ASSUMES THAT time_steps > MOSS
            self.time_steps = int(self.maxPyramidSize*(self.maxGlobalSweptStep+1)/self.intermediate+1) #Number of time

    def createOutputFile(self):
        """Use this function to create output file which will act as the input file as well."""
        #Create input file
        self.createTimeSteps()
        if self.rank == self.master_rank:
            hdf5_file = h5py.File(self.output, 'w', driver='mpio', comm=self.comm)
            hdf5_file.create_dataset("blocksize",(1,),data=(self.blocksize,))
            hdf5_file.create_dataset("share",(1,),data=(self.share,))
            hdf5_file.create_dataset("exid",(len(self.exid),),data=self.exid)
            hdf5_file.create_dataset("globals",(len(self.globals),),data=self.globals)
            hdf5_file.create_dataset("clocktime",(1,),data=0.0)
            hdf5_data_set = hdf5_file.create_dataset("data",(self.time_steps,)+self.arrayshape,dtype=self.dtype)
            hdf5_data_set[0,:,:,:] = self.initialConditions[:,:,:]
            hdf5_file.close()
        self.comm.Barrier()

    def verbosePrint(self,outString):
        """Use this function to print only when verbose option is specified."""
        if self.verbose and self.rank == self.master_rank:
            print(outString)

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
        #setting blocksize
        self.blocksize = yamlGet('blocksize',32)
        #setting share
        self.share = yamlGet('share',0.9)
        #setting exid
        self.exid = yamlGet('exid',list())
        #setting time variables
        self.time = yamlGet('time',{'t0': 0, 'tf': 1, 'dt': 0.1})
        #setting globals
        self.globals = yamlGet('globals',list())
        #setting intermediate steps
        self.intermediate = yamlGet('intermediate_steps',1)
        #setting operating points
        self.operating = yamlGet('operating_points',1)
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

    def sourceCopy(self):
        """Use this function to copy source code over."""
        #Copy GPU/CPU files
        cpath = os.path.join(os.path.dirname(__file__),'dcore')
        shutil.copyfile(swargs[5],os.path.join(cpath,'cpu.py'))
        shutil.copyfile(swargs[4],os.path.join(cpath,'gpu.h'))

    def numberOfProcesses(self):
        """Use this function to determine number of processes"""
        if self.share>0:
            gpu_rank = GPUtil.getAvailable(order = 'load',maxLoad=1,maxMemory=1,excludeID=self.exid,limit=1e8) #getting devices by load
            self.num_gpus = len(gpu_rank)
        else:
            gpu_rank = []
            self.num_gpus = 0
        node_gpus_list = self.comm.allgather(self.num_gpus)
        if self.rank == self.master_rank:
            #Multipler for Number of processes to add
            self.proc_mult += max(node_gpus_list)
        self.proc_mult = self.comm.bcast(self.proc_mult,root=self.master_rank)
        self.comm.Barrier()
        self.num_procs = int(self.proc_mult*self.comm.Get_size())

    def cudaCompiler(self,libname=None,recompile=False):
        """Use this function to create compiled lib."""
        if self.rank == self.master_rank:
            self.libname = libname
            if libname is None or recompile:
                self.verbosePrint('Attempting to create shared library...')
                if self.libname is None:
                    basename = os.path.basename(self.source['gpu']).split(".")[0]
                    self.libname = os.path.join(self.libpath,"lib{}.so".format(self.engine+basename))
                nvccStatement = '''nvcc -Xcompiler -fPIC -shared -DPYSWEEP_GPU_SOURCE=\'\"{}\"\' -o {} {}'''.format(self.source['gpu'],self.libname,self.engineGPU)
                os.system(nvccStatement)
            else:
                warnings.warn('Code was not recompiled, set recompile=True to force recompile')
                if not os.path.exists(self.libname):
                    raise UserWarning('{} could not be located'.format(self.libname))

    @classmethod
    def manualSolver(cls,initialConditions,sim=True,verbose=True,blocksize=32,share=0.9,exid=[],time={'t0': 0, 'tf': 1, 'dt': 0.1},globals=[],intermediate=1,operating=1,source=[None,None],filename="output.hdf5",dtype=numpy.dtype('float64')):
        """Use this function to manually set solver arguments."""


        solverObj = cls(initialConditions,sendWarning=False)
        solverObj.simulation = sim
        solverObj.verbose = verbose
        solverObj.blocksize = blocksize
        solverObj.share = share
        solverObj.globals = globals
        solverObj.intermediate = intermediate
        solverObj.operating = operating
        solverObj.source = source
        solverObj.output = filename
        solverObj.dtype = dtype
        solverObj.initialConditions = initialConditions.astype(solverObj.dtype)
        solverObj.arrayshape = numpy.shape(initialConditions)
        #set up MPI
        solverObj.mpiSetup()
        solverObj.verbosePrint("Setting up MPI...\n")
        #Creating simulatneous input and output file
        solverObj.verbosePrint('Creating output file...\n')
        solverObj.createOutputFile()
        #Copying source code
        solverObj.verbosePrint('Copying source code...\n')
        # solverObj.sourceCopy()

        #Setting engine
        solverObj.engine = "swept.py" if solverObj.simulation else "standard.py"
        solverObj.enginePath = os.path.join(os.path.dirname(os.path.abspath(__file__)),solverObj.engine)
        solverObj.verbosePrint("Setting engine: {}\n".format(solverObj.engine))

        #number of processes
        solverObj.verbosePrint("Finding number of processes...\n")
        solverObj.numberOfProcesses()
        #Final statement
        solverObj.verbosePrint(solverObj)
        solverObj.verbosePrint("Ready to run simulation.")

        return solverObj
