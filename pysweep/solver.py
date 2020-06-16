import os, yaml, numpy, h5py, mpi4py.MPI as MPI

class Solver(object):
    """docstring for Solver."""

    def __init__(self, initialConditions, yamlFileName):
        super(Solver, self).__init__()
        self.initialConditions = initialConditions
        self.arrayshape = numpy.shape(initialConditions)
        self.yamlFileName = yamlFileName
        #Managing inputs
        self.yamlManager()
        self.initialConditions = self.initialConditions.astype(self.dtype)
        #set up MPI
        self.mpiSetup()
        self.verbosePrint("Setting up MPI...\n")

        #Creating simulatneous input and output file
        self.verbosePrint('Creating output file...\n')
        # self.createOutputFile()

        #Copying source code
        self.verbosePrint('Copying source code...\n')
        # self.sourceCopy()

        #Setting engine
        self.engine = "swept.py" if self.simulation else "standard.py"
        self.enginePath = os.path.join(os.path.dirname(os.path.abspath(__file__)),self.engine)
        self.verbosePrint("Setting engine: {}\n".format(self.engine))

        #number of processes
        self.verbosePrint("Finding number of processes...\n")
        # self.numberOfProcesses()

        self.verbosePrint("Ready to run simulation.")


    def __call__():
        """Use this function to spawn processes."""
        self.verbosePrint('Running simulation...\n')
        if self.rank == self.master_rank:
            MPI.COMM_SELF.Spawn(sys.executable,args=[self.enginePath],maxprocs=self.um_procs)
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

        returnString+="\ttime data (t0,tf,dt): ({},{},{})\n".format(self.time['t0'],self.time['tf'],self.time['dt'])
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

    def createOutputFile(self):
        """Use this function to create output file which will act as the input file as well."""
        #Create input file
        hdf5_file = h5py.File(self.output, 'w', driver='mpio', comm=self.comm)
        hdf5_file.create_dataset("data",self.arrayshape,data=arr0)
        hdf5_file.create_dataset("swargs",(len(swargs[:-2]),),data=swargs[:-2])
        hdf5_file.create_dataset("gargs",(len(gargs),),data=gargs[:])
        FN = [ord(ch) for ch in filename]
        DT = [ord(ch) for ch in dType]
        hdf5_file.create_dataset("exid",(len(self.exid),),data=self.exid)
        hdf5_file.create_dataset("filename",(len(FN),),data=FN)
        hdf5_file.create_dataset("dType",(len(DT),),data=DT)
        comm.Barrier()
        hdf5_file.close()

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
        self.intermediate = yamlGet('intermediate_steps',0)
        #setting operating points
        self.operating = yamlGet('operating_points',1)
        #setting gpu source
        self.source = yamlGet('source',[None,None])
        #Setting output file
        self.output = yamlGet('filename','output.hdf5')
        #Settings datatype
        self.dtype = numpy.dtype(yamlGet('dtype','float64'))

        if self.verbose:
            print(self)

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
        self.proc_mult = comm.bcast(self.proc_mult,root=self.master_rank)
        self.comm.Barrier()
        self.num_procs = int(self.proc_mult*self.comm.Get_size())
