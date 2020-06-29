
import pysweep,numpy,sys,os,h5py,yaml,time

path = os.path.dirname(os.path.abspath(__file__))

def getKeyArg(currKwargs,key,standard):
    """Get a keyword argument or return the standard value."""
    if key in currKwargs.keys():
        return currKwargs[key]
    else:
        return standard

def configurations(func):
    """Use this function as a decorator for other tests.
    It's purpose is to contain a set of test conditions and wrap tests with those conditions.
    exid, share, globals, cpu, gpu, operating_points, and intermediate_steps should be set in test
    """
    def testConfigurations(*args,**kwargs):
        ct = getKeyArg(kwargs,"ct",352)
        filename = getKeyArg(kwargs,"filename","exampleConditions.hdf5") #IC file name
        nsizes = getKeyArg(kwargs,"nsizes",5) #number of array sizes to find
        shares = numpy.arange(0,1,0.05) #Shares for GPU
        sims = [True,False] #different simulations
        blocksizes = [8, 12, 16, 24, 32] #blocksizes with most options
        #Getting first n compatible array sizes
        sizes = []
        while len(sizes)<nsizes:
            while not numpy.all([ct%blk==0 for blk in blocksizes]):
                ct+=32
            sizes.append(ct)
            ct+=32
        #Creat solver object
        solver = pysweep.Solver(sendWarning=False)
        solver.dtypeStr = 'float64'
        solver.dtype = numpy.dtype(solver.dtypeStr)
        solver.verbose=False
        for sim in sims:
            for ns in sizes:
                for bs in blocksizes:
                    for sh in shares:
                        solver.moments = [time.time(),]
                        solver.share = sh
                        solver.simulation = sim
                        solver.blocksize = (bs,bs,1)
                        func(solver,ns)
                        # try:
                        #     func(solver,ns)
                        # except Exception as e:
                        #     pass #Add log file updated for failure
    return testConfigurations
    
    

@configurations
def testExample(*args,**kwargs):
    solver,ns = args
    filename = pysweep.equations.example.createInitialConditions(1,ns,ns)
    solver.assignInitialConditions(filename)
    solver.operating = 1
    solver.intermediate = 1
    solver.cpu = os.path.join(os.path.dirname(path),"equations")
    solver.gpu = os.path.join(solver.cpu,"example.cu")
    solver.gpuStr = os.path.join(solver.cpu,"example.cu")
    solver.cpu = os.path.join(solver.cpu,"example.py")
    solver.cpuStr = os.path.join(solver.cpu,"example.py")
    solver.output = "exampleOut.hdf5"
    solver.globals = [0,1,0.1,0.1,0.1]
    solver.exid = []
    pysweep.core.io.loadCPUModule(solver)
    solver()
    print("Simulation time: {}".format(solver.moments[-1]-solver.moments[0]))
    # if solver.clusterMasterBool:
    #     with h5py.File(solver.output,"r") as f:
    #         data = f["data"]
    #         for i in range(1,solver.arrayShape[0]+1):
    #             try:
    #                 assert numpy.all(data[i-1,0,:,:]==i)
    #             except Exception as e:
    #                 print("Simulation failed on index: {}.".format(i))
    if solver.clusterMasterBool:
        input()
    solver.comm.Barrier()


def testGPUProcessDestruction():
    """Use this function to check that node processes will be destroyed if GPUs' are not available."""

def testHeat(share=0.5,npx=16,npy=16):
    filename = pysweep.equations.heat.createInitialConditions(npx,npy)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,"heat.yaml")
    testSolver = pysweep.Solver(filename,yfile)
    testSolver()

    # if testSolver.clusterMasterBool:
    #     with h5py.File(testSolver.output,"r") as f:
    #         data = f["data"]
    #         for i in range(1,testSolver.arrayShape[0]+1):
    #             try:
    #                 assert numpy.all(data[i-1,0,:,:]==i)
    #             except Exception as e:
    #                 print("Simulation failed on index: {}.".format(i))
    #                 print(data[i-1,0,:,:])


if __name__ == "__main__":
    pass
    # MPI.COMM_SELF.Spawn(sys.executable, args=["from pysweep.tests import testExample; testExample()"], maxprocs=2)
