
import pysweep,numpy,sys,os,h5py,yaml,time
import matplotlib.pyplot as plt
path = os.path.dirname(os.path.abspath(__file__))

def writeOut(arr,prec="%.5f"):
        for row in arr:
            sys.stdout.write("[")
            for item in row:
                if numpy.isclose(item,0):
                    sys.stdout.write("\033[1;36m")
                else:
                    sys.stdout.write("\033[1;31m")
                sys.stdout.write((prec+", ")%item)
            sys.stdout.write("]\n")


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

def testSimpleOne(npx=384,npy=384):
    filename = pysweep.equations.example.createInitialConditions(1,npx,npy)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,"example.yaml")
    testSolver = pysweep.Solver(filename,yfile)
    testSolver()

    if testSolver.clusterMasterBool:
        with h5py.File(testSolver.output,"r") as f:
            data = f["data"]
            for i in range(testSolver.arrayShape[0]):
                try:
                    assert numpy.all(data[i,0,:,:]==i)
                except Exception as e:
                    failed = True
                    # print("Simulation failed on index: {}.".format(i))

def testSimpleTwo(npx=384,npy=384):
    filename = pysweep.equations.example.createInitialConditions(1,npx,npy)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,"example.yaml")
    testSolver = pysweep.Solver(filename,yfile)
    testSolver.intermediate = 2
    testSolver.globals[-1] = False
    testSolver()
    if testSolver.clusterMasterBool:
        failed = False
        with h5py.File(testSolver.output,"r") as f:
            data = f["data"]
            for i,value in enumerate(range(1,2*testSolver.arrayShape[0]-1,2),start=1):
                try:
                    assert numpy.all(data[i,0,:,:]==value)
                except Exception as e:
                    failed = True
                    print("Simulation failed on index: {}.".format(i))
                    print(data[i,0])
                    input()
            print("{}".format("Failed" if failed else "Success"))

def testChecker(npx=384,npy=384):
    filename = pysweep.equations.checker.createInitialConditions(1,npx,npy)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,"checker.yaml")
    testSolver = pysweep.Solver(filename,yfile)
    testSolver()
    if testSolver.clusterMasterBool:
        with h5py.File(testSolver.output,"r") as f:
            data = f["data"]
            nt,nv,nx,ny = numpy.shape(data)
            pattern1 = numpy.zeros((nx,ny))
            pattern2 = numpy.zeros((nx,ny))
            for i in range(0,nx,2):
                for j in range(0,ny,2):
                    pattern1[i,j]=1
                    pattern2[i+1,j]=1
            for i in range(1,nx,2):
                for j in range(1,ny,2):
                    pattern1[i,j]=1
                    pattern2[i-1,j]=1
            for i in range(0,testSolver.arrayShape[0]):
                try:
                    if i%2==0:
                        assert numpy.all(data[i,0,:,:]==pattern1)
                    else:
                        assert numpy.all(data[i,0,:,:]==pattern2)
                except Exception as e:
                    print("Simulation failed on index: {}.".format(i))
                    print(data[i,0,:,:])
                    input()

def testHeatForwardEuler(inputFile="heat.yaml",npx=384,npy=384):
    timeSteps = 100
    filename = pysweep.equations.heat.createInitialConditions(npx,npy,alpha=0.00016563)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,inputFile)
    testSolver = pysweep.Solver(filename,yfile)
    t0,tf,dt,dx,dy,alpha,scheme = testSolver.globals
    tf = timeSteps*dt
    testSolver.globals[1] = tf #reset tf for 500 steps
    testSolver()
    
    if testSolver.clusterMasterBool:
        steps = testSolver.arrayShape[0]
        stepRange = numpy.array_split(numpy.arange(0,steps),testSolver.comm.Get_size())
    else:
        stepRange = []
    stepRange = testSolver.comm.scatter(stepRange)
    error = []
    testSolver.comm.Barrier()
    with h5py.File(testSolver.output,"r",driver="mpio",comm=testSolver.comm) as f:
        data = f["data"]
        for i in stepRange:
            T,x,y = pysweep.equations.heat.analytical(npx,npy,t=dt*i,alpha=alpha)
            assert numpy.allclose(data[i,0],T[0])

def testHeatRungeKuttaTwo(inputFile="heat.yaml",npx=384,npy=384):
    timeSteps = 100
    filename = pysweep.equations.heat.createInitialConditions(npx,npy,alpha=0.00016563)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,inputFile)
    testSolver = pysweep.Solver(filename,yfile)
    t0,tf,dt,dx,dy,alpha,scheme = testSolver.globals
    tf = timeSteps*dt
    testSolver.globals[1] = tf #reset tf for 500 steps
    testSolver.globals[-1] = False #Set to RK2
    testSolver()
    
    if testSolver.clusterMasterBool:
        steps = testSolver.arrayShape[0]
        stepRange = numpy.array_split(numpy.arange(0,steps),testSolver.comm.Get_size())
    else:
        stepRange = []
    stepRange = testSolver.comm.scatter(stepRange)
    error = []
    testSolver.comm.Barrier()
    with h5py.File(testSolver.output,"r",driver="mpio",comm=testSolver.comm) as f:
        data = f["data"]
        for i in stepRange:
            T,x,y = pysweep.equations.heat.analytical(npx,npy,t=dt*i,alpha=alpha)
            try:
                assert numpy.allclose(data[i,0],T[0])
            except:
                print("Simulation failed at index {}".format(i))

    # error = testSolver.comm.allgather(error)
    # if testSolver.clusterMasterBool:
    #     finalError = []
    #     finalMean = []
    #     for eL in error:
    #         finalError.append(numpy.amax(eL))
    #         finalMean.append(numpy.mean(eL))
    
    #     print(numpy.amax(finalError))
    #     print(numpy.mean(finalMean))
    #     print(numpy.where(finalError==numpy.amax(finalError)))


def testCompareSolvers():
    """Use this function to compare cpu portion of solvers based on output hdf5 and error.
    This function is meant for only 1 process
    """
    #Adjustment of global variables
    d = 0.1 #Constant
    alpha = 1
    npx = npy = 384
    T,x,y = pysweep.equations.heat.analytical(npx,npy,0)
    dx = dy = float(x[1]-x[0])
    t0,tf = 0,0.001
    dt = float(d*dx**2/alpha)
    #Input files
    filename = pysweep.equations.heat.createInitialConditions(npx,npy,alpha=alpha)
    inputPath = os.path.join(path,"inputs")

    sweptSolver = pysweep.Solver(filename, os.path.join(inputPath,"heatSwept.yaml"))
    standardSolver = pysweep.Solver(filename, os.path.join(inputPath,"heatStandard.yaml"))

    #Set same globals
    standardSolver.globals =  [t0,tf,dt,dx,dx,alpha,True]
    sweptSolver.globals =  [t0,tf,dt,dx,dx,alpha,True]
    
    #execute solvers
    standardSolver()
    sweptSolver()
    
    #hdf5
    if standardSolver.clusterMasterBool:
        sweptHdf5 = h5py.File(sweptSolver.output,"r")
        standardHdf5 = h5py.File(standardSolver.output,"r")
        
        sweptData = sweptHdf5['data']
        standardData = standardHdf5['data']

        for i in range(standardSolver.timeSteps):
            T,x,y = pysweep.equations.heat.analytical(npx,npy,t=dt*i,alpha=alpha)
            assert numpy.allclose(sweptData[i,0],standardData[i,0])
        

def testOrderOfConvergence(ifile="heat.yaml"):
    """Use this function to test the order of convergence."""
    yfilepath = os.path.join(path,"inputs")
    yfile = os.path.join(yfilepath,ifile)
    d = 0.25 #Constant
    alpha = 1
    error = []
    # sizes = [120,240,360]
    sizes = [12,24,36,48, 60,72,84,96]
    ss = []
    ts = []
    pi = numpy.pi
    analytical = lambda x,y,t: numpy.sin(2*pi*x)*numpy.sin(2*pi*y)*numpy.exp(-8*pi*pi*alpha*t)
    for size in sizes:
        x = numpy.linspace(0,1,size,endpoint=False)
        dx = float(x[1]-x[0])
        t0,tf = 0,0.01
        dt = float(d*dx**2/alpha)
        filename = pysweep.equations.heat.createInitialConditions(size,size,alpha=0.00016563)
        testSolver = pysweep.Solver(filename,yfile)
        testSolver.globals =  [t0,tf,dt,dx,dx,alpha,True]
        testSolver()
        hdf5 = h5py.File(testSolver.output,"r")
        data = hdf5['data']
        num = data[int(tf//dt),0,int(size//2),int(size//2)]
        an = analytical(tf,int(size//2)*dx,int(size//2)*dx)
        error.append(float(abs(an-num)))
        ts.append(float(dt))
        ss.append(float(dx))
        hdf5.close()
    
    if testSolver.clusterMasterBool:
        with open('convergence.yaml',"w") as f:
            outdata  = {"error":error,"sizes":sizes,"dx":ss,"dt":ts}
            yaml.dump(outdata,f)
        makeOrderOfConvergencePlot()


def makeOrderOfConvergencePlot():
    """Using yaml file make Order of convergence plot."""
    with open("convergence.yaml","r") as f:
        data = yaml.load(f,Loader=yaml.FullLoader)
        fig,ax = plt.subplots()
        ax.loglog(data['dt'],data['error'])
        ax.set_xlim([10e-6,10e-4])
        plt.show()
        
# if __name__ == "__main__":
    # pass
    # MPI.COMM_SELF.Spawn(sys.executable, args=["from pysweep.tests import testExample; testExample()"], maxprocs=2)
