
import pysweep,numpy,sys,os,h5py,yaml,time,warnings
import matplotlib.pyplot as plt
path = os.path.dirname(os.path.abspath(__file__))
eqnPath = os.path.join(os.path.dirname(path),"equations")
testTimeSteps=50

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

def changeSolverTimeSteps(solver,numberOfTimeSteps):
    solver.globals[1] = numberOfTimeSteps*solver.globals[2] #reset tf for numberOfTimeSteps

def getEqnPath(eqn):   
    """Use this function to get equation path."""
    return os.path.join(eqnPath,eqn)

def testing(func):
    """Use this function as a decorator for other tests.
    It's purpose is to contain a set of test conditions and wrap tests with those conditions.
    exid, share, globals, cpu, gpu, operating_points, and intermediate_steps should be set in test
    """
    def testConfigurations():
        arraysize = 384
        shares = [0,0.625,1] #Shares for GPU
        sims = [True,False] #different simulations
        blocksizes = [8, 12, 16, 24] #blocksizes with most options
        #Creat solver object
        solver = pysweep.Solver(sendWarning=False)
        solver.dtypeStr = 'float64'
        solver.dtype = numpy.dtype(solver.dtypeStr)
        solver.verbose=False
        for sim in sims:
            for bs in blocksizes:
                for sh in shares:
                    solver.moments = [time.time(),]
                    solver.share = sh
                    solver.simulation = sim
                    solver.blocksize = (bs,bs,1)
                    func(solver,arraysize)
    return testConfigurations


def debugging(func):
    """Use this function as a decorator for other tests.
    It's purpose is to contain a set of test conditions and wrap tests with those conditions.
    exid, share, globals, cpu, gpu, operating_points, and intermediate_steps should be set in test
    """
    def testConfigurations():
        arraysize = 32
        shares = [0,] #Shares for GPU
        sims = [True,False] #different simulations
        blocksizes = [16,] #blocksizes with most options
        #Creat solver object
        solver = pysweep.Solver(sendWarning=False)
        solver.dtypeStr = 'float64'
        solver.dtype = numpy.dtype(solver.dtypeStr)
        solver.verbose=False
        for sim in sims:
            for bs in blocksizes:
                for sh in shares:
                    solver.moments = [time.time(),]
                    solver.share = sh
                    solver.simulation = sim
                    solver.blocksize = (bs,bs,1)
                    func(solver,arraysize)
    return testConfigurations

@debugging
def testHalf(solver,arraysize):
    """Use this function to run the half test."""
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "halfConditions.hdf5"
    # if not os.path.exists(filename):
    filename = pysweep.equations.half.createInitialConditions(arraysize,arraysize)
    solver.assignInitialConditions(filename)
    solver.globals = [0,1,0.1,0.1,0.1]
    changeSolverTimeSteps(solver,testTimeSteps)
    solver.operating = 1
    solver.intermediate = 2
    solver.setCPU(getEqnPath("half.py"))
    solver.setGPU(getEqnPath("half.cu"))
    solver.exid = []
    solver.output = "half.hdf5"
    solver.loadCPUModule()
    solver()
    hdf5File = h5py.File(solver.output,"r")
    data = hdf5File['data']
    actual = data[:3,:,:,:]
    solver.compactPrint()
    for i in range(1,testTimeSteps):
        for idx, idy in numpy.ndindex(actual.shape[2:]):
            actual[1,0,idx,idy] = (actual[0,0,(idx+1)%arraysize,idy]-2*actual[0,0,idx,idy]+actual[0,0,idx-1,idy])+(actual[0,0,idx,(idy+1)%arraysize]-2*actual[0,0,idx,idy]+actual[0,0,idx,idy-1])
            actual[1,0,idx,idy] *= 0.5/100
            actual[1,0,idx,idy] += actual[0,0,idx,idy]
        for idx, idy in numpy.ndindex(actual.shape[2:]):
            actual[2,0,idx,idy] = (actual[1,0,(idx+1)%arraysize,idy]-2*actual[1,0,idx,idy]+actual[1,0,idx-1,idy])+(actual[1,0,idx,(idy+1)%arraysize]-2*actual[1,0,idx,idy]+actual[1,0,idx,idy-1])
            actual[2,0,idx,idy] /= 100
            actual[2,0,idx,idy] += actual[0,0,idx,idy]
        assert numpy.allclose(actual[2,0,:,:],data[i,0,:,:],rtol=9e-15,atol=9e-15)
        actual[0,:,:,:] = actual[2,:,:,:]
        
@testing
def testHeatForwardEuler(solver,arraysize,printError=True):
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "heatConditions.hdf5"
    adjustHeatGlobals(solver,arraysize,True,timesteps=testTimeSteps)
    if not os.path.exists(filename):
        filename = pysweep.equations.heat.createInitialConditions(arraysize,arraysize,alpha=solver.globals[-2])
    solver.assignInitialConditions(filename)
    solver.operating = 1
    solver.intermediate = 1
    solver.setCPU(getEqnPath("heat.py"))
    solver.setGPU(getEqnPath("heat.cu"))
    solver.exid = []
    solver.output = "testing.hdf5"
    solver.loadCPUModule()
    solver()
    if solver.clusterMasterBool:
        steps = solver.arrayShape[0]
        stepRange = numpy.array_split(numpy.arange(0,steps),solver.comm.Get_size())
    else:
        stepRange = []
    stepRange = solver.comm.scatter(stepRange)
    error = []
    with h5py.File(solver.output,"r",driver="mpio",comm=solver.comm) as f:
        data = f["data"]
        failed = False
        for i in stepRange:
            T,x,y = pysweep.equations.heat.analytical(arraysize,arraysize,t=solver.globals[2]*i,alpha=solver.globals[-2])
            error.append(numpy.amax(numpy.absolute(data[i,0]-T[0])))
            try:
                assert numpy.allclose(data[i,0],T[0])
            except Exception as e:
                failed = True

    failed = solver.comm.allgather(failed)
    if solver.clusterMasterBool:
        solver.compactPrint()
        masterFail = numpy.all(failed)
        print("{} testHeatForwardEuler".format("Failed:" if masterFail else "Success:"))

    if printError:
        error = solver.comm.allgather(error)
        if solver.clusterMasterBool:
            finalError = []
            for eL in error:
                finalError.append(numpy.amax(eL))
            print("Max Error: {}, Process Max Average: {}".format(numpy.amax(finalError),numpy.mean(finalError)))
    if solver.clusterMasterBool:
        print("\n") #Final end line
    solver.comm.Barrier()

@debugging
def testHeatRungeKuttaTwo(solver,arraysize,printError=True):
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "heatConditions.hdf5"
    adjustHeatGlobals(solver,arraysize,False,timesteps=testTimeSteps)
    if not os.path.exists(filename):
        filename = pysweep.equations.heat.createInitialConditions(arraysize,arraysize,alpha=solver.globals[-2])
    solver.assignInitialConditions(filename)
    solver.operating = 1
    solver.intermediate = 2
    solver.setCPU(getEqnPath("heat.py"))
    solver.setGPU(getEqnPath("heat.cu"))
    solver.exid = []
    solver.output = "testing.hdf5"
    solver.loadCPUModule()
    solver()
    if solver.clusterMasterBool:
        steps = solver.arrayShape[0]
        stepRange = numpy.array_split(numpy.arange(0,steps),solver.comm.Get_size())
    else:
        stepRange = []
    stepRange = solver.comm.scatter(stepRange)
    error = []
    with h5py.File(solver.output,"r",driver="mpio",comm=solver.comm) as f:
        data = f["data"]
        failed = False
        for i in stepRange:
            T,x,y = pysweep.equations.heat.analytical(arraysize,arraysize,t=solver.globals[2]*i,alpha=solver.globals[-2])
            error.append(numpy.amax(numpy.absolute(data[i,0]-T[0])))
            try:
                assert numpy.allclose(data[i,0],T[0])
            except Exception as e:
                failed = True

    failed = solver.comm.allgather(failed)
    if solver.clusterMasterBool:
        solver.compactPrint()
        masterFail = numpy.all(failed)
        print("{} testHeatRungeKuttaTwo".format("Failed:" if masterFail else "Success:"))

    if printError:
        error = solver.comm.allgather(error)
        if solver.clusterMasterBool:
            finalError = []
            for eL in error:
                finalError.append(numpy.amax(eL))
            print("Max Error: {}, Process Max Average: {}".format(numpy.amax(finalError),numpy.mean(finalError)))
    if solver.clusterMasterBool:
        print("\n") #Final end line
    solver.comm.Barrier()

def adjustHeatGlobals(solver,arraysize,scheme,timesteps=50):
    """Use this function to adjust heat equation step size based on arraysize"""
    d = 0.1
    alpha = 1
    dx = 1/arraysize
    dt = float(d*dx**2/alpha)
    solver.globals = [0,dt*timesteps,dt,dx,dx,alpha,scheme]

def compareGPUToCPU():
    """Use this function to compare GPU to CPU based on file output."""
    #Adjustment of global variables
    d = 0.1 #Constant
    alpha = 1
    npx = npy = 8
    T,x,y = pysweep.equations.heat.analytical(npx,npy,0)
    dx = dy = float(x[1]-x[0])
    t0,tf = 0,0.001
    dt = float(d*dx**2/alpha)
    timesteps = 10
    #Input files
    filename = pysweep.equations.heat.createInitialConditions(npx,npy,alpha=alpha)
    inputPath = os.path.join(path,"inputs")
    solver = pysweep.Solver(filename, os.path.join(inputPath,"heatSwept.yaml"))
    adjustHeatGlobals(solver,npx,False,timesteps=timesteps)
    solver.intermediate = 2
    solver.blocksize = (npx,npx,1)
    #Solve CPU
    solver.output = "cpu.hdf5"
    solver.share = 0
    solver()
    cpuFile = h5py.File(solver.output,'r')
    cdata = cpuFile['data']
    #Solver GPU
    solver.assignInitialConditions(filename)
    solver.output = "gpu.hdf5"
    solver.share = 1
    solver()
    gpuFile = h5py.File(solver.output,'r')
    gdata = gpuFile['data']
    #compare
    for i in range(timesteps):
        print(gdata[i,0,:,:]-cdata[i,0,:,:])
        input()
    

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
        solver = pysweep.Solver(filename,yfile)
        solver.globals =  [t0,tf,dt,dx,dx,alpha,True]
        solver()
        hdf5 = h5py.File(solver.output,"r")
        data = hdf5['data']
        num = data[int(tf//dt),0,int(size//2),int(size//2)]
        an = analytical(tf,int(size//2)*dx,int(size//2)*dx)
        error.append(float(abs(an-num)))
        ts.append(float(dt))
        ss.append(float(dx))
        hdf5.close()
    
    if solver.clusterMasterBool:
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


#-------------------------------------Completed Tests------------------#
@testing
def testSimpleOne(solver,arraysize):
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "exampleConditions.hdf5"
    if not os.path.exists(filename):
        filename = pysweep.equations.example.createInitialConditions(1,arraysize,arraysize)
    solver.assignInitialConditions(filename)
    solver.operating = 1
    solver.intermediate = 1
    solver.setCPU(getEqnPath("example.py"))
    solver.setGPU(getEqnPath("example.cu"))
    solver.output = "testing.hdf5"
    solver.globals = [0,1,0.1,0.1,0.1,True]
    changeSolverTimeSteps(solver,testTimeSteps)
    solver.exid = [-1,]
    solver.loadCPUModule()
    solver()
    
    if solver.clusterMasterBool:
        failed = False
        solver.compactPrint()
        with h5py.File(solver.output,"r") as f:
            data = f["data"]
            for i in range(solver.arrayShape[0]):
                try:
                    assert numpy.all(data[i,0,:,:]==i)
                except Exception as e:
                    failed = True
        print("{} testSimpleOne\n".format("Failed:" if failed else "Success:"))
    solver.comm.Barrier()

@testing 
def testSimpleTwo(solver,arraysize):
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "exampleConditions.hdf5"
    if not os.path.exists(filename):
        filename = pysweep.equations.example.createInitialConditions(1,arraysize,arraysize)
    solver.assignInitialConditions(filename)
    solver.operating = 2
    solver.intermediate = 2
    solver.setCPU(getEqnPath("example.py"))
    solver.setGPU(getEqnPath("example.cu"))
    solver.output = "testing.hdf5"
    solver.globals = [0,1,0.1,0.1,0.1,False]
    changeSolverTimeSteps(solver,testTimeSteps)
    solver.exid = []
    solver.loadCPUModule()
    solver()
    if solver.clusterMasterBool:
        failed = False
        solver.compactPrint()
        with h5py.File(solver.output,"r") as f:
            data = f["data"]
            for i in range(solver.arrayShape[0]):
                try:
                    assert numpy.all(data[i,0,:,:]==2*i)
                except Exception as e:
                    failed = True
                    print(i,data[i,0])
                    input()
        print("{} testSimpleTwo\n".format("Failed:" if failed else "Success:"))
    solver.comm.Barrier()

@testing
def testCheckerOne(solver,arraysize):
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "checkerConditions.hdf5"
    if not os.path.exists(filename):
        filename = pysweep.equations.checker.createInitialConditions(1,arraysize,arraysize)
    solver.assignInitialConditions(filename)
    solver.operating = 1
    solver.intermediate = 1
    solver.setCPU(getEqnPath("checker.py"))
    solver.setGPU(getEqnPath("checker.cu"))
    solver.output = "testing.hdf5"
    solver.globals = [0,1,0.1,0.1,0.1,True]
    changeSolverTimeSteps(solver,testTimeSteps)
    solver.exid = []
    solver.loadCPUModule()
    solver()
    if solver.clusterMasterBool:
        failed = False
        solver.compactPrint()
        with h5py.File(solver.output,"r") as f:
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
            for i in range(0,solver.arrayShape[0]):
                try:
                    if i%2==0:
                        assert numpy.all(data[i,0,:,:]==pattern1)
                    else:
                        assert numpy.all(data[i,0,:,:]==pattern2)
                except Exception as e:
                    failed = True
        print("{} testCheckerOne\n".format("Failed:" if failed else "Success:"))
    solver.comm.Barrier()


@testing
def testCheckerTwo(solver,arraysize):
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "checkerConditions.hdf5"
    if not os.path.exists(filename):
        filename = pysweep.equations.checker.createInitialConditions(1,arraysize,arraysize)
    solver.assignInitialConditions(filename)
    solver.operating = 2
    solver.intermediate = 2
    solver.setCPU(getEqnPath("checker.py"))
    solver.setGPU(getEqnPath("checker.cu"))
    solver.output = "testing.hdf5"
    solver.globals = [0,1,0.1,0.1,0.1,False]
    changeSolverTimeSteps(solver,testTimeSteps)
    solver.exid = []
    solver.loadCPUModule()
    solver()
    if solver.clusterMasterBool:
        failed = False
        solver.compactPrint()
        with h5py.File(solver.output,"r") as f:
            data = f["data"]
            nt,nv,nx,ny = numpy.shape(data)
            pattern1 = numpy.zeros((nx,ny))
            pattern2 = numpy.zeros((nx,ny))
            for i in range(0,nx,2):
                for j in range(0,ny,2):
                    pattern1[i,j]=1
            for i in range(1,nx,2):
                for j in range(1,ny,2):
                    pattern1[i,j]=1
            for i in range(0,solver.arrayShape[0]):
                try:
                    assert numpy.all(data[i,0,:,:]==pattern1)
                except Exception as e:
                    failed = True
        print("{} testCheckerTwo\n".format("Failed:" if failed else "Success:"))
    solver.comm.Barrier()



# if __name__ == "__main__":
    # pass
    # MPI.COMM_SELF.Spawn(sys.executable, args=["from pysweep.tests import testExample; testExample()"], maxprocs=2)
