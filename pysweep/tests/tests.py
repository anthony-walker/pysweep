
import pysweep,numpy,sys,os,h5py,yaml,time,warnings
import matplotlib.pyplot as plt
path = os.path.dirname(os.path.abspath(__file__))
eqnPath = os.path.join(os.path.dirname(path),"equations")
testTimeSteps=50
globalArraySize = None #Used to prevent repeated file creation

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

def performance(func):
    """Use this function as a decorator for other tests.
    It's purpose is to contain a set of test conditions and wrap tests with those conditions.
    exid, share, globals, cpu, gpu, operating_points, and intermediate_steps should be set in test
    """
    def testConfigurations():
        numSizes = 5
        shares = [0,0.625,1] #Shares for GPU
        sims = [True,False] #different simulations
        blocksizes = [8, 12, 16, 24, 32] #blocksizes with most options
        arraysizes = [int(768*i) for i in range(1,6,1)] #Performance array sizes
        #Create solver object
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
        sims = [True,] #different simulations
        blocksizes = [32,] #blocksizes with most options
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

#----------------------------------End Decorator Functions-------------------------------------------
@debugging
def testValidateVortex(solver,arraysize):
    """Use this funciton to validate the swept solver with a 2D euler vortex"""

    #--------------------------------_Swept Vortex-------------------------------------#
    swept_file = "\""+os.path.join(dpath,"dist_swept_vortex")+"\""
    sfn = os.path.join(dpath,"dist_swept_vortex.hdf5")
    sfp = "\""+sfn+"\""
    t0,tf,dt,npx,X,blks,aff,nps = args
    npy=npx
    Y=X
    gamma = 1.4
    times = np.arange(t0,tf+dt,dt)
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+ " "

    if not os.path.isfile(sfn):
    #Create data using solver
        print("Executing Swept Unsteady Vortex")
        estr = "mpiexec -n "+str(nps)+" "+hoststr+" "+" python "+os.path.join(path[:-5],"pst.py")+" unsteady_swept_vortex "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + swept_file + pts +time_str
        os.system(estr)


    decomp_file = "\""+os.path.join(dpath,"dist_decomp_vortex")+"\""
    dfn = os.path.join(dpath,"dist_decomp_vortex.hdf5")
    dfp = "\""+dfn+"\""

    if not os.path.isfile(dfn):
    #Create data using solver
        print("Executing Standard Unsteady Vortex")
        estr = "mpiexec -n "+str(nps)+" "+hoststr+" "+" python "+os.path.join(path[:-5],"pst.py")+" unsteady_standard_vortex "
        estr += "-b "+str(blks)+" -a "+str(aff)+ " "
        estr += " --hdf5 " + decomp_file + pts +time_str
        os.system(estr)

    vfn = os.path.join(dpath,'vortex0.hdf5')
    if not os.path.isfile(vfn):
        print("Making AnalyticalVortex")
        cvics = vortex.vics()
        cvics.STC(gamma)
        vortex.create_steady_data(cvics,npx,npy,times=times, filepath = dpath,filename = "/vortex",fdb=True)

    pfn = os.path.join(dpath,"clawres.hdf5")



    shdf5_file = h5py.File(sfn, 'r')
    swept = shdf5_file['data'][:,:,:,:]
    dhdf5_file = h5py.File(dfn, 'r')
    decomp = dhdf5_file['data'][:,:,:,:]
    vhdf5_file = h5py.File(vfn, 'r')
    avortex = vhdf5_file['data'][:,:,:,:]
    pack_file = h5py.File(pfn,'r')
    cpack = pack_file['data'][:,:,:,:]

    sfs = "%0.2e"
    # err1 = [np.amax(abs(swept[i]-decomp[i])) for i in range(len(swept))]
    err2 = [np.amax(abs(swept[i]-avortex[i])) for i in range(len(swept))]
    err3 = [np.amax(abs(decomp[i]-avortex[i])) for i in range(len(decomp))]
    err4 = [np.amax(abs(cpack[i]-avortex[i])) for i in range(len(decomp))]
    # tarr = cpack[0]-avortex[1]
    # print(err3)
    print(max(err2))
    print(max(err3))
    print(max(err4))
    xpts = np.linspace(-X/2,X/2,npx,dtype=np.float64)
    ypts = np.linspace(-Y/2,Y/2,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')

    xptsn = np.linspace(-0.5,0.5,npx,dtype=np.float64)
    yptsn = np.linspace(-0.5,0.5,npy,dtype=np.float64)
    xgridn,ygridn = np.meshgrid(xptsn,yptsn,sparse=False,indexing='ij')

    pidx = -1
    fig,axes = plt.subplots(2,2)
    axes1,axes2 = axes
    ax1,ax2 = axes1
    ax3,ax4 = axes2

    ax1.contourf(xgrid,ygrid,avortex[pidx,0,:,:],levels=20,cmap=cm.inferno)
    ax2.contourf(xgridn,ygridn,cpack[pidx,0,:,:],levels=20,cmap=cm.inferno)
    ax3.contourf(xgrid,ygrid,decomp[pidx,0,:,:],levels=20,cmap=cm.inferno)
    ax4.contourf(xgrid,ygrid,swept[pidx,0,:,:],levels=20,cmap=cm.inferno)
    plt.show()

    vhdf5_file.close()
    dhdf5_file.close()
    # pack_file.close()
    if remove_file:
        os.system("rm "+sfp)
        os.system("rm "+dfp)
        os.system("rm "+vfn)

@debugging
def testHalf(solver,arraysize):
    """Use this function to run the half test."""
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "halfConditions.hdf5"
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
    for i in range(1,numpy.shape(data)[0]):
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

@debugging
def testHalf(solver,arraysize):
    """Use this function to run the half test."""
    warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = "halfConditions.hdf5"
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
    for i in range(1,numpy.shape(data)[0]):
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

@testing
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

# if __name__ == "__main__":
    # pass
    # MPI.COMM_SELF.Spawn(sys.executable, args=["from pysweep.tests import testExample; testExample()"], maxprocs=2)
