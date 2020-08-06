#Programmer: Anthony Walker
#This is a file for testing decomp and sweep
import pysweep, argparse, warnings, os, numpy
import pysweep.utils.figures as figures
path = os.path.dirname(os.path.abspath(__file__))
eqnPath = os.path.join(os.path.dirname(path),"equations")

def getEqnPath(eqn):   
    """Use this function to get equation path."""
    return os.path.join(eqnPath,eqn)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected as string.')

def adjustEulerGlobals(solver,arraysize,timesteps=50):
    """Use this function to adjust heat equation step size based on arraysize"""
    

def runEuler(args):
    """Use this function to run Euler example."""
    if not args.ignore:
        warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = pysweep.equations.euler.createInitialConditions(args.spacesteps,args.spacesteps)
    simwarn = True if args.ignore else False
    solver = pysweep.Solver(initialConditions=filename,sendWarning=simwarn)
    #Setting globals
    d = 0.1
    gamma = 1.4
    dx = 10/(args.spacesteps-1)
    dt = d*dx
    solver.globals = [0,dt*args.timesteps,dt,dx,dx,gamma]
    solver.dtypeStr = 'float64'
    solver.dtype = numpy.dtype(solver.dtypeStr)
    solver.verbose = False
    solver.operating = 2
    solver.intermediate = 2
    solver.share = args.share
    solver.simulation = True if args.swept else False
    solver.blocksize = (args.blocksize,args.blocksize,1)
    solver.setCPU(getEqnPath("euler.py"))
    solver.setGPU(getEqnPath("euler.cu"))
    solver.exid = []
    solver.output = "eulerOutput.hdf5"
    solver.loadCPUModule()
    if args.verbose:
        if solver.rank == 0:
            print(solver)
    solver()
    #Clean up 
    if args.clean:
        if solver.rank == 0:
            try:
                os.system("rm {}".format(filename))
                os.system("rm {}".format(solver.output))
            except Exception as e:
                pass

def runHeat(args):
    """Use this function to run Heat example."""
    if not args.ignore:
        warnings.filterwarnings('ignore') #Ignore warnings for processes
    filename = pysweep.equations.heat.createInitialConditions(args.spacesteps,args.spacesteps)
    simwarn = True if args.ignore else False
    solver = pysweep.Solver(initialConditions=filename,sendWarning=simwarn)
    #Setting globals
    d = 0.1
    alpha = 1
    dx = 1/(args.spacesteps-1)
    dt = float(d*dx**2/alpha)
    solver.globals = [0,dt*args.timesteps,dt,dx,dx,alpha,True]
    solver.dtypeStr = 'float64'
    solver.dtype = numpy.dtype(solver.dtypeStr)
    solver.verbose = False 
    solver.operating = 2
    solver.intermediate = 1
    solver.share = args.share
    solver.simulation = True if args.swept else False
    solver.blocksize = (args.blocksize,args.blocksize,1)
    solver.setCPU(getEqnPath("heat.py"))
    solver.setGPU(getEqnPath("heat.cu"))
    solver.exid = []
    solver.output = "heatOutput.hdf5"
    solver.loadCPUModule()
    if args.verbose:
        if solver.rank==0:
            print(solver)
    solver()
    #Clean up 
    if args.clean:
        if solver.rank == 0:
            try:
                os.system("rm {}".format(filename))
                os.system("rm {}".format(solver.output))
            except Exception as e:
                pass

def commandLine():
    """Use this function to run pysweep from the commandline."""
    #Running simulation 
    parser = argparse.ArgumentParser()
    functionMap = {"euler":runEuler,"heat":runHeat}
    parser.add_argument("-f","--function",nargs="?",default=None,choices=functionMap,type=str,help="This specifies which function example to run.")
    parser.add_argument('--swept', action='store_true', help="Set simulation type to swept.")
    parser.add_argument("-b","--blocksize",default=12,nargs="?",type=int,help="blocksize should be 2^n and constrained by your GPU. It is also constrained by the relation mod(b,2*k)==0 where k is the number of points on each side of the operating point, e.g., 2 for a 5 point stencil.")
    parser.add_argument("-nx","--spacesteps",default=24,nargs="?",type=int,help="This specifies the number of points in x and y. Square domains only with this feature.")
    parser.add_argument("-nt","--timesteps",default=10,nargs="?",type=int,help="This specifies the number of timesteps to take.")
    parser.add_argument("-s","--share",default=0.5,nargs="?",type=float,help="This specifies the GPU share, a value from 0 to 1.")
    parser.add_argument('--clean', action='store_true', help="Clean up any file not necessary to performance testing including results.")
    parser.add_argument('--ignore', action='store_true', help="Ignore warnings with this option.")
    parser.add_argument('-v','--verbose', action='store_true', help="Enable verbose simulation.")
    #Command line args for figure creation
    figureMap = {"up":figures.Up1,'y1':figures.Y1,'c1':figures.Comm1,'x1':figures.X1,'oct':figures.Oct1,'y2':figures.Y2,'c2':figures.Comm2,'x2':figures.X2,'down':figures.DWP1,"all":figures.createAll}
    keyStr = ", ".join(figureMap.keys())
    parser.add_argument("-p","--plot",nargs="+",choices=figureMap.keys(),type=str,help="This option runs the specified plotting function and outputs it as a pdf.")
    args = parser.parse_args()
    #Figure generation
    if args.plot is not None:
        for arg in args.plot:
            try:
                figureMap[arg]()
            except Exception as e:
                warnings.warn('Figure function not found: {}'.format(arg))
    
    if args.function is not None:
        functionMap[args.function](args)