#Programmer: Anthony Walker
#This is a file for testing decomp and sweep
import pysweep, argparse, warnings
import mpi4py.MPI as MPI
import pysweep.utils.figures as figures

def execute(args):
    """Use this function to execute pysweep in parallel by spawning MPI processes."""
    print("Executing {} with {} processes.".format(args.filename,args.processes))
    if args.hostfile is not None:
        print("Hostfile provided: {}".format(args.hostfile))
    

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected as string.')

def commandLine():
    """Use this function to run pysweep from the commandline."""
    parser = argparse.ArgumentParser()
    fmap = ["euler","heat","example"]
    parser.add_argument("-f","--filename",nargs="?",type=str,help="This is the name of the input file that will be used in execution.")
    parser.add_argument("-n","--processes",nargs="?",type=int,help="This is the number of processes to execute.")
    parser.add_argument("--hostfile",nargs="?",type=str,help="This is a file containing hosts to run on.")
    parser.add_argument("-y","--yaml",nargs="?",choices=fmap,type=str,help="This generates a yaml file based on the provided equation packages.")
    
    figuremap = {"up":figures.Up1,'y1':figures.Y1,'c1':figures.Comm1,'x1':figures.X1,'oct':figures.Oct1,'y2':figures.Y2,'c2':figures.Comm2,'x2':figures.X2,'down':figures.DWP1,"all":figures.createAll}
    keyStr = ", ".join(figuremap.keys())
    parser.add_argument("-p","--plot",nargs="+",choices=figuremap.keys(),type=str,help="This option runs the specified plotting function and outputs it as a pdf.")
    
    args = parser.parse_args()
    #Figure generation
    for arg in args.plot:
        try:
            figuremap[arg]()
        except Exception as e:
            warnings.warn('Figure function not found: {}'.format(arg))
    
    
    #MPIEXEC ENTRY
    # comm = MPI.COMM_SELF.Spawn("/home/anthony-walker/miniconda3/envs/pysweep-dev/bin/pysweep",
    #                        args=["-h",],
    #                        maxprocs=2)