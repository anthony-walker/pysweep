#Programmer: Anthony Walker
#This is a file for testing decomp and sweep
import pysweep, argparse
import mpi4py.MPI as MPI

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
    args = parser.parse_args()
    
    if args.filename is not None and args.processes is not None: #Executing simulation
        execute(args)
    
    if args.yaml is not None: #Generating yaml
        pysweep.generateInputFile(args.yaml)