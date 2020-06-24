#Programmer: Anthony Walker
#This is a file for testing decomp and sweep
import pysweep
import mpi4py.MPI as MPI

def execute(filename,numberOfProcesses,hostfile):
    """Use this function to execute pysweep in parallel by spawning MPI processes."""

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
    parser.add_argument("-f","--filename",nargs="?",type=str)
    parser.add_argument("-n","--processes",nargs="?",type=int)
    parser.add_argument("-h","--hostfile",nargs="?",type=int)
    parser.add_argument("-y","--yaml",nargs="?",choices=fmap,type=str)
    args = parser.parse_args()
