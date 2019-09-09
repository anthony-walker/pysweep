#Programmer: Anthony Walker
#This is a file for testing decomp and sweep
from sweep import *
from analytical import *
from equations import *
from decomp import *
import argparse
import numpy as np


def AnalyticalVortex(args):
    """Use this function to create analytical vortex data."""
    cvics = vics()
    cvics.Shu(args.gamma)
    #Creating initial vortex from analytical code
    create_vortex_data(cvics,args.nx,args.nx,times=np.arange(args.t0,args.tf,args.dt),filepath="./",filename=args.hdf5)

def SweptVortex(args):
    #Analytical properties
    cvics = vics()
    cvics.Shu(args.gamma)
    #Dimensions and steps
    dx = 2*cvics.L/args.nx
    dy = 2*cvics.L/args.ny
    #Creating initial vortex from analytical code
    flux_vortex = cvics.Shu(gamma,args.nx).flux[0]
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (args.tso,args.ops,args.block,args.affinity,args.gpu,args.cpu)
    sweep(flux_vortex,gargs,swargs,filename=args.hdf5)

def StandardVortex(args):
    #Analytical properties
    cvics = vics()
    cvics.Shu(args.gamma)
    #Dimensions and steps
    dx = 2*cvics.L/args.nx
    dy = 2*cvics.L/args.ny
    #Creating initial vortex from analytical code
    flux_vortex = cvics.Shu(gamma,args.nx).flux[0]
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (args.tso,args.ops,args.block,args.affinity,args.gpu,args.cpu)
    decomp(flux_vortex,gargs,swargs,filename=args.hdf5)

def SweptTestPattern(args):
    comm = MPI.COMM_WORLD
    master_rank = 0
    rank = comm.Get_rank()  #current rank
    arr = np.ones((4,args.nx,args.ny))
    printer = pysweep_printer(rank,master_rank)
    X = 1
    Y = 1
    #Dimensions and steps
    dx = X/args.nx
    dy = Y/args.ny
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (args.tso,args.ops,args.block,args.affinity,args.gpu,args.cpu)
    sweep(arr,gargs,swargs,filename=args.hdf5)

def DecompTestPattern(args):
    comm = MPI.COMM_WORLD
    master_rank = 0
    rank = comm.Get_rank()  #current rank
    arr = np.ones((4,args.nx,args.ny))
    printer = pysweep_printer(rank,master_rank)
    X = 1
    Y = 1
    #Dimensions and steps
    dx = X/args.nx
    dy = Y/args.ny
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (args.tso,args.ops,args.block,args.affinity,args.gpu,args.cpu)
    decomp(arr,gargs,swargs,filename=args.hdf5)


parser = argparse.ArgumentParser()
fmap = {'swept' : SweptVortex,
                'standard' : StandardVortex, "stest":SweptTestPattern, "dtest":DecompTestPattern, "analytical":AnalyticalVortex}
parser.add_argument('fcn', choices=fmap.keys())
parser.add_argument("-b","--block",nargs="?",default=8,type=int)
parser.add_argument("-o","--ops",nargs="?",default=2,type=int)
parser.add_argument("--tso",nargs="?",default=2,type=int)
parser.add_argument("-a","--affinity",nargs="?",default=0.5,type=float)
parser.add_argument("-g","--gpu",nargs="?",default="./src/equations/euler.h",type=str)
parser.add_argument("-c","--cpu",nargs="?",default="./src/equations/euler.py",type=str)
parser.add_argument("--hdf5",nargs="?",default="./results/result",type=str)
parser.add_argument("-nx",nargs="?",default=32,type=int)
parser.add_argument("-ny",nargs="?",default=32,type=int)
parser.add_argument("-X",nargs="?",default=1,type=float)
parser.add_argument("-Y",nargs="?",default=1,type=float)
parser.add_argument("--gamma",nargs="?",default=1.4,type=float)
parser.add_argument("-t0",nargs="?",default=0,type=float)
parser.add_argument("-tf",nargs="?",default=1,type=float)
parser.add_argument("-dt",nargs="?",default=0.05,type=float)
args = parser.parse_args()
fmap[args.fcn](args)