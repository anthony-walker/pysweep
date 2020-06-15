#Programmer: Anthony Walker
#This is a file for testing decomp and sweep
import sys,argparse, os
from node.sweep import nodesweep
from distributed.sweep import distsweep
from analytical import vortex
from node.decomposition import decomp
from distributed.decomposition import ddecomp
import numpy as np
from mpi4py import MPI

path = os.path.dirname(os.path.abspath(__file__))
epath = os.path.join(path,'equations')

def AnalyticalVortex(args):
    """Use this function to create analytical vortex data."""
    cvics = vortex.vics()
    cvics.Shu(args.gamma)
    #Creating initial vortex from analytical code
    create_vortex_data(cvics,args.nx,args.nx,times=np.arange(args.t0,args.tf,args.dt),filepath="./",filename=args.hdf5)

def SweptVortex(args):
    #Analytical properties
    cvics = vortex.vics()
    cvics.Shu(args.gamma)
    #Creating initial vortex from analytical code
    Mstat = 0 if args.stationary else None
    flux_vortex = vortex.steady_vortex(cvics,args.nx,args.ny,M_o=Mstat)[0]
    #Dimensions and steps
    dx = 2*cvics.L/(args.nx-1)
    dy = 2*cvics.L/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'euler.h'),os.path.join(epath,'euler.py'))
    if args.distributed:
        distsweep.dsweep(flux_vortex,gargs,swargs,filename=args.hdf5)
    else:
        nodesweep.nsweep(flux_vortex,gargs,swargs,filename=args.hdf5)

def StandardVortex(args):
    #Analytical properties
    cvics = vortex.vics()
    cvics.Shu(args.gamma)
    #Creating initial vortex from analytical code
    Mstat = 0 if args.stationary else None
    flux_vortex = vortex.steady_vortex(cvics,args.nx,args.ny,M_o=Mstat)[0]
    #Dimensions and steps
    dx = 2*cvics.L/(args.nx-1)
    dy = 2*cvics.L/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'euler.h'),os.path.join(epath,'euler.py'))
    if args.distributed:
        ddecomp.ddecomp(flux_vortex,gargs,swargs,filename=args.hdf5)
    else:
        decomp.decomp(flux_vortex,gargs,swargs,filename=args.hdf5)

def UnsteadySweptVortex(args):
    #Analytical properties
    cvics = vortex.vics()
    cvics.STC(args.gamma)
    #Creating initial vortex from analytical code
    flux_vortex = vortex.steady_vortex(cvics,args.nx,args.ny)[0]
    #Dimensions and steps
    dx = 2*cvics.L/(args.nx-1)
    dy = 2*cvics.L/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'euler.h'),os.path.join(epath,'euler.py'))
    if args.distributed:
        distsweep.dsweep(flux_vortex,gargs,swargs,filename=args.hdf5)
    else:
        nodesweep.nsweep(flux_vortex,gargs,swargs,filename=args.hdf5)

def UnsteadyStandardVortex(args):
    #Analytical properties
    cvics = vortex.vics()
    cvics.STC(args.gamma)
    #Creating initial vortex from analytical code
    flux_vortex = vortex.steady_vortex(cvics,args.nx,args.ny)[0]
    #Dimensions and steps
    dx = 2*cvics.L/(args.nx-1)
    dy = 2*cvics.L/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'euler.h'),os.path.join(epath,'euler.py'))
    if args.distributed:
        ddecomp.ddecomp(flux_vortex,gargs,swargs,filename=args.hdf5)
    else:
        decomp.decomp(flux_vortex,gargs,swargs,filename=args.hdf5)

def SweptShock(args):
    #Creating initial vortex from analytical code
    leftBC = (1,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    arr2D = np.zeros((4,args.nx,args.ny))
    nyh = int(args.ny/2)
    nxh = int(args.nx/2)
    nxq = int(nxh/2)
    nyq = int(nyh/2)
    if args.orient == 0:
        for i in range(nxh):
            for j in range(args.ny):
                arr2D[:,i,j] = leftBC
                arr2D[:,i+nxh,j] = rightBC
    else:
        for i in range(nyh):
            for j in range(args.nx):
                arr2D[:,j,i] = leftBC
                arr2D[:,j,i+nyh] = rightBC
    #Dimensions and steps
    dx = args.X/(args.nx-1)
    dy = args.Y/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'euler.h'),os.path.join(epath,'euler.py'))
    if args.distributed:
        distsweep.dsweep(arr2D,gargs,swargs,filename=args.hdf5,exid=[1])
    else:
        nodesweep.nsweep(arr2D,gargs,swargs,filename=args.hdf5)

def StandardShock(args):
    #Creating initial vortex from analytical code
    leftBC = (1,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    arr2D = np.zeros((4,args.nx,args.ny))
    nyh = int(args.ny/2)
    nxh = int(args.nx/2)
    nxq = int(nxh/2)
    nyq = int(nyh/2)
    if args.orient == 0:
        for i in range(nxh):
            for j in range(args.ny):
                arr2D[:,i,j] = leftBC
                arr2D[:,i+nxh,j] = rightBC
    else:
        for i in range(nyh):
            for j in range(args.nx):
                arr2D[:,j,i] = leftBC
                arr2D[:,j,i+nyh] = rightBC
    #Dimensions and steps
    dx = args.X/(args.nx-1)
    dy = args.Y/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'euler.h'),os.path.join(epath,'euler.py'))
    if args.distributed:
        ddecomp.ddecomp(arr2D,gargs,swargs,filename=args.hdf5,exid=[1])
    else:
        decomp.decomp(arr2D,gargs,swargs,filename=args.hdf5)


def SweptSimple(args):
    #Analytical properties
    arr0 = np.zeros((4,args.nx,args.ny))
    ct = 0
    for idz in range(arr0.shape[0]):
        for idx in range(arr0.shape[1]):
            for idy in range(arr0.shape[2]):
                arr0[idz,idx,idy] = ct
                ct += 1
    #Dimensions and steps
    dx = args.X/(args.nx-1)
    dy = args.Y/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'eqt.h'),os.path.join(epath,'eqt.py'))
    if args.distributed:
        distsweep.dsweep(arr0,gargs,swargs,filename=args.hdf5,exid=[1])
    else:
        nodesweep.nsweep(arr0,gargs,swargs,filename=args.hdf5)

def StandardSimple(args):
    #Analytical properties
    arr0 = np.zeros((4,args.nx,args.ny))
    ct = 0
    for idz in range(arr0.shape[0]):
        for idx in range(arr0.shape[1]):
            for idy in range(arr0.shape[2]):
                arr0[idz,idx,idy] = ct
                ct += 1
    #Dimensions and steps
    dx = args.X/(args.nx-1)
    dy = args.Y/(args.ny-1)
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'eqt.h'),os.path.join(epath,'eqt.py'))
    if args.distributed:
        ddecomp.ddecomp(arr0,gargs,swargs,filename=args.hdf5,exid=[1])
    else:
        decomp.decomp(arr0,gargs,swargs,filename=args.hdf5)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected as string.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    fmap = {'unsteady_swept_vortex' : UnsteadySweptVortex,
                    'unsteady_standard_vortex' : UnsteadyStandardVortex,'swept_vortex' : SweptVortex,
                    'standard_vortex' : StandardVortex,"analytical_vortex":AnalyticalVortex,'swept_simple':SweptSimple,'standard_simple':StandardSimple,'swept_shock':SweptShock,'standard_shock':StandardShock}
    parser.add_argument('function', choices=fmap.keys())
    parser.add_argument("-b","--block",nargs="?",default=8,type=int)
    parser.add_argument("-a","--affinity",nargs="?",default=0.5,type=float)
    parser.add_argument("--hdf5",nargs="?",default="./result",type=str)
    parser.add_argument('--distributed',nargs="?",default=True,type=str2bool)

    #Case specific arguements
    parser.add_argument("-o","--orient",nargs="?",default=0,type=int)
    parser.add_argument("-TH",nargs="?",default=373.0,type=float)
    parser.add_argument("-TL",nargs="?",default=298.0,type=float)
    parser.add_argument("--alpha",nargs="?",default=25e-6,type=float)
    parser.add_argument("-R",nargs="?",default=1,type=float)
    parser.add_argument("--gamma",nargs="?",default=1.4,type=float)
    parser.add_argument('--stationary',nargs="?",default=False,type=str2bool)

    #Dimensional arguments
    parser.add_argument("-t0",nargs="?",default=0,type=float)
    parser.add_argument("-tf",nargs="?",default=1,type=float)
    parser.add_argument("-dt",nargs="?",default=0.05,type=float)
    parser.add_argument("-nx",nargs="?",default=32,type=int)
    parser.add_argument("-ny",nargs="?",default=32,type=int)
    parser.add_argument("-X",nargs="?",default=1,type=float)
    parser.add_argument("-Y",nargs="?",default=1,type=float)

    #Arguments that cannot change for the given equations
    # parser.add_argument("-o","--ops",nargs="?",default=2,type=int)
    # parser.add_argument("--tso",nargs="?",default=2,type=int)
    # parser.add_argument("-g","--gpu",nargs="?",default=os.path.join(epath,'euler.h'),type=str)
    # parser.add_argument("-c","--cpu",nargs="?",default=os.path.join(epath,'euler.py'),type=str)

    args = parser.parse_args()
    fmap[args.function](args)
