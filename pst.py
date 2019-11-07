#Programmer: Anthony Walker
#This is a file for testing decomp and sweep
import sys,argparse, os
from node.sweep import nodesweep
from distributed.sweep import distsweep
from analytical import vortex,ahde
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
    #Dimensions and steps
    dx = 2*cvics.L/args.nx
    dy = 2*cvics.L/args.ny
    #Creating initial vortex from analytical code
    flux_vortex = cvics.Shu(args.gamma,args.nx).flux[0]
    # #Changing arguments
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
    #Dimensions and steps
    dx = 2*cvics.L/args.nx
    dy = 2*cvics.L/args.ny
    #Creating initial vortex from analytical code
    flux_vortex = cvics.Shu(args.gamma,args.nx).flux[0]
    # #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'euler.h'),os.path.join(epath,'euler.py'))
    if args.distributed:
        ddecomp.ddecomp(flux_vortex,gargs,swargs,filename=args.hdf5)
    else:
        decomp.decomp(flux_vortex,gargs,swargs,filename=args.hdf5)

def SweptHDE(args):
    #Analytical properties
    arr0 = ahde.TIC(args.nx,args.ny,args.X,args.Y,args.R,args.TH,args.TL)[0,:,:,:]
    #Dimensions and steps
    dx = args.X/args.nx
    dy = args.Y/args.ny
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.alpha)
    swargs = (2,1,args.block,args.affinity,os.path.join(epath,'hde.h'),os.path.join(epath,'hde.py'))
    if args.distributed:
        distsweep.dsweep(arr0,gargs,swargs,filename=args.hdf5)
    else:
        nodesweep.nsweep(arr0,gargs,swargs,filename=args.hdf5)

def StandardHDE(args):
    #Analytical properties
    arr0 = ahde.TIC(args.nx,args.ny,args.X,args.Y,args.R,args.TH,args.TL)[0,:,:,:]
    #Dimensions and steps
    dx = args.X/args.nx
    dy = args.Y/args.ny
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.alpha)
    swargs = (2,1,args.block,args.affinity,os.path.join(epath,'hde.h'),os.path.join(epath,'hde.py'))
    if args.distributed:
        ddecomp.ddecomp(arr0,gargs,swargs,filename=args.hdf5)
    else:
        decomp.decomp(arr0,gargs,swargs,filename=args.hdf5)


def SweptSimple(args):
    #Analytical properties
    arr0 = arr = np.ones((4,args.nx,args.ny))
    #Dimensions and steps
    dx = args.X/args.nx
    dy = args.Y/args.ny
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'eqt2.h'),os.path.join(epath,'eqt2.py'))
    if args.distributed:
        distsweep.dsweep(arr0,gargs,swargs,filename=args.hdf5)
    else:
        nodesweep.nsweep(arr0,gargs,swargs,filename=args.hdf5)

def StandardSimple(args):
    #Analytical properties
    arr0 = arr = np.ones((4,args.nx,args.ny))
    #Dimensions and steps
    dx = args.X/args.nx
    dy = args.Y/args.ny
    #Changing arguments
    gargs = (args.t0,args.tf,args.dt,dx,dy,args.gamma)
    swargs = (2,2,args.block,args.affinity,os.path.join(epath,'eqt2.h'),os.path.join(epath,'eqt2.py'))
    if args.distributed:
        ddecomp.ddecomp(arr0,gargs,swargs,filename=args.hdf5)
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
    fmap = {'swept_vortex' : SweptVortex,
                    'standard_vortex' : StandardVortex,"analytical_vortex":AnalyticalVortex,'swept_hde' : SweptHDE,
                                    'standard_hde' : StandardHDE,'swept_simple':SweptSimple,'standard_simple':StandardSimple}
    parser.add_argument('function', choices=fmap.keys())
    parser.add_argument("-b","--block",nargs="?",default=8,type=int)
    parser.add_argument("-a","--affinity",nargs="?",default=0.5,type=float)
    parser.add_argument("--hdf5",nargs="?",default="./result",type=str)
    parser.add_argument('--distributed',nargs="?",default=True,type=str2bool)

    #Case specific arguements
    parser.add_argument("-TH",nargs="?",default=373.0,type=float)
    parser.add_argument("-TL",nargs="?",default=298.0,type=float)
    parser.add_argument("--alpha",nargs="?",default=25e-6,type=float)
    parser.add_argument("-R",nargs="?",default=1,type=float)
    parser.add_argument("--gamma",nargs="?",default=1.4,type=float)

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
