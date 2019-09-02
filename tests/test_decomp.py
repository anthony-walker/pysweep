#Programmer: Anthony Walker
#Use the functions in this file to test the decomposition code.
import numpy as np
import sys
import os
cwd = os.getcwd()
sys.path.insert(1,cwd+"/src")
import matplotlib
from analytical import *
from equations import *
from decomp import *
import numpy as np

def test_reg_edge_comm():
    """Use this function to test the communication"""
    t = 3
    v = 2
    x = 20
    ops = 2
    bs = 8
    tarr = np.zeros((t,v,x,x))
    tarr[:,:,ops:x-ops,ops:2*ops] = 2
    tarr[:,:,ops:2*ops,ops:x-ops] = 3
    tarr[:,:,ops:x-ops,x-2*ops:x-ops] = 5
    tarr[:,:,x-2*ops:x-ops,ops:x-ops] = 6
    wrs = tuple()
    wrs += (slice(0,t,1),slice(0,v,1),slice(ops,bs+ops,1),slice(ops,bs+ops,1)),
    wrs += (slice(0,t,1),slice(0,v,1),slice(bs+ops,2*bs+ops,1),slice(ops,bs+ops,1)),
    wrs += (slice(0,t,1),slice(0,v,1),slice(ops,bs+ops,1),slice(bs+ops,2*bs+ops,1)),
    wrs += (slice(0,t,1),slice(0,v,1),slice(bs+ops,2*bs+ops,1),slice(bs+ops,2*bs+ops,1)),
    brs = tuple()
    for wr in wrs:
        brs += create_boundary_regions(wr,tarr.shape,ops),
    for i,br in enumerate(brs):
        reg_edge_comm(tarr,ops,br,wrs[i])
    tarr[0,0,ops:x-ops,0:ops]-=tarr[0,0,ops:x-ops,x-2*ops:x-ops]
    tarr[0,0,0:ops,ops:x-ops]-=tarr[0,0,x-2*ops:x-ops,ops:x-ops]
    assert (tarr[0,0,:,0:ops]==0).all()
    assert (tarr[0,0,0:ops,:]==0).all()

def test_decomp():
    """Use this function to test decomp"""
    #Properties
    gamma = 1.4
    #Analytical properties
    cvics = vics()
    cvics.Shu(gamma)
    initial_args = cvics.get_args()
    X = cvics.L
    Y = cvics.L
    #Dimensions and steps
    npx = 64
    npy = 64
    dx = X/npx
    dy = Y/npy
    #Time testing arguments
    t0 = 0
    t_b = 1
    dt = 0.01
    targs = (t0,t_b,dt)
    # Creating initial vortex from analytical code
    initial_vortex = vortex(cvics,X,Y,npx,npy,times=(0,1))
    flux_vortex = convert_to_flux(initial_vortex,gamma)[0]
    tarr = np.ones(flux_vortex.shape)
    #GPU Arguments
    gpu_source = "/home/walkanth/pysweep/src/equations/euler.h"
    cpu_source = "/home/walkanth/pysweep/src/equations/euler.py"
    ops = 2 #number of atomic operations
    tso = 2 #RK2
    #File args
    decomp_name = "./results/decomp"
    #Changing arguments
    block_size = 16
    affinity = 1
    gargs = (t0,t_b,dt,dx,dy,gamma)
    swargs = (tso,ops,block_size,affinity,gpu_source,cpu_source)
    ct = decomp(flux_vortex,gargs,swargs,filename="./tests/test_results/decomp")
    hdf5_file = h5py.File(filename, 'w', driver='mpio', comm=comm)



test_decomp()
