#Programmer: Anthony Walker
import sys, os
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from master import controller
#Cuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#C
from ctypes import *
#From PySweep
from analytical import *
from equations import *
from decomp import *

def test_flux():
    """Use this function to test the python version of the euler code.
    This test uses a formerly validate 1D code and computes the fluxes in each direction
    to test whether the 2D fluxes are correctly computing
    """
    #Sod Shock BC's
    t0 = 0
    tf = 1
    dt = 0.01
    dx = 0.1
    dy = 0.1
    gamma = 1.4
    leftBC = (1.0,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    num_test = np.zeros((2,4,5,5))
    for i in range(3):
        for j in range(3):
            num_test[0,:,i,j]=leftBC[:]
    for i in range(2,5):
        for j in range(2,5):
            num_test[0,:,i,j]=rightBC[:]

    Qx = np.zeros((5,3))
    Qx[:,0] = num_test[0,0,:,2]
    Qx[:,1] = num_test[0,1,:,2]
    Qx[:,2] = num_test[0,3,:,2]
    P = np.zeros(5)
    P[:] = [1,1,0.1,0.1,0.1]
    # #Get source module
    source_mod_2D = build_cpu_source("./src/equations/euler.py")
    source_mod_2D.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
    source_mod_1D = build_cpu_source("./src/equations/euler1D.py")
    iidx = (0,slice(0,4,1),2,2)
    #Testing Flux
    dfdx,dfdy = source_mod_2D.dfdxy(num_test,iidx)
    df = source_mod_1D.fv5p(Qx,P)[2]
    print(df)
    print(dfdx)
    print(dfdy)
    assert np.isclose(df[1],dfdx[1])
    assert np.isclose(df[1],dfdy[2])
    assert np.isclose(df[0],dfdx[0])
    assert np.isclose(df[0],dfdy[0])
    assert np.isclose(df[2],dfdx[3])
    assert np.isclose(df[2],dfdy[3])

test_flux()

def test_RK2_CPU():
    """Use this function to test the python version of the euler code.
    This test uses a formerly validate 1D code and computes the fluxes in each direction
    to test whether the 2D fluxes are correctly computing
    """
    #Sod Shock BC's
    t0 = 0
    tf = 1
    dt = 0.01
    dx = 0.1
    dy = 0.1
    gamma = 1.4
    leftBC = (1.0,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    num_test = np.zeros((3,4,5,5))
    for i in range(3):
        for j in range(3):
            num_test[0,:,i,j]=leftBC[:]
    for i in range(2,5):
        for j in range(2,5):
            num_test[0,:,i,j]=rightBC[:]
    Qx = np.zeros((5,3))
    Qx[:,0] = num_test[0,0,:,2]
    Qx[:,1] = num_test[0,1,:,2]
    Qx[:,2] = num_test[0,3,:,2]
    P = np.zeros(5)
    P[:] = [1,1,0.1,0.1,0.1]
    # #Get source module
    source_mod_2D = build_cpu_source("./src/equations/euler.py")
    source_mod_2D.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
    source_mod_1D = build_cpu_source("./src/equations/euler1D.py")
    iidx = (2,2),
    #Testing RK2 step 1
    f1d = source_mod_1D.RK2S1(Qx,P)
    num_test = source_mod_2D.step(num_test,iidx,0,0)
    #Checking velocities in each direction as they should be the same
    assert np.isclose(f1d[2,1], num_test[1,1,2,2])
    assert np.isclose(f1d[2,1], num_test[1,2,2,2])
    f1d = source_mod_1D.RK2S2(Qx,Qx)
    num_test[1,:,:,:] = num_test[0,:,:,:]
    #Test second step
    num_test = source_mod_2D.step(num_test,iidx,1,1)
    assert np.isclose(f1d[2,1], num_test[2,1,2,2])
    assert np.isclose(f1d[2,1], num_test[2,2,2,2])

def test_RK2_GPU():
    """Use this function to test the python version of the euler code.
    This test uses a formerly validate 1D code and computes the fluxes in each direction
    to test whether the 2D fluxes are correctly computing
    """
    #Sod Shock BC's
    t0 = 0
    tf = 1
    dt = 0.01
    dx = 0.1
    dy = 0.1
    gamma = 1.4
    leftBC = (1.0,0,0,2.5)
    rightBC = (0.125,0,0,.25)
    x = 8
    v = 4
    bs = (x,x,1)
    ops = 2
    TSO = 2
    num_test = np.zeros((3,v,x,x),dtype=np.float32)

    g_mod_2D = build_gpu_source("./src/equations/euler.h")
    c_mod_2D = build_cpu_source("./src/equations/euler.py")
    c_mod_2D.set_globals(True,g_mod_2D,*(t0,tf,dt,dx,dy,gamma))
    gpu_fcn = g_mod_2D.get_function("test_step")
    NV = v
    SGIDS = (bs[0])*(bs[0])
    STS = SGIDS*NV #Shared time shift
    VARS =  x*x
    TIMES = VARS*NV
    const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"OPS":ops,"TSO":TSO,"STS":STS})
    swept_constant_copy(g_mod_2D,const_dict)
    #Test x direction
    for i in range(int(x/2)):
        for j in range(x):
            num_test[0,:,i,j]=leftBC[:]
    for i in range(int(x/2),x):
        for j in range(x):
            num_test[0,:,i,j]=rightBC[:]
    gpu_fcn(cuda.InOut(num_test),np.int32(36),np.int32(0),block=bs)
    #1D stuff
    leftBC1 = (1.0,0,2.5)
    rightBC1 = (0.125,0,.25)
    Qx = np.zeros((5,3))
    Qx[0,:] = leftBC1[:]
    Qx[1,:] = leftBC1[:]
    Qx[2,:] = rightBC1[:]
    Qx[3,:] = rightBC1[:]
    Qx[4,:] = rightBC1[:]
    P = np.zeros(5)
    P[:] = [1,1,0.1,0.1,0.1]
    # #Get source module
    source_mod_1D = build_cpu_source("./src/equations/euler1D.py")
    f1d = source_mod_1D.RK2S1(Qx,P)
    assert np.isclose(f1d[2,1], num_test[0,1,4,4])
    #Test y direction
    for i in range(x):
        for j in range(int(x/2)):
            num_test[0,:,i,j]=leftBC[:]
    for i in range(x):
        for j in range(int(x/2),x):
            num_test[0,:,i,j]=rightBC[:]
    gpu_fcn(cuda.InOut(num_test),np.int32(36),np.int32(0),block=bs)
    assert np.isclose(f1d[2,1], num_test[0,2,4,4])


def test_hde_cpu():
    tf = 0.5
    dt = 0.1
    t0 = 0
    npx=npy= 20
    aff = 0
    ops = 1
    X=10
    Y=10
    alpha = 25e-6
    dx = X/npx
    dy = X/npy
    R=1
    Th = 373
    Tl = 298
    lt =10

    state = TIC(npx,npy,X,X,R,Th,Tl,lt)
    SM = build_cpu_source("./src/equations/hde.py")
    SM.set_globals(False,SM,*(t0,tf,dt,dx,dy,alpha))
    iidx = [(i,j) for i in range(ops,npx-ops,1) for j in range(ops,npy-ops,1)]
    state = SM.step(state,iidx,0,0)
