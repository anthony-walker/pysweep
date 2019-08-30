#Programmer: Anthony Walker
import sys
import os
cwd = os.getcwd()
sys.path.insert(1,cwd+"/src")
import matplotlib
from analytical import *
from equations import *
from sweep import *
from decomp import *
import numpy as np
#Plotting
import matplotlib as mpl
mpl.use("Tkagg")
import matplotlib.pyplot as plt
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
#Cuda
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#C
from ctypes import *

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

    assert np.isclose(df[1],dfdx[1])
    assert np.isclose(df[1],dfdy[2])
    assert np.isclose(df[0],dfdx[0])
    assert np.isclose(df[0],dfdy[0])
    assert np.isclose(df[2],dfdx[3])
    assert np.isclose(df[2],dfdy[3])

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
    num_test = source_mod_2D.step(num_test,iidx,0,1)
    #Checking velocities in each direction as they should be the same
    assert np.isclose(f1d[2,1], num_test[1,1,2,2])
    assert np.isclose(f1d[2,1], num_test[1,2,2,2])
    f1d = source_mod_1D.RK2S2(Qx,Qx)
    num_test[1,:,:,:] = num_test[0,:,:,:]
    #Test second step
    num_test = source_mod_2D.step(num_test,iidx,1,2)
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
    gpu_fcn(cuda.InOut(num_test),np.int32(36),np.int32(1),block=bs)
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
    gpu_fcn(cuda.InOut(num_test),np.int32(36),np.int32(1),block=bs)
    assert np.isclose(f1d[2,1], num_test[0,2,4,4])


def test_euler_vortex():
    """Use this function to test the python version of the euler code."""
    #Properties
    gamma = 1.4
    #Analytical properties
    cvics = vics()
    cvics.Shu(gamma)
    initial_args = cvics.get_args()
    X = cvics.L
    Y = cvics.L
    #Dimensions and steps
    npx = 96
    npy = 96
    dx = (2*X)/npx
    dy = (2*Y)/npy
    #Time testing arguments
    t0 = 0
    t_b = 0.1
    dt = 0.01
    targs = (t0,t_b,dt)
    time_steps = int((t_b-t0)/dt)   #number of time steps
    #Analytical Array
    analyt_vortex = vortex(cvics,X,Y,npx,npy,times=np.linspace(t0,t_b,time_steps+1))
    flux_vortex = convert_to_flux(analyt_vortex[:,:,:,:],gamma)
    #Numerical Array

    avs = analyt_vortex.shape
    ops = 2
    #Create flux vortex array
    num_vortex = np.zeros((time_steps+1,avs[1],avs[2]+2*ops,avs[3]+2*ops))
    num_vortex[0,:,ops:-ops,ops:-ops] = flux_vortex[0,:,:,:]
    num_vortex[0,:,:ops,:ops] = num_vortex[0,:,-2*ops:-ops,-2*ops:-ops]
    num_vortex[0,:,-ops:,-ops:] = num_vortex[0,:,ops:2*ops,ops:2*ops]
    #Testing
    #Setting globals
    source_mod_2D.set_globals(False,None,*(t0,tf,dt,dx,dy,gamma))
    #Get source module
    source_mod = build_cpu_source("./src/equations/euler.py")
    #Create iidx sets
    iidx = tuple()
    for i in range(npx):
        for j in range(npy):
            iidx+=(i+ops,j+ops),

    # Solve sets
    for ts in range(time_steps):
        num_vortex = step(num_vortex,iidx,ts)
        #Periodic bc
        num_vortex[ts+1,:,:ops,:ops] = num_vortex[ts+1,:,-2*ops:-ops,-2*ops:-ops]
        num_vortex[ts+1,:,-ops:,-ops:] = num_vortex[ts+1,:,ops:2*ops,ops:2*ops]

    # for ts in range:
    # #Testing if close
    for i in range(0,time_steps+1):
        # res = np.isclose(flux_vortex[i,:,:,:],num_vortex[i,:,ops:-ops,ops:-ops],rtol=1.e-4,atol=1.e-4)
        print(flux_vortex[i,0,30,30],num_vortex[i,0,30,30])

        # assert res.all() == True

    xpts = np.linspace(-X,X,npx,dtype=np.float64)
    ypts = np.linspace(-Y,Y,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')

    fig, ax =plt.subplots()
    ax.set_ylim(-Y, Y)
    ax.set_xlim(-X, X)
    ax.set_title("Test")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')
    fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax)
    animate = lambda i: ax.contourf(xgrid,ygrid,num_vortex[i,0,ops:-ops,ops:-ops],levels=20,cmap=cm.inferno)
    frames = len(tuple(range(time_steps+1)))
    anim = animation.FuncAnimation(fig,animate,frames)
    anim.save("test.gif",writer="imagemagick")
