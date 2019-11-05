#Programmer: Anthony Walker
#Use this file to generate figures for the 2D swept paper
import sys, os, h5py
sys.path.insert(0, './pysweep')
import numpy as np
import matplotlib as mpl
mpl.use("tkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gsc
from matplotlib import cm
import matplotlib.animation as animation
from node.sweep.ncore.mplambda import sweep_lambda

def myContour(args):
    i,fig,ax1,ax2,ax3,xgrid,ygrid,ddata,sdata=args
    ax1.contourf(xgrid,ygrid,sdata[i,:,:],levels=20,cmap=cm.inferno)
    ax2.contourf(xgrid,ygrid,ddata[i,:,:],levels=20,cmap=cm.inferno)
    ax3.contourf(xgrid,ygrid,abs(sdata[i,:,:]-ddata[i,:,:]),levels=20,cmap=cm.inferno)

def set_lims(fig,axes):
    """Use this function to set axis limits"""
    lim1 = 300*np.ones(3)
    lim2 = 375*np.ones(3)
    lim1[2] = 0
    lim2[2] = 1e-10
    for i,ax in enumerate(axes):
        ax.set_ylim(-5, 5)
        ax.set_xlim(-5, 5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(lim1[i],lim2[i],10))

def comp_gif(filename="vsdc0.gif"):
    decomp_file = "./pysweep/tests/data/decomp_hde.hdf5"
    swept_file =  "./pysweep/tests/data/dswept_hde.hdf5"
    #Opening the data files
    decomp_hdf5 = h5py.File(decomp_file, 'r')
    swept_hdf5 = h5py.File(swept_file, 'r')
    tsdata = swept_hdf5['data'][:,0,:,:]
    tddata = decomp_hdf5['data'][:,0,:,:]
    adj = 10
    sdata = np.zeros((int(len(tsdata)/adj),tsdata.shape[1],tsdata.shape[2]))
    ddata = np.zeros((int(len(tddata)/adj),tddata.shape[1],tddata.shape[2]))
    for i,si in enumerate(range(0,len(tsdata)-adj,adj)):
        sdata[i,:,:] = tsdata[si,:,:]
        ddata[i,:,:] = tddata[si,:,:]
    X=Y=10
    # # Meshgrid
    xpts = np.linspace(-X/2,X/2,len(sdata[0]),dtype=np.float64)
    ypts = np.linspace(-Y/2,Y/2,len(ddata[0]),dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')
    gs = gsc.GridSpec(2, 2)
    fig = plt.figure()

    ax1 = plt.subplot(gs[0,0])
    ax1.set_title("Swept")
    ax2 = plt.subplot(gs[0,1])
    ax2.set_title("Decomp")
    ax3 = plt.subplot(gs[1,:])
    set_lims(fig,(ax1,ax2,ax3))

    animate = sweep_lambda((myContour,fig,ax1,ax2,ax3,xgrid,ygrid,ddata,sdata))
    frames = len(ddata)
    anim = animation.FuncAnimation(fig,animate,frames=frames,repeat=False)
    anim.save(filename,writer="imagemagick")
    #Closing files
    swept_hdf5.close()

def create_hdf_gif(swept_file = "./pysweep/tests/data/swept_hde.hdf5",filename="shde.gif",idx=0):
    #Opening the data files
    swept_hdf5 = h5py.File(swept_file, 'r')
    tsdata = swept_hdf5['data'][:,idx,:,:]
    # tsdata /= swept_hdf5['data'][:,0,:,:]
    adj = 10
    sdata = np.zeros((int(len(tsdata)/adj),tsdata.shape[1],tsdata.shape[2]))
    for i,si in enumerate(range(0,len(tsdata)-adj,adj)):
        sdata[i,:,:] = tsdata[si,:,:]
    X=Y=10
    # # Meshgrid
    xpts = np.linspace(-X/2,X/2,len(sdata[0]),dtype=np.float64)
    ypts = np.linspace(-Y/2,Y/2,len(sdata[0]),dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')
    gs = gsc.GridSpec(2, 2)
    fig = plt.figure()

    levels = 10
    fig, ax =plt.subplots()
    ax.set_ylim(-Y, Y)
    ax.set_xlim(-X, X)
    ax.set_title(property)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # pos = ax.imshow(Zpos, cmap='Blues', interpolation='none')
    fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(-1,1,10))
    animate = lambda i: ax.contourf(xgrid,ygrid,sdata[i,:,:],levels=levels,cmap=cm.inferno)
    frames = len(sdata)
    anim = animation.FuncAnimation(fig,animate,frames=frames)
    anim.save(filename,writer="imagemagick")
    #Closing files
    swept_hdf5.close()

if __name__ == "__main__":
    # comp_gif("hdec.gif")
    create_hdf_gif("./pysweep/tests/data/decomp_hde.hdf5")
