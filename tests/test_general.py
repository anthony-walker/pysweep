#Programmer: Anthony Walker
#This file contains test that pertain to both decomp and swept solvers
import numpy as np
import sys
import os
cwd = os.getcwd()
sys.path.insert(1,cwd+"/src")
import matplotlib as mpl
mpl.use("Tkagg")
import matplotlib.pyplot as plt
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation
from analytical import *
from equations import *
from decomp import *
import numpy as np
sys.path.insert(1,cwd+"/notipy")
from notipy import NotiPy

sm = "Hi,\nYour function run is complete.\n"
notifier = NotiPy(None,tuple(),sm,"asw42695@gmail.com",timeout=None)

def pm(arr,i):
    for item in arr[i,0,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write("%.1e"%si+", ")
        sys.stdout.write("]\n")


def test_comparison():
    """Use this function to compare the values obtain during a run of both solvers"""
    savepath = "./comp_plot"
    swept_file = "\"./tests/data/sweptc_hde\""
    sfp = "./tests/data/swept_hde.hdf5"
    afp = "./tests/data/analyt_hde0.hdf5"
    decomp_file = "\"./tests/data/decompc_hde\""
    os.system("rm "+sfp)
    tf = 1
    dt = 0.01
    npx=npy= 40
    aff = 0.5
    X=10
    Y=10
    np = 16
    bk = 10
    Th = 373
    Tl = 298
    alp = 1
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(np)+" python ./src/pst.py standard_hde "
        estr += "-b "+str(bk)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./src/equations/hde.h\" -c \"./src/equations/hde.py\" "
        estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alp)+" -TH "+str(Th)+" -TL "+str(Tl)
        os.system(estr)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(np)+" python ./src/pst.py swept_hde "
        estr += "-b "+str(bk)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./src/equations/hde.h\" -c \"./src/equations/hde.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str + "--alpha "+str(alp)+" -TH "+str(Th)+" -TL "+str(Tl)
        os.system(estr)

    #Opening the data files
    swept_hdf5 = h5py.File(sfp, 'r')
    data = swept_hdf5['data'][:,0,:,:]
    time = np.arange(0,tf,dt)[:len(data)]

    # Meshgrid
    xpts = np.linspace(-X/2,X/2,npx,dtype=np.float64)
    ypts = np.linspace(-Y/2,Y/2,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')

    fig, ax =plt.subplots()
    ax.set_ylim(-Y/2, Y/2)
    ax.set_xlim(-X/2, X/2)
    ax.set_title("Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(300,375,10))
    animate = lambda i: ax.contourf(xgrid,ygrid,data[i,:,:],levels=20,cmap=cm.inferno)
    if isinstance(time,Iterable):
        frames = len(tuple(time))
        anim = animation.FuncAnimation(fig,animate,frames=frames,repeat=False)
        anim.save(savepath+".gif",writer="imagemagick")
    else:
        animate(0)
        fig.savefig(savepath+".png")
        plt.show()

    #Closing files
    swept_hdf5.close()
