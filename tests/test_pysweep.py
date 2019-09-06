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
from sweep import *
import numpy as np
#Cuda imports
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import matplotlib as mpl
mpl.use("Tkagg")
import matplotlib.pyplot as plt
from matplotlib import cm
from collections.abc import Iterable
import matplotlib.animation as animation

def pm(arr,i):
    for item in arr[i,0,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write("%.1f"%si+", ")
        sys.stdout.write("]\n")

def test_sweep(GRB = True):
    """Use this function to test the UpPyramid communication."""
    for x in (10,16):
        for i in range(2):
            GRB = not GRB
            t0 = 0
            tf = 1
            dt = 0.01
            dx = 0.1
            dy = 0.1
            gamma = 1.4
            v = 4
            ts = 40
            BS = (x,x,1)
            OPS = 2
            TSO = 2
            GRD = (2,2)
            dType = np.float32
            up_sets = create_up_sets(BS,OPS)
            down_sets = create_down_sets(BS,OPS)[:-1]
            oct_sets = down_sets+up_sets[1:]
            MPSS = len(up_sets)
            MOSS = len(oct_sets)
            SPLITX = int(BS[0]/2)   #Split computation shift - add OPS
            SPLITY = int(BS[1]/2)   #Split computation shift
            IEP = 1 if MPSS%2==0 else 0 #This accounts for even or odd MPSS
            #Shared arr
            sarr = np.zeros((MOSS+TSO+1,v,2*x+2*OPS+SPLITX,2*x+2*OPS+SPLITY),dtype=dType)
            shared_shape = sarr.shape
            carr = np.zeros((MOSS+TSO+1,v,2*x+2*OPS,2*x+2*OPS),dtype=dType)
            WR = (slice(0,ts,1),slice(0,v,1),slice(OPS,OPS+2*x,1),slice(OPS,OPS+2*x,1))
            sarr[WR] = carr[:,:,OPS:-OPS,OPS:-OPS]
            ssb = np.zeros((2,v,BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes
            carr[TSO-1,:,:,:] = 1
            #File
            test_file = "testfile.hdf5"
            hdf5_file = h5py.File(test_file, 'w')
            hdf5_data_set = hdf5_file.create_dataset("data",(ts+1,v,2*x,2*x),dtype=dType)
            hregion = (WR[1],slice(WR[2].start-OPS,WR[2].stop-OPS,1),slice(WR[3].start-OPS,WR[3].stop-OPS,1))
            hdf5_data_set[0,hregion[0],hregion[1],hregion[2]] = sarr[TSO-1,WR[1],WR[2],WR[3]]
            #Regions
            bridge_sets, bridge_slices = create_bridge_sets(BS,OPS,MPSS)
            RR = create_read_region(WR,OPS)   #Create read region
            SRR,SWR,XR,YR = create_shift_regions(RR,SPLITX,SPLITY,shared_shape,OPS)  #Create shifted read region
            BDR,ERS = create_boundary_regions(WR,SPLITX,SPLITY,OPS,shared_shape,bridge_slices)
            wxt = create_standard_bridges(XR,OPS,bridge_slices[0],shared_shape,BS)
            wyt = create_standard_bridges(YR,OPS,bridge_slices[1],shared_shape,BS)
            wxts = create_shifted_bridges(YR,OPS,bridge_slices[0],shared_shape,BS)
            wyts = create_shifted_bridges(XR,OPS,bridge_slices[1],shared_shape,BS)
            #Source mods
            g_mod_2D = build_gpu_source("./src/equations/eqt.h")
            c_mod_2D = build_cpu_source("./src/equations/eqt.py")
            #Setting globals
            c_mod_2D.set_globals(True,g_mod_2D,*(t0,tf,dt,dx,dy,gamma))
            c_mod_2D.set_globals(False,c_mod_2D,*(t0,tf,dt,dx,dy,gamma))
            #Swept globals
            NV = carr.shape[1]
            SGIDS = (BS[0]+2*OPS)*(BS[1]+2*OPS)
            STS = SGIDS*NV #Shared time shift
            VARS =  carr.shape[2]*carr.shape[3]
            TIMES = VARS*NV
            const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"SPLITX":SPLITX,"SPLITY":SPLITY,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})
            swept_constant_copy(g_mod_2D,const_dict)
            CRS = create_blocks_list(carr.shape,BS,OPS)
            #Getting function
            mods = {True:g_mod_2D,False:c_mod_2D}
            pargs = (mods[GRB],GRB,BS,GRD,CRS,OPS,TSO,ssb)
            wb = 0
            cwt = 1
            UpPyramid(sarr,carr,WR,BDR,up_sets,wb,pargs)
            wb+=1
            #Bridge
            xarr = np.copy(sarr[XR])
            yarr = np.copy(sarr[YR])
            Bridge(sarr,xarr,yarr,wxt,wyt,bridge_sets,wb,pargs) #THis modifies shared array
            # First Octahedron test
            larr = np.copy(sarr[SRR])
            for GST in range(4):
                Octahedron(sarr,larr,SWR,tuple(),oct_sets,wb,pargs)
                # print(sarr[0:,0,x,x])
                for i in range(TSO,MPSS+TSO):
                    for j in range(NV):
                        assert (sarr[i,j,SWR[2],SWR[3]]-sarr[i-1,j,SWR[2],SWR[3]]==1).all()
                # Shifting Data Step
                edge_shift(sarr,ERS,1)
                cwt,wb = hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO,IEP)
                boundary_update(sarr,OPS,SPLITX,SPLITY)
                #Reverse bridge
                xarr = np.copy(sarr[YR]) #Regions are purposely switched here
                yarr = np.copy(sarr[XR])

                Bridge(sarr,xarr,yarr,wxts,wyts,bridge_sets,wb,pargs) #THis modifies shared array
                larr = np.copy(sarr[RR])

                Octahedron(sarr,larr,WR,BDR,oct_sets,wb,pargs)
                # print(sarr[0:,0,int(1.5*x),int(1.5*x)])
                for i in range(TSO,MPSS+TSO):
                    for j in range(NV):
                        assert (sarr[i,j,WR[2],WR[3]]-sarr[i-1,j,WR[2],WR[3]]==1).all()
                cwt,wb = hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO,IEP)
                boundary_update(sarr,OPS,SPLITX,SPLITY) #Communicate all boundaries
                #Next bridge
                xarr = np.copy(sarr[XR])
                yarr = np.copy(sarr[YR])
                Bridge(sarr,xarr,yarr,wxt,wyt,bridge_sets,wb,pargs) #THis modifies shared array
                #Down Pyramid
                larr = np.copy(sarr[SRR])

            DownPyramid(sarr,larr,SWR,down_sets,wb,pargs)
            edge_shift(sarr,ERS,1)
            for i in range(TSO,MPSS+TSO):
                for j in range(NV):
                    assert (sarr[i,j,SWR[2],SWR[3]]-sarr[i-1,j,SWR[2],SWR[3]]==1).all()
            cwt,wb = hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO,IEP)
            boundary_update(sarr,OPS,SPLITX,SPLITY) #Communicate all boundaries
            hdf5_file.close()
            os.system("rm "+test_file) #Deleting testfile


def test_sweep_write():
    estr = "mpiexec -n 8 python ./src/pst.py stest "
    estr += "-b 10 -o 2 --tso 2 -a 0.5 -g \"./src/equations/eqt.h\" -c \"./src/equations/eqt.py\" "
    estr += "--hdf5 \"./stest\" -nx 40 -ny 40"
    os.system(estr)
    test_file = "./stest.hdf5"
    hdf5_file = h5py.File(test_file, 'r')
    hdf5_data_set = hdf5_file['data']
    for i in range(1,len(hdf5_data_set[:,0,:,:])):
        assert (hdf5_data_set[i,0,:,:]-hdf5_data_set[i-1,0,:,:]==2).all()
    os.system("rm "+test_file)

def test_sweep_vortex():
    savepath = "./vortex_plot"
    swept_file = "\"./tests/data/swept\""
    sfp = "./tests/data/swept.hdf5"
    afp = "./tests/data/analyt0.hdf5"
    analyt_file = "\"./tests/data/analyt\""
    tf = 0.05
    dt = 0.001

    npx = 64
    npy = 64
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx 64 -ny 64 "

    if not os.path.isfile(afp):
        #Create analytical data
        astr = "python ./src/pst.py analytical "+time_str
        astr += "--hdf5 " + analyt_file+pts
        os.system(astr)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n 8 python ./src/pst.py swept "
        estr += "-b 16 -o 2 --tso 2 -a 0 -g \"./src/equations/euler.h\" -c \"./src/equations/euler.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str
        os.system(estr)

    #Opening the data files
    swept_hdf5 = h5py.File(sfp, 'r')
    analyt_hdf5 = h5py.File(afp, 'r')
    data = swept_hdf5['data'][:,0,:,:]
    time = np.arange(0,tf,dt)[:len(data)]
    X = 1
    Y = 1
    #Meshgrid
    xpts = np.linspace(-X,X,npx,dtype=np.float64)
    ypts = np.linspace(-Y,Y,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')

    fig, ax =plt.subplots()
    ax.set_ylim(-Y, Y)
    ax.set_xlim(-X, X)
    ax.set_title("Density")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')
    fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(-1,1,10))
    animate = lambda i: ax.contourf(xgrid,ygrid,data[i,:,:],levels=10,cmap=cm.inferno)

    if isinstance(time,Iterable):
        frames = len(tuple(time))
        anim = animation.FuncAnimation(fig,animate,frames)
        anim.save(savepath+".gif",writer="imagemagick")
    else:
        animate(time)
        fig.savefig(savepath+".png")
        plt.show()

    #Closing files
    swept_hdf5.close()
    analyt_hdf5.close()

test_sweep_vortex()
# test_sweep_write()
# test_sweep()
