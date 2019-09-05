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

def pm(arr,i):
    for item in arr[i,0,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write("%.1f"%si+", ")
        sys.stdout.write("]\n")

def test_sweep():
    """Use this function to test the UpPyramid communication."""
    GRB = True
    for i in range(2):
        GRB = not GRB
        t0 = 0
        tf = 1
        dt = 0.01
        dx = 0.1
        dy = 0.1
        gamma = 1.4
        x = 10
        v = 4
        ts = 20
        BS = (x,x,1)
        OPS = 1
        TSO = 2
        GRD = (2,2)
        dType = np.float32
        up_sets = create_up_sets(BS,OPS)
        down_sets = create_down_sets(BS,OPS)
        oct_sets = down_sets+up_sets[1:]
        MPSS = len(up_sets)
        MOSS = len(oct_sets)
        SPLITX = int(BS[0]/2)   #Split computation shift - add OPS
        SPLITY = int(BS[1]/2)   #Split computation shift
        #Shared arr
        sarr = np.zeros((ts,v,2*x+2*OPS+SPLITX,2*x+2*OPS+SPLITY),dtype=dType)
        shared_shape = sarr.shape
        carr = np.zeros((ts,v,2*x+2*OPS,2*x+2*OPS),dtype=dType)
        patt = np.zeros((2*x+2*OPS+1),dtype=dType)
        for i in range(1,2*x+2*OPS+1,2):
            patt[i] = 1.0
        sb = True
        for i in range(4):
            for j in range(0,2*x+2*OPS,1):
                if sb:
                    carr[TSO-1,i,j,:] = patt[0:-1]
                else:
                    carr[TSO-1,i,j,:] = patt[1:]
                sb = not sb
        WR = (slice(0,ts,1),slice(0,v,1),slice(OPS,OPS+2*x,1),slice(OPS,OPS+2*x,1))
        sarr[WR] = carr[:,:,OPS:-OPS,OPS:-OPS]
        ssb = np.zeros((2,v,BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes
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
        #Testing Up Pyramid
        for i in range(v):
            for ts, set in enumerate(up_sets[1:],start=1):
                tsum = 0
                for idx in set:
                    tsum+=carr[ts,i,idx[0],idx[1]]
                assert 2*tsum==len(set)
        #Bridge
        xarr = np.copy(sarr[XR])
        yarr = np.copy(sarr[YR])
        Bridge(sarr,xarr,yarr,wxt,wyt,bridge_sets,wb+1,pargs) #THis modifies shared array
        # First Octahedron test
        larr = np.copy(sarr[SRR])
        for GST in range(0,3):
            Octahedron(sarr,larr,SWR,tuple(),oct_sets,wb+1,pargs)
            for i in range(v):
                for ts, set in enumerate(oct_sets,start=1):
                    tsum = 0
                    for idx in set:
                        tsum+=larr[ts,i,idx[0],idx[1]]
                    assert 2*tsum==len(set)
            # Shifting Data Step
            edge_shift(sarr,ERS,1)
            cwt,wb = hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO)
            boundary_update(sarr,OPS,SPLITX,SPLITY)
            #Reverse bridge
            xarr = np.copy(sarr[YR]) #Regions are purposely switched here
            yarr = np.copy(sarr[XR])
            Bridge(sarr,xarr,yarr,wxts,wyts,bridge_sets,wb+1,pargs) #THis modifies shared array
            # Next octahedron
            larr = np.copy(sarr[RR])
            Octahedron(sarr,larr,WR,BDR,oct_sets,wb+1,pargs)
            for i in range(v):
                for ts, set in enumerate(oct_sets,start=1):
                    tsum = 0
                    for idx in set:
                        tsum+=larr[ts,i,idx[0],idx[1]]
                    assert 2*tsum==len(set)
            cwt,wb = hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO)
            boundary_update(sarr,OPS,SPLITX,SPLITY) #Communicate all boundaries
            #Next bridge
            xarr = np.copy(sarr[XR])
            yarr = np.copy(sarr[YR])
            Bridge(sarr,xarr,yarr,wxt,wyt,bridge_sets,wb+1,pargs) #THis modifies shared array
            #Down Pyramid
            larr = np.copy(sarr[SRR])
        DownPyramid(sarr,larr,SWR,down_sets,wb+1,pargs)
        edge_shift(sarr,ERS,1)
        for i in range(v):
            for ts, set in enumerate(down_sets,start=1):
                tsum = 0
                for idx in set:
                    tsum+=larr[ts,i,idx[0],idx[1]]
                assert 2*tsum==len(set)
        hdf_swept_write(cwt,wb,sarr,WR,hdf5_data_set,hregion,MPSS,TSO)
        boundary_update(sarr,OPS,SPLITX,SPLITY) #Communicate all boundaries

        #Testing Writing
        for element in hdf5_data_set[1:-3]: #Last couple steps arent hit
            assert sum(element[0,:,:]-hdf5_data_set[0,0,:,:]).all() == 0
        hdf5_file.close()
        os.system("rm "+test_file) #Deleting testfile

def test_sweep_pst():
    estr = "mpiexec -n 4 python ./src/pst.py stest "
    estr += "-b 10 -o 1 --tso 2 -a 0.5 -g \"./src/equations/eqt.h\" -c \"./src/equations/eqt.py\" "
    estr += "--hdf5 \"./stest\" -nx 40 -ny 40"
    os.system(estr)
    test_file = "./stest.hdf5"
    hdf5_file = h5py.File(test_file, 'r')
    hdf5_data_set = hdf5_file['data']
    for element in hdf5_data_set[1:10]: #Last couple steps arent hit
        assert sum(element[0,:,:]-hdf5_data_set[0,0,:,:]).all() == 0
    os.system("rm "+test_file)


def test_sweep_vortex():

    swept_file = "\"./data/swept\""
    analyt_file = "\"./data/analyt\""
    tf = 5
    dt = 0.01
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx 64 -ny 64 "

    if not os.path.isfile("./tests/data/analyt0.hdf5"):
        #Create analytical data
        astr = "python ./src/pst.py analytical "+time_str
        astr += "--hdf5 " + analyt_file+pts
        os.system(astr)

    if not os.path.isfile("./tests/data/swept.hdf5"):
        #Create data using solver
        estr = "mpiexec -n 8 python ./src/pst.py swept "
        estr += "-b 16 -o 2 --tso 2 -a 0.75 -g \"./src/equations/euler.h\" -c \"./src/equations/euler.py\" "
        estr += "--hdf5 " + swept_file + pts +time_str
        os.system(estr)

    #Opening the data files
    swept_hdf5 = h5py.File(swept_file, 'r')
    swept_data = swept_hdf5['data']
    analyt_hdf5 = h5py.File(analyt_file, 'r')
    analyt_data = analyt_hdf5['data']
    print(swept_data[10,0,10:20,10:20]-analyt_data[10,0,10:20,10:20])
    # for element in hdf5_data_set[1:10]: #Last couple steps arent hit
    #     assert sum(element[0,:,:]-hdf5_data_set[0,0,:,:]).all() == 0
    #Closing files
    swept_hdf5.close()
    analyt_hdf5.close()
