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

def pm(arr,i):
    for item in arr[i,0,:,:]:
        sys.stdout.write("[ ")
        for si in item:
            sys.stdout.write("%.1f"%si+", ")
        sys.stdout.write("]\n")

def test_UpPyramid():
    """Use this function to test the GPU edge communication."""
    estr = "colorcode mpiexec -n 4 python ./src/pst.py stest "
    estr += "-b 10 -o 1 --tso 2 -a 0.5 -g \"./src/equations/eqt.h\" -c \"./src/equations/eqt.py\" "
    estr += "--hdf5 \"./results/stest\" -nx 40 -ny 40"
    # os.system(estr)

    #Sod Shock BC's
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
    # SPLITX = SPLITX if SPLITX%2==0 else SPLITX-1
    SPLITY = int(BS[1]/2)   #Split computation shift

    sarr = np.zeros((ts,v,2*x+2*OPS,2*x+2*OPS),dtype=dType)
    WR = (slice(0,ts,1),slice(0,v,1),slice(OPS,OPS+2*x,1),slice(OPS,OPS+2*x,1))
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
    sarr[WR] = carr[:,:,OPS:-OPS,OPS:-OPS]
    ssb = np.zeros((2,v,BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes
    #Source mods
    g_mod_2D = build_gpu_source("./src/equations/eqt.h")
    c_mod_2D = build_cpu_source("./src/equations/eqt.py")
    #Setting globals
    c_mod_2D.set_globals(True,g_mod_2D,*(t0,tf,dt,dx,dy,gamma))
    c_mod_2D.set_globals(False,g_mod_2D,*(t0,tf,dt,dx,dy,gamma))
    #Swept globals
    NV = carr.shape[1]
    SGIDS = (BS[0]+2*OPS)*(BS[1]+2*OPS)
    STS = SGIDS*NV #Shared time shift
    VARS =  carr.shape[2]*carr.shape[3]
    TIMES = VARS*NV
    const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"SPLITX":SPLITX,"SPLITY":SPLITY,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})
    swept_constant_copy(g_mod_2D,const_dict)
    #Getting function
    pargs = (g_mod_2D,True,BS,GRD,None,OPS,TSO,ssb)
    UpPyramid(sarr,carr,WR,tuple(),up_sets,0,pargs)
    for i in range(v):
        for ts, set in enumerate(up_sets[1:],start=1):
            tsum = 0
            for idx in set:
                tsum+=carr[ts,i,idx[0],idx[1]]
            assert 2*tsum==len(set)
    
