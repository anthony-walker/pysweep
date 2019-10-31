#Programmer: Anthony Walker
#Use the functions in this file to test the decomposition code.
import sys, os, h5py
sys.path.insert(0, './pysweep')
import numpy as np
from decomposition import decomp_functions

def test_reg_edge_comm(args=None):
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


def test_decomp(args=None):
    GRB=True
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
            tsarr = 3
            #Shared arr
            sarr = np.zeros((tsarr,v,2*x+2*OPS,2*x+2*OPS),dtype=dType)
            sarr[:TSO,:,:,:] = 1
            shared_shape = sarr.shape

            regions = (slice(0,tsarr,1),slice(0,v,1),slice(OPS,OPS+2*x,1),slice(OPS,OPS+2*x,1))
            regions = (decomp_functions.create_read_region(regions,OPS),regions) #Creating read region
            brs = decomp_functions.create_boundary_regions(regions[1],shared_shape,OPS)
            decomp_functions.reg_edge_comm(sarr,OPS,brs,regions[1])
            carr = np.copy(sarr[regions[0]])
            ssb = np.zeros((2,v,BS[0]+2*OPS,BS[1]+2*OPS),dtype=dType).nbytes
            WR = regions[1]
            wr = regions[1]
            #File
            test_file = "testfile.hdf5"
            hdf5_file = h5py.File(test_file, 'w')
            hdf5_data_set = hdf5_file.create_dataset("data",(ts+1,v,2*x,2*x),dtype=dType)
            hregion = (WR[1],slice(WR[2].start-OPS,WR[2].stop-OPS,1),slice(WR[3].start-OPS,WR[3].stop-OPS,1))
            hdf5_data_set[0,hregion[0],hregion[1],hregion[2]] = sarr[TSO-1,WR[1],WR[2],WR[3]]
            #Source mods
            g_mod_2D = decomp_functions.build_gpu_source("./pysweep/equations/eqt.h")
            c_mod_2D = decomp_functions.build_cpu_source("./pysweep/equations/eqt.py")
            #Setting globals
            c_mod_2D.set_globals(True,g_mod_2D,*(t0,tf,dt,dx,dy,gamma))
            c_mod_2D.set_globals(False,c_mod_2D,*(t0,tf,dt,dx,dy,gamma))
            #Swept globals
            NV = carr.shape[1]
            SGIDS = (BS[0]+2*OPS)*(BS[1]+2*OPS)
            STS = SGIDS*NV #Shared time shift
            VARS =  carr.shape[2]*carr.shape[3]
            TIMES = VARS*NV
            const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"OPS":OPS,"TSO":TSO,"STS":STS})
            decomp_constant_copy(g_mod_2D,const_dict)
            decomp_set = create_decomp_sets(sarr[regions[0]].shape,OPS)
            #Getting function
            mods = {True:g_mod_2D,False:c_mod_2D}
            wb = 0
            ct = 1
            for i in range(0,10):
                local_array = np.copy(sarr[regions[0]])
                decomposition(mods[GRB],local_array,GRB, BS, GRD,regions[1],decomp_set,sarr,OPS,i,TSO,ssb)
                reg_edge_comm(sarr,OPS,brs,regions[1])
                #Writing Data after it has been shifted
                if (i+1)%TSO==0:
                    for j in range(NV):
                        assert (sarr[1,j,WR[2],WR[3]]-sarr[0,j,WR[2],WR[3]]==2).all()
                    sarr[0,wr[1],wr[2],wr[3]] = sarr[1,wr[1],wr[2],wr[3]]
                    hdf5_data_set[ct,hregion[0],hregion[1],hregion[2]] = sarr[0,regions[1][1],regions[1][2],regions[1][3]]
                    ct+=1
            hdf5_file.close()

def test_decomp_write(args=None):
    estr = "mpiexec -n 4 python ./pysweep/pst.py dtest "
    estr += "-b 10 -o 2 --tso 2 -a 0.5 -g \"./pysweep/equations/eqt.h\" -c \"./pysweep/equations/eqt.py\" "
    estr += "--hdf5 \"./dtest\" -nx 40 -ny 40"
    os.system(estr)
    test_file = "./dtest.hdf5"
    hdf5_file = h5py.File(test_file, 'r')
    hdf5_data_set = hdf5_file['data']
    for i in range(1,len(hdf5_data_set[:,0,:,:])):
        assert (hdf5_data_set[i,0,:,:]-hdf5_data_set[i-1,0,:,:]==2).all()
    os.system("rm "+test_file)

def test_decomp_vortex(args=(2,0.01,40,0,10,10,4)):
    decomp_file = "\"./pysweep/tests/data/decomp_vortex\""
    sfp = "./pysweep/tests/data/decomp_vortex.hdf5"
    os.system("rm "+sfp)
    tf,dt,npx,aff,X,blks,nps = args
    npy=npx
    Y=X
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)

    if not os.path.isfile(sfp):
    #Create data using solver
        estr = "mpiexec -n "+str(nps)+" python ./pysweep/pst.py standard_vortex "
        estr += "-b "+str(blks)+" -o 2 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/euler.h\" -c \"./pysweep/equations/euler.py\" "
        estr += "--hdf5 " + decomp_file + pts +time_str
        os.system(estr)

def test_decomp_hde(args=(8,40,0.75,10,0.24,5,10,4)):
    savepath = "./decomp_hde_plot"
    decomp_file = "\"./pysweep/tests/data/decomp_hde\""
    sfp = "./pysweep/tests/data/decomp_hde.hdf5"
    afp = "./pysweep/tests/data/analyt_hde0.hdf5"
    analyt_file = "\"./pysweep/tests/data/analyt_hde\""
    os.system("rm "+sfp)
    tf,npx,aff,X,Fo,alpha,blks,nps = args
    npy=npx
    Y=X
    dt = Fo*(X/npx)**2/alpha
    time_str = " -dt "+str(dt)+" -tf "+str(tf)+ " "
    pts = " -nx "+str(npx)+ " -ny "+str(npx)+" -X "+str(X)+ " -Y "+str(Y)

    if not os.path.isfile(sfp):
        #Create data using solver
        estr = "mpiexec -n "+str(nps)+"--hostfile=decomp-nodes python ./pysweep/pst.py standard_hde "
        estr += "-b "+str(blks)+" -o 1 --tso 2 -a "+str(aff)+" -g \"./pysweep/equations/hde.h\" -c \"./pysweep/equations/hde.py\" "
        estr += "--hdf5 " + decomp_file + pts +time_str + "--alpha "+str(alpha)+" -TH 373 -TL 298"
        os.system(estr)
