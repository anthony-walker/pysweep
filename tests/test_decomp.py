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

def test_decomp_vortex():
    pass
    # estr = "mpiexec -n 4 python ./src/pst.py dtest "
    # estr += "-b 10 -o 1 --tso 2 -a 1 -g \"./src/equations/eqt.h\" -c \"./src/equations/eqt.py\" "
    # estr += "--hdf5 \"./dtest\" -nx 40 -ny 40"
    # os.system(estr)
    # test_file = "./dtest.hdf5"
    # hdf5_file = h5py.File(test_file, 'r')
    # hdf5_data_set = hdf5_file['data']
    # for element in hdf5_data_set[1:10]: #Last couple steps arent hit
    #     assert sum(element[0,:,:]-hdf5_data_set[0,0,:,:]).all() == 0
    # hdf5_file.close()
    # os.system("rm "+test_file)


def test_decomp_pst():
    """Test both affinity of 1 and 0"""
    GRB = True
    for i in range(2):
        GRB = not GRB
        if GRB:
            aff = 1
        else:
            aff = 0
        estr = "mpiexec -n 4 python ./src/pst.py dtest "
        estr += "-b 10 -o 1 --tso 2 -a " +str(aff)+ " -g \"./src/equations/eqt.h\" -c \"./src/equations/eqt.py\" "
        estr += "--hdf5 \"./dtest\" -nx 40 -ny 40"
        os.system(estr)
        test_file = "./dtest.hdf5"
        hdf5_file = h5py.File(test_file, 'r')
        hdf5_data_set = hdf5_file['data']
        for element in hdf5_data_set[1:10]: #Last couple steps arent hit
            assert sum(element[0,:,:]-hdf5_data_set[0,0,:,:]).all() == 0
        hdf5_file.close()
        os.system("rm "+test_file)
