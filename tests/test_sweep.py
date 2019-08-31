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

def test_edge_comm():
    """Use this function to test the GPU edge communication."""
    estr = "colorcode mpiexec -n 4 python ./src/pst.py swept "
    estr += "-b 10 -o 2 --tso 2 -a 0.5 -g \"./src/equations/eqt.h\" -c \"./src/equations/eqt.py\""
    estr += "--hdf5 \"./results/stest\" -nx 40 -ny 40"
    os.system(estr)

test_edge_comm()
