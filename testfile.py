import os
import sys

#DataStructure and Math imports
import math
import numpy as np
from collections import deque


#GPU Imports
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    import pycuda.gpuarray as gpuarray
except Exception as e:
    print(e)
import GPUtil
from mpi4py import MPI

#Multiprocessing Imports
import multiprocessing as mp
import ctypes


def create_consts(const_dict):
    """Use this function to global constants in the source code."""
    rstr = """//Python Created Constants\n"""
    cstr = "const "
    for key in const_dict:
        rstr += cstr+const_dict[key][0]+" "+key+" = "+str(const_dict[key][1])+";\n"
    return rstr

def source_code_read(filename):
    """Use this function to generate a multi-line string for pycuda from a source file."""
    with open(filename,"r") as f:
        source = """\n"""
        line = f.readline()
        while line:
            source+=line
            line = f.readline()
    f.closed
    return source

#Test dims
x = 32
y = 32
t = 2
v = 4

#Test grid dims
gdx = 2
gdy = 2

#Creating test array
arr = np.zeros((t,x,y,v),dtype=np.float32)
for i,item in enumerate(arr):
    item[:,:,0] = i+1
    item[:,:,1] = i-1
    item[:,:,2] = i+1
    item[:,:,3] = i-1
    item[0,0,0] = 1234.00

int_cast = lambda x:np.int32(x)
const_dict = {'nx':(int_cast,x),'ny':(int_cast,y),'nv':(int_cast,v),'nt':(int_cast,t),'nd':(int_cast,2)}
#Creating source code
source_code = source_code_read("./src/equations/euler.h")
source_code += source_code_read("./src/sweep/sweep.h")
source_mod = SourceModule(source_code)

#Constants
for key in const_dict:
    c_ptr,_ = source_mod.get_global(key)
    cuda.memcpy_htod(c_ptr,const_dict[key][0](const_dict[key][1]))

arr = arr.astype(np.float32)
gpu_fcn = source_mod.get_function("UpPyramid")
bs = (int(4),int(4),1)
gs = (int(x/bs[0]),int(y/bs[1]))
print(arr)
print("-----------------------------------------------------------------------")
gpu_fcn(cuda.InOut(arr),grid=gs, block=bs)
print(arr)
