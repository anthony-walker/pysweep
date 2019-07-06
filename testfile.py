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
x = 4
y = 4
t = 5
v = 4
#Test grid dims
gdx = 2
gdy = 2
#Creating test array
arr = np.zeros((t,x,y,v),dtype=np.float32)

for i,item in enumerate(arr):
    item[:,:,0] = i+1
    item[:,:,1] = i-1
    item[:,:,2] = i
    item[:,:,3] = i-1

    # item[0,2,:] = 30
    # item[0,3,:] = 40
    # item[0,1,0] = 10
    # item[0,1,1] = 20
    # item[0,1,2] = 30
    # item[0,1,3] = 40
    # print("--------------------")
    # print(item)
    # print("--------------------")
# print(arr)
arr[0,0,0,1]=1234.123

int_cast = lambda x:np.int32(x)
const_dict = {'nx':(int_cast,x),'ny':(int_cast,y),'nv':(int_cast,v),'nt':(int_cast,t),'nd':(int_cast,2)}
# source_code = create_consts(const_dict)
source_code = source_code_read("./src/sweep/sweep.h")
# print(source_code)
source_mod = SourceModule(source_code)
#Constants
for key in const_dict:
    c_ptr,_ = source_mod.get_global(key)
    cuda.memcpy_htod(c_ptr,const_dict[key][0](const_dict[key][1]))
#Copying to GPU memory
# conv = lambda x:np.array([x,],dtype=ctypes.c_int)


x = int_cast(x)


arr = arr.astype(np.float32)
# print(arr)
arr_gpu = cuda.mem_alloc(arr.nbytes)
cuda.memcpy_htod(arr_gpu, arr)

gpu_fcn = source_mod.get_function("UpPyramid")
bs = (int(x),int(y),1)
gs = (gdx,gdy)
print(gs, bs)
gpu_fcn(arr_gpu,grid=(1,1), block=bs)
arr_doubled = np.empty_like(arr)
cuda.memcpy_dtoh(arr_doubled, arr_gpu)
