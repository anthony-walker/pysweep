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
x = 8
y = 8
t = 5
v = 4
#Test grid dims
gdx = 2
gdy = 2
#Creating test array
arr = np.zeros((x,y,t,v))
for i in range(t):
    for row in arr:
        for col in row:
            col[i,0] = i
            col[i,1] = 2*i
            col[i,2] = 3*i
            col[i,3] = 4*i
# print(arr)

source_code = source_code_read("./src/sweep/sweep.h")
source_mod = SourceModule(source_code)
print(source_code)
#Constants
mss = np.array([4,],dtype=int)
va = np.array([4,],dtype=np.float32)
print(type(mss),type(va))
mss_ptr,_ = source_mod.get_global("mss")
v_ptr,_ = source_mod.get_global("vlen")
print(v_ptr)
# print(type(v_ptr),type(mss_ptr))
#Copying to GPU memory
print(type(mss),type(va))
cuda.memcpy_htod(mss_ptr,mss)
cuda.memcpy_htod(v_ptr,va)


#Copying global


# cuda.memcpy_htod(v_ptr,v)
arr = arr.astype(np.float32)
arr_gpu = cuda.mem_alloc(arr.nbytes)
cuda.memcpy_htod(arr_gpu, arr)
gpu_fcn = source_mod.get_function("UpPyramid")
gpu_fcn(arr_gpu,grid=(1,2), block=(int(x/gdx),int(y/gdy),1))
arr_doubled = np.empty_like(arr)
cuda.memcpy_dtoh(arr_doubled, arr_gpu)
