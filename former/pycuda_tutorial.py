#Programmer: Anthony Walker
#This is a tutorial using pycuda

#Imports
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import time

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



#Array
time_len = 2
block = (256,256,2)
a =  np.ones(block,dtype=np.float32)


#Getting cuda source
source_code = source_code_read("./src/sweep/sweep.h")
# print(source_code)
# source_code += "\n"+source_code_read("/home/walkanth/pysweep/csrc/euler_source.cu")
#Creating cuda source
mod = SourceModule(source_code)

#Constants
dt = np.array([0.01,],dtype=np.float32)
mss = np.array([100,],dtype=int)

dt_ptr,_ = mod.get_global("dt")
mss_ptr,_ = mod.get_global("mss")

#Copying to GPU memory
cuda.memcpy_htod(mss_ptr,mss)
cuda.memcpy_htod(dt_ptr,dt)

#Executing cuda source
func = mod.get_function("dummy_fcn")
func(cuda.InOut(a),grid=(1,1,1),block=(1024,1,1))
# for x in a:
#     for y in x:
#         print("new time level")
#         for t in y:
#             print(t)
