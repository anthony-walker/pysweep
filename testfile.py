# import os
# import sys
#
# #DataStructure and Math imports
# import math
# import numpy as np
# from collections import deque
#
#
# #GPU Imports
# try:
#     import pycuda.driver as cuda
#     import pycuda.autoinit
#     from pycuda.compiler import SourceModule
#     import pycuda.gpuarray as gpuarray
# except Exception as e:
#     print(e)
# import GPUtil
# from mpi4py import MPI
#
# #Multiprocessing Imports
# import multiprocessing as mp
# import ctypes


for x in set(((1,1),(2,2),(2,1))):
    print(x)

# def create_consts(const_dict):
#     """Use this function to global constants in the source code."""
#     rstr = """//Python Created Constants\n"""
#     cstr = "const "
#     for key in const_dict:
#         rstr += cstr+const_dict[key][0]+" "+key+" = "+str(const_dict[key][1])+";\n"
#     return rstr
#
# def source_code_read(filename):
#     """Use this function to generate a multi-line string for pycuda from a source file."""
#     with open(filename,"r") as f:
#         source = """\n"""
#         line = f.readline()
#         while line:
#             source+=line
#             line = f.readline()
#     f.closed
#     return source
#
# def create_blocks(arr,block_size):
#     """Use this function to create blocks from an array based on the given blocksize."""
#     sx = arr.shape[1]/block_size[0] #split x dimensions
#     sy = arr.shape[2]/block_size[1] #split y dimension
#     num_blocks = sx*sy  #number of blocks
#     hcs = lambda x,s: np.array_split(x,s,axis=x.shape.index(max(x.shape[1:])))   #Lambda function for creating blocks for cpu testing
#     blocks =  [item for subarr in hcs(arr,sx) for item in hcs(subarr,sy)]    #applying lambda function
#     return blocks,num_blocks
#
# #Test dims
# dims = (1,int(128),int(128),4)
#
# block_size = (32,32,1)
# gdx = int(dims[1]/block_size[0])
# gdy = int(dims[2]/block_size[1])
# t = dims[0]
# v = dims[3]
#
# #Creating test array
# arr = np.zeros(dims,dtype=np.float32)
# for i,item in enumerate(arr):
#     item[:,:,0] = i+1
#     item[:,:,1] = i-1
#     item[:,:,2] = i+1
#     item[:,:,3] = i-1
#     item[0,0,0] = 1234.00
# shared_size = arr[0,:block_size[0],:block_size[1],:].nbytes
# blocks,num_blocks = create_blocks(arr,block_size)
#
# stream1 = cuda.Stream()
# stream2 = cuda.Stream()
#
# int_cast = lambda x:np.int32(x)
# const_dict = {'nx':(int_cast,block_size[0]),'ny':(int_cast,block_size[1]),'nv':(int_cast,v),'nt':(int_cast,t),'nd':(int_cast,2)}
# #Creating source code
# source_code = source_code_read("./src/equations/euler.h")
# source_code += source_code_read("./src/sweep/sweep.h")
# source_mod = SourceModule(source_code)
#
# #Constants
# for key in const_dict:
#     c_ptr,_ = source_mod.get_global(key)
#     cuda.memcpy_htod_async(c_ptr,const_dict[key][0](const_dict[key][1]))
#
# arr = arr.astype(np.float32)
# gpu_fcn = source_mod.get_function("UpPyramid")
# gs = (gdx,gdy)
# # print(arr)
# print("-----------------------------------------------------------------------")
# gpu_fcn(cuda.InOut(arr),grid=gs, block=block_size,shared=shared_size)
# # print(arr)
