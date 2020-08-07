# Programmer: Anthony Walker
# This file contains functions used in strictly testing a GPU kernel
import os, numpy
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

def step(state,arrayTimeIndex,globalTimeStep,ops=2):
    """This function takes a step using cuda."""
    global stateCUDA, cudaStep
    nt,nv,nx,ny = state.shape
    #Copy to GPU
    cuda.memcpy_htod(stateCUDA, state)
    #Call GPU
    cudaStep(stateCUDA,numpy.int32(arrayTimeIndex),numpy.int32(globalTimeStep), block=(int(nx-2*ops),int(nx-2*ops),1))
    #Copy from GPU
    cuda.memcpy_dtoh(state,stateCUDA)

def cudaArrayMalloc(state,sourceMod):
    global stateCUDA, cudaStep
    stateCUDA = cuda.mem_alloc(state.nbytes)
    cuda.memcpy_htod(stateCUDA, state) #copy state to GPU
    cudaStep = sourceMod.get_function("stepTestInterface")

def setSweptConstants(shape,sourceMod,ops=2,its=2):
    """Use this function to obtain swept global variables.
    shape - array shape
    ops - number of points beside current point (on each side)
    its -  number of intermediate steps in the scheme
    """
    bsp = lambda x: numpy.int32(numpy.prod(shape[x:])) #block shape product returned as an integer
    icast = lambda x: numpy.int32(x) 
    globalConstants = {"NV":icast(shape[1]),'SX':icast(shape[2]),'SY':icast(shape[3]),"TIMES":bsp(1),"VARS":bsp(2),"MPSS":icast(1),"MOSS":icast(2),"OPS":icast(ops),"ITS":icast(its)}
    for key in globalConstants:
        ckey,_ = sourceMod.get_global(key)
        cuda.memcpy_htod(ckey,globalConstants[key])