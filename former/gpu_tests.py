import numpy
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import pycuda.gpuarray as gpuarray
nt = 10
d1 = 256
d2 = 256

def G1():
    total_time = 0.0
    for i in range(nt):
        start = time.time()
        a = numpy.random.randn(d1,d2)
        a = a.astype(numpy.float32)
        a_gpu = cuda.mem_alloc(a.nbytes)
        cuda.memcpy_htod(a_gpu, a)
        mod = SourceModule("""
          __global__ void doublify(float *a)
          {
            int idx = threadIdx.x + threadIdx.y*4;
            for(int i=0;i<100000;i++)
            {
                a[idx] *= 1.01;
            }

          }
          """)
        func = mod.get_function("doublify")
        func(a_gpu, block=(4,4,1))
        a_doubled = numpy.empty_like(a)
        cuda.memcpy_dtoh(a_doubled, a_gpu)
        stop = time.time()
        total_time+=(stop-start)
    return total_time/nt

def G2():
    total_time = 0.0
    for i in range(nt):
        start = time.time()
        a_gpu = gpuarray.to_gpu(numpy.random.randn(d1,d2).astype(numpy.float32))
        for i in range(100000):
            a_gpu *= 1.01
        stop = time.time()
        total_time+=(stop-start)
    return total_time/nt

def G3():
    total_time = 0.0
    for i in range(nt):
        start = time.time()
        a_gpu = gpuarray.to_gpu(numpy.random.randn(d1,d2).astype(numpy.float32))
        gpu_tf(a_gpu)
        stop = time.time()
        total_time+=(stop-start)
    return total_time/nt

def gpu_tf(arr):
    for i in range(100000):
        arr *= 1.01

if __name__ == "__main__":
    print(G1())
    print(G2())
    print(G3())
