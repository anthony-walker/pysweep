#Programmer: Anthony Walker
#This is a tutorial using pycuda

#Imports
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import dev

dev_attrs = dev.getDeviceAttrs(0)
time_len = 10

a =  np.ones((256,256,time_len),dtype=np.float32)
for i in range(time_len):
    a[:,:,i] = float(i)
# b_num = 2
# for i in range(4):
#     a[0,i]=b_num
#     a[3,i]=b_num
#     a[i,0]=b_num
#     a[i,3]=b_num

# shape = np.array(np.shape(a)).astype(np.float32)

#Allocating cuda memory
a_gpu = cuda.mem_alloc(a.nbytes)
# shape_gpu = cuda.mem_alloc(shape.nbytes)

#Copying to GPU memory
cuda.memcpy_htod(a_gpu,a)
# cuda.memcpy_htod(shape_gpu,shape)

#Creating cuda source
mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    for(int i;i<10;i++)
    {

    }
    printf("idx: %d, a[idx]: %0.2f \\n",idx,a[idx]);
  }
  """)

#Executing cuda source
func = mod.get_function("doublify")
func(a_gpu,block=(32,32,1))

#Copying from GPU
a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)

print(a_doubled)
