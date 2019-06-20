#Programmer: Anthony Walker
#This is a tutorial using pycuda

#Imports
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import dev

dev_attrs = dev.getDeviceAttrs(0)


a =  np.ones((4,4))+1
b_num = 1
for i in range(4):
    a[0,i]=b_num
    a[3,i]=b_num
    a[i,0]=b_num
    a[i,3]=b_num

a = a.astype(np.float32)
shape = np.array(np.shape(a)).astype(np.float32)

a_gpu = cuda.mem_alloc(a.nbytes)
shape_gpu = cuda.mem_alloc(shape.nbytes)

cuda.memcpy_htod(a_gpu,a)
cuda.memcpy_htod(shape_gpu,shape)

mod = SourceModule("""
  __global__ void doublify(float *a,float *shape)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2*shape[0];
    float aV = a[idx];
    printf("idx: %d, a[idx]: %0.2f \\n",idx,aV);
  }
  """)

func = mod.get_function("doublify")
func(a_gpu,shape_gpu,block=(4,4,1))

a_doubled = np.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)

print(a_doubled)
