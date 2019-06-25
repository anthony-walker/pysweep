    //Programmer: Anthony Walker
    //This file contains cuda/c++ source code for launching the pycuda swept rule.


  //Constants
  __device__ __constant__  int mss;
  __device__ __constant__ float dt;

  //Kernels
  __global__ void sweep(float *a)
  {
    // __shared__ float * a_shared[blockDim.x][blockDim.y];
    // printf("(%f, %d)", dt,mss);
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int iPrev = 0;
    for(int i=4;i<=12;i+=4) //Time loop
    {

        a[threadId+i] += a[threadId+iPrev];
        iPrev = i;
    }
    printf("%d: %0.2f\n",threadId,a[threadId]);
  }

  __global__ void dummy_fcn(float *a)
  {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int iPrev = 0;
    for(int i=4;i<=12;i+=4) //Time loop
    {

        a[threadId+i] += a[threadId+iPrev];
        iPrev = i;
    }
    printf("%d: %0.2f\n",threadId,a[threadId]);
  }
