    //Programmer: Anthony Walker
    //This file contains cuda/c++ source code for launching the pycuda swept rule.

    // #include<cstdio>

  //Constants

  __device__ __constant__  int mss;

  __device__ __constant__  float dt;

  __device__ __constant__  int vlen;

  __global__ void UpPyramid(float *state)
  {
    // __shared__ float * a_shared[blockDim.x][blockDim.y];
    // printf("(%f, %d)", dt,mss);
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    int ct = 1;
    for (int i = 0; i < 4; i++)
    {
        ct *= blockDim.x;
    }
    ct = 0;
    printf("( %d %f)",threadId,state[threadId+ct]);
    // for(int i=4;i<=12;i+=4) //Time loop
    // {
    //     a[threadId+i] += a[threadId+iPrev];
    //     iPrev = i;
    // }
  }


  __global__ void dummy_fcn(float *a)
  {
    // __shared__ float* shared_state[nx][ny];
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    a[threadId] = a[threadId]*a[threadId];
  }



  // __global__ void upTriangle(states *state, const int tstep)
  // {
  // 	// extern __shared__ states sharedstate[];
  //
  // 	int gid = blockDim.x * blockIdx.x + threadIdx.x; //Global Thread ID
  // 	int tidx = threadIdx.x; //Block Thread ID
  //     int mid = blockDim.x >> 1;
  //
  //     // Using tidx as tid is kind of confusing for reader but looks valid.
  //
  // 	sharedstate[tidx] = state[gid + 1];
  //
  //     __syncthreads();
  //
  // 	for (int k=1; k<mid; k++)
  // 	{
  // 		if (tidx < (blockDim.x-k) && tidx >= k)
  // 		{
  //             stepUpdate(sharedstate, tidx, tstep + k);
  // 		}
  // 		__syncthreads();
  // 	}
  //     state[gid + 1] = sharedstate[tidx];
  // }
  //
