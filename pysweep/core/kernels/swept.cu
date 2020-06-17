
/*
Programmer: Anthony Walker
This file contains cuda/c++ source code for launching the pycuda swept rule.
Other information can be found in the euler.h file
*/


#include "cuda.h"
#include <iostream>


#if defined(_MSC_VER)
    //  Microsoft
    #define EXPORT extern "C" __declspec(dllexport)
#elif defined(__GNUC__)
    //  GCC
    #define EXPORT extern "C" __attribute__((visibility("default")))
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #pragma warning Unknown dynamic link export semantics.
#endif

//Constant Memory Values
__device__ __constant__ const double RTWO=2;
//Swept constants
__device__ __constant__ int VARS; //shift in variables in data
__device__ __constant__ int TIMES; //shift in times in data
__device__ __constant__  int MPSS; //max pyramid swept steps
__device__ __constant__  int MOSS; //max octahedron swept steps
__device__ __constant__  int NV; //number of variables
__device__ __constant__ int OPS; //number of atomic operations
__device__ __constant__ int TSO; //Time scheme order
__device__ __constant__ int SX; //Space in x
__device__ __constant__ int SY; //Space in y
//GPU Constants
__device__ __constant__ const int LB_MIN_BLOCKS = 1;    //Launch bounds min blocks
__device__ __constant__ const int LB_MAX_THREADS = 1024; //Launch bounds max threads per block

#include PYSWEEP_GPU_SOURCE
/*
    Use this function to get the global id
*/
__device__ int get_idx(int tmod)
{
  return SY*(threadIdx.x)+threadIdx.y+blockDim.y*blockIdx.y+blockDim.x*blockIdx.x*SY+tmod*TIMES;
}

/*
    Use this function to create and return the GPU UpPyramid
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
UpPyramid(double *state, int gts)
{

    //Other quantities for indexing
    int gid = get_idx(TSO-1); //global index
    //Creating swept boundaries
    int lx = OPS; //Lower x swept bound
    int ly = OPS; // Lower y swept bound
    int ux = blockDim.x-OPS; //upper x
    int uy = blockDim.y-OPS; //upper y
    //Swept loop
    for (int k = 0; k < MPSS; k++)
    {
        //Putting sweep values into next time step
        if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
        {
            // Solving step function
            //step(state,gid,gts);
        }
        __syncthreads();
        //Updating global time step
        gts+=1;
        gid+=TIMES; //Updating gid so it works for next step
        //Update swept bounds
        ux -= OPS;
        uy -= OPS;
        lx += OPS;
        ly += OPS;
    }
}

/*
    Use this function to create and return the GPU Y-Bridge
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
YBridge(double *state, int gts)
{

    int gid = get_idx(TSO-1)+blockDim.x/2; //global index
    //Creating swept boundaries
    int lx = OPS; //Lower x swept bound
    int ux = blockDim.x-OPS; //upper x
    int ly = blockDim.y/2-OPS; // Lower y swept bound
    int uy = blockDim.y/2+OPS; //upper y
    for (int k = 0; k < MPSS; k++)
    {

        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
        {
          //step(state,gid,gts);
        }
        __syncthreads();
        //Updating global time step
        gts+=1;
        gid+=TIMES; //Updating gid so it works for next step
        //Update swept bounds
        lx+=OPS;
        ux-=OPS;
        ly-=OPS;
        uy+=OPS;
    }

}


/*
    Use this function to create and return the GPU Y-Bridge
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
YBT(double *state, int gts)
{
    int gid = get_idx(MPSS+1); //global index
    //gts-=1; //Subtract 1 so that TSO works
    //Creating swept boundaries
    int lx = OPS; //Lower x swept bound
    int ux = blockDim.x-OPS; //upper x
    int ly = blockDim.y/2-OPS; // Lower y swept bound
    int uy = blockDim.y/2+OPS; //upper y
    for (int k = 0; k < MPSS; k++)
    {
        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
        {
            //step(state,gid,gts);
            // state[gid+TIMES]=3;
        }
        __syncthreads();
        //Updating global time step
        gts+=1;
        gid+=TIMES; //Updating gid so it works for next step
        //Update swept bounds
        lx+=OPS;
        ux-=OPS;
        ly-=OPS;
        uy+=OPS;
        __syncthreads(); //Sync threads here to ensure all initial values are copied
    }

}

/*
    Use this function to create and return the GPU Y-Bridge
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
XBridge(double *state, int gts)
{
    int gid = get_idx(TSO-1); //global index
    //Creating swept boundaries
    int ly = OPS; //Lower x swept bound
    int uy = blockDim.y-OPS; //upper x
    int lx = blockDim.x/2-OPS; // Lower y swept bound
    int ux = blockDim.x/2+OPS; //upper y

    for (int k = 0; k < MPSS; k++)
    {
      // Solving step function
      if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
      {
          //step(state,gid,gts);
          // state[gid+TIMES]=2;
      }
      __syncthreads();
      //Updating global time step
      gts+=1;
      gid+=TIMES; //Updating gid so it works for next step
      //Update swept bounds
        lx-=OPS;
        ux+=OPS;
        ly+=OPS;
        uy-=OPS;
    }

}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
Octahedron(double *state, int gts)
{
    int gid = get_idx(TSO-1)-2*SY-OPS+blockDim.y/2; //global index
    //gts-=1; //Subtract 1 so that TSO works
    int TOPS = 2*OPS;
    int MDSS = MOSS-MPSS;
    //------------------------DOWNPYRAMID of OCTAHEDRON-----------------------------
    //Creating swept boundaries
    int lx =(blockDim.x+TOPS)/RTWO-OPS; //lower x
    int ly = (blockDim.y+TOPS)/RTWO-OPS; //lower y
    int ux =(blockDim.x+TOPS)/RTWO+OPS; //upper x
    int uy = (blockDim.y+TOPS)/RTWO+OPS; //upper y
    //Calculate Down Pyramid - Down Step
    for (int k = 0; k < MDSS; k++)
    {
      // Solving step function
      if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
      {
          //step(state,gid,gts);
          // state[gid+TIMES]=1;
      }
      __syncthreads();
      //Updating global time step
      gts+=1;
      gid+=TIMES; //Updating gid so it works for next step
      //Update swept bounds
      ux += OPS;
      uy += OPS;
      lx -= OPS;
      ly -= OPS;
    }

    //------------------------UPPYRAMID of OCTAHEDRON-----------------------------
    //Reassigning swept bounds
    lx = OPS; //Lower x swept bound
    ly = OPS; // Lower y swept bound
    ux = blockDim.x+OPS; //upper x
    uy = blockDim.y+OPS; //upper y
    //Swept loop
    for (int k = MDSS; k < MOSS; k++)
    {
        //Update swept bounds - It is first here so it does not do the full base
        ux -= OPS;
        uy -= OPS;
        lx += OPS;
        ly += OPS;
        __syncthreads(); //Sync threads here to ensure all initial values are copied
        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
        {
            //step(state,gid,gts);
            // state[gid+TIMES]=1;
        }
        __syncthreads();
        //Updating global time step
        gts+=1;
        gid+=TIMES; //Updating gid so it works for next step
        //Update swept bounds
    }
}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
DownPyramid(double *state, int gts)
{
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int gid = get_idx(TSO-1)+blockDim.x/2; //global index
    //gts-=1; //Subtract 1 so that TSO works
    int TOPS = 2*OPS;
    int MDSS = MOSS-MPSS;
    //------------------------DOWNPYRAMID of OCTAHEDRON-----------------------------

    //Creating swept boundaries
    int lx =(blockDim.x+TOPS)/RTWO-OPS; //lower x
    int ly = (blockDim.y+TOPS)/RTWO-OPS; //lower y
    int ux =(blockDim.x+TOPS)/RTWO+OPS; //upper x
    int uy = (blockDim.y+TOPS)/RTWO+OPS; //upper y
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    // printf("%d\n",MDSS );
    for (int k = 0; k < MDSS; k++)
    {
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
          //step(state,gid,gts);
        }
        __syncthreads();
        gid+=TIMES;
        gts += 1;   //Update gts
        //Update swept bounds
        ux += OPS;
        uy += OPS;
        lx -= OPS;
        ly -= OPS;
    }
}

//Interface functions
typedef void (*FunctionArray)(double *state, int gts); //creating type for function array
FunctionArray functions[] = {UpPyramid,XBridge,YBridge,YBT,Octahedron,DownPyramid}; //Creating global function array
long long int ARRAY_BYTES; //Size of array
int array_size;
double *globalState; //globalState array

EXPORT void CudaArrayAllocate(int arr_size)
{
  ARRAY_BYTES = arr_size * sizeof(double) ;
  cudaMalloc((void **) &globalState, ARRAY_BYTES) ;
}

EXPORT void CudaArrayFree()
{
  cudaFree(globalState);
}

//
EXPORT void CudaFunctionCall(double * state, int gts, int fcnIdx)
{
	cudaMemcpy(globalState, state, ARRAY_BYTES, cudaMemcpyHostToDevice) ;
  functions[fcnIdx]<<< 1, array_size >>>(state, gts) ;
  cudaDeviceSynchronize();
	cudaMemcpy(state, globalState, ARRAY_BYTES, cudaMemcpyDeviceToHost) ;
}

int main(int argc, char const *argv[])
{
  printf("%s\n", PYSWEEP_GPU_SOURCE);
  return 0;
}
