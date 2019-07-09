
/*
Programmer: Anthony Walker
This file contains cuda/c++ source code for launching the pycuda swept rule.
Other information can be found in the euler.h file
*/


//Constant Memory Values
__device__ __constant__  int mss=8;

__device__ __constant__  int nv;

__device__ __constant__  float dt;

__device__ int getGlobalIdx_2D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// /*
//     Use this function to create and return the GPU UpPyramid
// */
__global__ void UpPyramid(float *state)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    //Creating indexing
    int gid = getGlobalIdx_2D_2D()*nv;  //Getting 2D global index //Adjusting gid for number of variables
    int shift = blockDim.x*blockDim.y*nv; //Use this variable to shift in time and for sgid
    int sgid = gid%shift; //Shared global index
    // Filling shared state array with variables initial variables
    __syncthreads(); //Sync threads here to ensure all initial values are copied

    int k = 0;
    for (int i = 0; i < nv; i++)
    {
        shared_state[sgid+i] = state[gid+shift*k+i];
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    // Solving step function

    step(shared_state,sgid,k);

    __syncthreads();   //Sync threads after solving
    // Place values back in original matrix
    for (int j = 0; j < nv; j++)
    {
        state[gid+shift*(k+1)+j]=shared_state[sgid+j];
    }

}
