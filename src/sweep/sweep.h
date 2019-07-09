
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
    int gid = getGlobalIdx_2D_2D();  //Getting 2D global index
    int sgid_shift = blockDim.x*blockDim.y;
    int var_shift = sgid_shift*gridDim.x*gridDim.y;//This is the variable used to shift between values
    int time_shift = var_shift*nv; //Use this variable to shift in time and for sgid
    int sgid = gid%(sgid_shift); //Shared global index
    // printf("%d,%d, %d, %d\n",gid,sgid,var_shift,time_shift );
    // printf("%d,%d,%d,%d\n",gid,gid+1*var_shift,gid+2*var_shift,gid+3*var_shift );
    // printf("%f,%f,%f,%f\n",state[gid],state[gid+var_shift],state[gid+var_shift*2],state[gid+var_shift*3]);

    // Filling shared state array with variables initial variables
    __syncthreads(); //Sync threads here to ensure all initial values are copied

    int k = 0;
    for (int i = 0; i < nv; i++)
    {
        shared_state[sgid+i*sgid_shift] = state[gid+i*var_shift];
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    // Solving step function

    step(shared_state,sgid,k);

    __syncthreads();   //Sync threads after solving
    // Place values back in original matrix
    // for (int j = 0; j < nv; j++)
    // {
    //     state[gid+shift*(k+1)+j]=shared_state[sgid+j];
    // }

}
