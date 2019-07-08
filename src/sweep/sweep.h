
/*
Programmer: Anthony Walker
This file contains cuda/c++ source code for launching the pycuda swept rule.
Other information can be found in the euler.h file
*/


//Constant Memory Values
__device__ __constant__  int mss;

__device__ __constant__  int nx;

__device__ __constant__  int ny;

__device__ __constant__  int nt;

__device__ __constant__  int nv;

__device__ __constant__  int nd;

__device__ __constant__  float dt;

__device__ int getGlobalIdx_2D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}


/*
    Creating shared array (3 Dimensions (x , y, variables))
*/
__device__ float ***Array3D(int nv)
{
    float ***shared_state = new float**[blockDim.x];
    for(int i =0; i<blockDim.x; i++){
       shared_state[i] = new float*[blockDim.y];
       for(int j =0; j<blockDim.y; j++){
           shared_state[i][j] = new float[nv];
       }
    }
    return shared_state;
}

__device__ float **Array2D(int nv)
{
    int arr_size = blockDim.x*blockDim.y;
    float **shared_array = new float*[arr_size];
    for (int i = 0; i < arr_size; i++)
    {
        shared_array[i] = new float[nv];
    }
    return shared_array;
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
    int ct = 0; //temporary current time variable
    __syncthreads(); //Sync threads here to ensure the pointer is fully allocated
    for (int i = 0; i < nv; i++)
    {
        shared_state[sgid+i] = state[gid+i];
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    // Solving function
    step(shared_state,gid);

    __syncthreads();   //Sync threads after solving
    //Place values back in original matrix
    // for (int i = 0; i < nv; i++)
    // {
    //     state[gid+i]=shared_state[gid+i];
    // }
    //
}
