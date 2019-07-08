
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
    //Creating indexing
    int gid = getGlobalIdx_2D_2D();  //Getting 2D global index
    // int idxy = gid%(blockDim.y); //Getting y index from global,
    // int idxx = gid/(blockDim.y); //Getting x index from global, Using type int truncates appropriately

    // gid*=nv;//Adjusting gid for number of variables
    int shift = blockDim.x*blockDim.y; //Use this variable to shift in time
    int sgid = gid%shift;
    //Creating flattened shared array ptr (x,y,v) length

    __shared__ float *shared_state;
    shared_state = new float[shift];

    // Filling shared state array with variables initial variables
    int ct = 0; //temporary current time variable
    __syncthreads(); //Sync threads here to ensure the pointer is fully allocated
    printf("%d\n",shared_state[sgid]);
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    // // // Solving function
    // // step(shared_state,gid);
    //
    // __syncthreads();   //Sync threads after solving
    // //Place values back in original matrix
    // for (int i = 0; i < nv; i++)
    // {
    //     state[gid+i]=shared_state[gid+i];
    // }
    // delete[] shared_state;
}
