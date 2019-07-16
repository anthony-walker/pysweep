
/*
Programmer: Anthony Walker
This file contains cuda/c++ source code for launching the pycuda swept rule.
Other information can be found in the euler.h file
*/

//Constant Memory Values
__device__ __constant__ const float ZERO=0;
__device__ __constant__ const float QUARTER=0.25;
__device__ __constant__ const float HALF=0.5;
__device__ __constant__ const float ONE=1;
__device__ __constant__ const float TWO=2;
__device__ __constant__ int SGIDS; //shift in shared data
__device__ __constant__ int VARS; //shift in variables in data
__device__ __constant__ int TIMES; //shift in times in data
__device__ __constant__  int MPSS; //max pyramid swept steps
__device__ __constant__  int MOSS; //max octahedron swept steps
__device__ __constant__  int NV; //number of variables
__device__ __constant__ int OPS; //number of atomic operations
__device__ __constant__ int SGNVS;
__device__ __constant__  float DX;
__device__ __constant__  float DY;
__device__ __constant__  float DT;
__device__ __constant__  float SPLITX;
__device__ __constant__  float SPLITY;

//!!(@#


__device__ int getGlobalIdx_2D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

/*
    Use this function to create and return the GPU UpPyramid
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
UpPyramid(float *state)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    //Creating swept boundaries
    int lxy = 0; //Lower swept bound
    int ux = blockDim.x; //upper x
    int uy = blockDim.y;//blockDim.y; //upper y
    //Creating indexing
    int gid = getGlobalIdx_2D_2D();  //Getting 2D global index

    int sgid = gid%(SGIDS); //Shared global index
    int tlx = sgid/blockDim.x;  //Location x
    int tly = sgid%blockDim.y;  //Location y
    // float fluxUpdate[4];
    // Filling shared state array with variables initial variables
    // printf("%d,%d,%d,%d,%d\n",gid,sgid,sgid+SGIDS,sgid+2*SGIDS,sgid+3*SGIDS);
    // printf("%d, %f\n",gid,state[gid]);

    for (int k = 0; k < MPSS; k++)
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*SGIDS] = state[gid+i*VARS+k*TIMES]; //Current time step
        }
        __syncthreads(); //Sync threads here to ensure all initial values are copied

        //Update swept bounds
        ux -= OPS;
        uy -= OPS;
        lxy += OPS;
        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lxy && threadIdx.y<uy && threadIdx.y>=lxy)
        {
            step(shared_state,sgid);
        }

        // Place values back in original matrix
        for (int j = 0; j < NV; j++)
        {
            state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
        }
    }

}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
Octahedron(float *state)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    //Creating swept boundaries
    int lxy = 0; //Lower swept bound
    int ux = blockDim.x; //upper x
    int uy = blockDim.y;//blockDim.y; //upper y
    //Creating indexing
    int gid = getGlobalIdx_2D_2D();  //Getting 2D global index

    int sgid = gid%(SGIDS); //Shared global index
    int tlx = sgid/blockDim.x;  //Location x
    int tly = sgid%blockDim.y;  //Location y
    // float fluxUpdate[4];
    // Filling shared state array with variables initial variables
    // printf("%d,%d,%d,%d,%d\n",gid,sgid,sgid+SGIDS,sgid+2*SGIDS,sgid+3*SGIDS);
    // printf("%d, %f\n",gid,state[gid]);

    for (int k = 0; k < MOSS; k++)
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*SGIDS] = state[gid+i*VARS+k*TIMES]; //Current time step
        }
        __syncthreads(); //Sync threads here to ensure all initial values are copied

        //Update swept bounds
        ux -= OPS;
        uy -= OPS;
        lxy += OPS;
        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lxy && threadIdx.y<uy && threadIdx.y>=lxy)
        {
            step(shared_state,sgid);
        }

        // Place values back in original matrix
        for (int j = 0; j < NV; j++)
        {
            state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
        }
    }

}
