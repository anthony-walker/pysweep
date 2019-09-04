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

//Standard decomposition constants
__device__ __constant__ int SGIDS; //shift in shared data
__device__ __constant__ int VARS; //shift in variables in data
__device__ __constant__ int TIMES; //shift in times in data
__device__ __constant__  int NV; //number of variables
__device__ __constant__ int OPS; //number of atomic operations
__device__ __constant__ int TSO; //Time scheme order
__device__ __constant__ int STS;
//GPU Constants
__device__ __constant__ const int LB_MIN_BLOCKS = 1;    //Launch bounds min blocks
__device__ __constant__ const int LB_MAX_THREADS = 1024; //Launch bounds max threads per block

//!!(@#

__device__ int getGlobalIdx_2D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}


__device__ int get_sgid(int tidx, int tidy)
{
    return tidy + (blockDim.y+2*OPS)*tidx;
}

/*
    Use this function to get the global id
*/
__device__ int get_gid()
{
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int M = gridDim.y*blockDim.y+2*OPS;
    return M*(OPS+threadIdx.x)+OPS+threadIdx.y+blockDim.y*blockIdx.y+blockDim.x*blockIdx.x*M;
}

/*
    Use this function for testing points by thread indices
*/
__device__ void test_gid(int tdx,int tdy)
{
    if (tdx == threadIdx.x && tdy == threadIdx.y)
    {
        int bid = blockIdx.x + blockIdx.y * gridDim.x;
        int M = gridDim.y*blockDim.y+2*OPS;
        int gid = M*(OPS+threadIdx.x)+OPS+threadIdx.y+blockDim.y*blockIdx.y+blockDim.x*blockIdx.x*M;
        printf("%d\n", gid);
    }
}

/*
    Use this function to communicate edges more effectively
*/
__device__
void shared_state_fill(float * shared_state, float * state, int smod, int tmod)
{
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+smod*STS; //Shared global index
    int gid = get_gid()+tmod*TIMES; //global index
    int bdy = blockDim.y+2*OPS;
    int gbdy = blockDim.y*gridDim.y+2*OPS;
    int lsgid = sgid-bdy;
    int usgid = sgid+bdy;
    int lgid = gid-gbdy;
    int ugid = gid+gbdy;
    for (int j = 0; j < NV; j++)
    {
        shared_state[lsgid+j*SGIDS] = state[lgid+j*VARS];
        shared_state[usgid+j*SGIDS] = state[ugid+j*VARS];
        shared_state[lsgid-OPS+j*SGIDS] = state[lgid-OPS+j*VARS];
        shared_state[usgid+OPS+j*SGIDS] = state[ugid+OPS+j*VARS];
        shared_state[lsgid+OPS+j*SGIDS] = state[lgid+OPS+j*VARS];
        shared_state[usgid-OPS+j*SGIDS] = state[ugid-OPS+j*VARS];
    }
}

/*
    Use this function to clear out former values in the shared array. (Zero them)
*/
__device__
void shared_state_clear(float * shared_state)
{
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy); //Shared global index
    int bdx = blockDim.x+2*OPS;
    int bdy = blockDim.y+2*OPS;
    int lsgid = sgid-bdx;
    int usgid = sgid+bdx;
    for (int i = 0; i < TWO; i++)
    {
        for (int j = 0; j < NV; j++)
        {
            shared_state[lsgid+STS*i+j*SGIDS] = 0;
            shared_state[usgid+STS*i+j*SGIDS] = 0;
            shared_state[lsgid-OPS+STS*i+j*SGIDS] = 0;
            shared_state[usgid+OPS+STS*i+j*SGIDS] = 0;
            shared_state[lsgid+OPS+STS*i+j*SGIDS] = 0;
            shared_state[usgid-OPS+STS*i+j*SGIDS] = 0;
        }
    }
}

__device__
void print_sarr(float * shared_state,bool test_bool,int smod)
{
    int bdx = blockDim.x+OPS;
    int bdy = blockDim.y+OPS;
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+smod*STS; //Shared global index
    int id;
    __syncthreads();
    if (test_bool)
    {
      printf("%s\n", "---------------------------------------");
        for (int i = 0; i < blockDim.x; i++)
        {   printf("%s","[" );
            for (int j = 0; j < blockDim.y; j++)
            {
                id = sgid+i*(bdx+OPS)+j;
                printf("%0.1f, ",shared_state[id],id);
            }
            printf("%s\n","]" );
        }
        printf("%s\n", "---------------------------------------");
    }
    __syncthreads();
}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
Decomp(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_clear(shared_state);
    __syncthreads();
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int gid = get_gid()+TIMES; //global index
    bool tb = threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0;
    //Communicating edge values to shared array
    shared_state_fill(shared_state,state,ZERO,(gts%TSO));
    shared_state_fill(shared_state,state,ONE,(TSO-ONE));
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    // Solving step function
    step(shared_state,sgid,gts);
    // Place values back in original matrix
    for (int j = 0; j < NV; j++)
    {
    state[gid+j*VARS+TIMES]=shared_state[sgid+j*SGIDS];
    }
    __syncthreads();
}
