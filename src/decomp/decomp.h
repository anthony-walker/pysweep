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
    Use this function to communicate the initial swept edges.
*/
__device__
void edge_comm(float * shared_state, float * state, int curr_time)
{
    int M = (gridDim.y*blockDim.y+2*OPS); //Major y axis length
    int tgid = getGlobalIdx_2D_2D()%(blockDim.x*blockDim.y);
    int gid = tgid+blockIdx.y*blockDim.y+blockDim.x*blockIdx.x*M;

    if (tgid < blockDim.y+TWO*OPS)
    {
        int ntgid;
        int ngid;
        int i;
        int j;
        for (i = 0; i < OPS; i++)
        {

            for (j = 0; j < NV; j++) {
                ntgid = tgid+i*(blockDim.y+TWO*OPS)+j*SGIDS;
                ngid = gid+i*M+j*VARS+curr_time*TIMES;
                shared_state[ntgid] =  state[ngid];
            }
        }

        //Back edge comm
        for (i = blockDim.x+OPS; i < blockDim.x+TWO*OPS; i++)
        {
            for (j = 0; j < NV; j++) {
                ntgid = tgid+i*(blockDim.y+TWO*OPS)+j*SGIDS;
                ngid = gid+i*M+j*VARS+curr_time*TIMES;
                shared_state[ntgid] =  state[ngid];
            }
        }

        if (tgid<OPS || blockDim.x+OPS<=tgid)
        {
            for (i = 0; i < blockDim.x+TWO*OPS; i++) {
                for (j = 0; j < NV; j++) {
                    ntgid = tgid+i*(blockDim.y+TWO*OPS)+j*SGIDS;
                    ngid = gid+i*M+j*VARS+curr_time*TIMES;
                    shared_state[ntgid] =  state[ngid];
                }
            }
        }
    }
    __syncthreads();
}

__device__
void shared_state_clear(float * shared_state)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int sgid = get_sgid(tidx,tidy); //Shared global index
    for (int i = 0; i < TWO; i++)
    {
        for (int j = 0; j < NV; j++)
        {
            shared_state[sgid+j*SGIDS+STS*i] = 0;
        }
    }
    tidx = threadIdx.x+2*OPS;
    tidy = threadIdx.y+2*OPS;
    sgid = get_sgid(tidx,tidy); //Shared global index
    for (int i = 0; i < TWO; i++)
    {
        for (int j = 0; j < NV; j++)
        {
            shared_state[sgid+j*SGIDS+STS*i] = 0;
        }
    }
}
/*
    Use this function to create and return the GPU UpPyramid
*/
/*
    Use this function to create and return the GPU UpPyramid
*/
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

    //Communicating edge values to shared array
    edge_comm(shared_state, state,ZERO);
    if (gts%TSO==0)
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-TIMES]; //Current time step
            shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Current time step
        }
    }
    else
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Current time step
            // printf("%d\n", state[gid+i*VARS]);
        }
    }

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
