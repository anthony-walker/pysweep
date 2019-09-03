
//Constant Memory Values
__device__ __constant__ const float ZERO=0;
__device__ __constant__ const float QUARTER=0.25;
__device__ __constant__ const float HALF=0.5;
__device__ __constant__ const float ONE=1;
__device__ __constant__ const float TWO=2;
//Swept constants
__device__ __constant__ int SGIDS; //shift in shared data
__device__ __constant__ int VARS; //shift in variables in data
__device__ __constant__ int TIMES; //shift in times in data
__device__ __constant__  int MPSS; //max pyramid swept steps
__device__ __constant__  int MOSS; //max octahedron swept steps
__device__ __constant__  int NV; //number of variables
__device__ __constant__ int OPS; //number of atomic operations
__device__ __constant__  float SPLITX; //half an octahedron x
__device__ __constant__  float SPLITY; //half an octahedron y
__device__ __constant__ int TSO; //Time scheme order
__device__ __constant__ int STS;



__device__ int get_sgid(int tidx, int tidy)
{
    return tidy + (blockDim.y+2*OPS)*(tidx);
}

__device__ int sgid_inside()
{
    return threadIdx.y + (blockDim.y)*(threadIdx.x);
}

__device__ int get_gid_ftid(int tidx, int tidy)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * ((blockDim.x+2*OPS)*(blockDim.y+2*OPS))+ (tidy *(blockDim.x+2*OPS)) + tidx;
    return threadId;
}

__device__ int get_gid()
{
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int M = gridDim.y*blockDim.y+2*OPS;
    return M*(OPS+threadIdx.x)+OPS+threadIdx.y+blockDim.y*blockIdx.y+blockDim.x*blockIdx.x*M;
}

__device__
void edge_comm(float * shared_state, int sgid, int tidx,int fill)
{

}

__device__
void print_sarr(float * shared_state,bool test_bool)
{
    int bdx = blockDim.x+OPS;
    int bdy = blockDim.y+OPS;
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int id;
    __syncthreads();
    if (test_bool)
    {
        for (int i = -OPS; i < bdx; i++)
        {   printf("%s","[" );
            for (int j = -OPS; j < bdy; j++)
            {
                id = sgid+i*(bdx+OPS)+j;
                printf("%0.1f, ",shared_state[id],id);
            }
            printf("%s\n","]" );
        }
    }
    __syncthreads();
}

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

__global__
void tec(float * state)
{
    extern __shared__ float shared_state[];    //Shared state specified externally
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    shared_state_clear(shared_state);
    __syncthreads();

    // int gid = get_gid()+(TSO-1)*TIMES; //global index
    bool testbool = threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==1 && blockIdx.y==1;
    print_sarr(shared_state,testbool);
    __syncthreads();
    if (testbool) {
        printf("%s\n","--------------------------" );
    }
    __syncthreads();
    shared_state_fill(shared_state,state,1,TSO-1);
    __syncthreads();
    print_sarr(shared_state,testbool);

}
