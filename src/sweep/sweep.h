
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
__device__ __constant__  float SPLITX; //half an octahedron x
__device__ __constant__  float SPLITY; //half an octahedron y

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
void plane_shared_comm(float * shared_state, float * state, int time)
{
    int M = (gridDim.y*blockDim.y+2*OPS); //Major y axis length
    int tgid = getGlobalIdx_2D_2D()%(blockDim.x*blockDim.y);
    int gid = tgid+blockIdx.y*blockDim.y+blockDim.x*blockIdx.x*M;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    if (tgid < blockDim.y+TWO*OPS)
    {
        int i;
        //Front edge comm
        for (i = 0; i < OPS; i++)
        {
            int ntgid = tgid+i*(blockDim.y+TWO*OPS);
            int ngid = gid+i*M;
            for (int j = 0; j < NV; i++)
            {
                shared_state[ntgid+j*SGIDS] = state[ngid+j*VARS+time*TIMES]; //Current time step
            }
        }

        //Back edge comm
        for (i = blockDim.x+OPS; i < blockDim.x+TWO*OPS; i++)
        {
            int ntgid = tgid+i*(blockDim.y+TWO*OPS);
            int ngid = gid+i*M;
            shared_state[ntgid] =  state[ngid]+1;
        }

        if (tgid<OPS || blockDim.x+OPS<=tgid)
        {
            for (i = 0; i < blockDim.x+TWO*OPS; i++) {
                int ntgid = tgid+i*(blockDim.y+TWO*OPS);
                int ngid = gid+i*M;
                shared_state[ntgid] =  state[ngid]+1;
            }
        }
        // if (bid == 3)
        // {
        //     for (i = 0; i < blockDim.x+TWO*OPS; i++)
        //     {
        //         int ntgid = tgid+i*(blockDim.y+TWO*OPS);
        //         int ngid = gid+i*M;
        //         printf("%d\n",ngid );
        //     }
        //     // printf("%s\n","," );
        // }
    }
    __syncthreads();
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



    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy); //Shared global index
    int gid = get_gid(); //global index

    //Creating swept boundaries
    int lx = OPS; //Lower x swept bound
    int ly = OPS; // Lower y swept bound
    int ux = blockDim.x+OPS; //upper x
    int uy = blockDim.y+OPS; //upper y

    // printf("%d,%d,%d\n",TIMES,VARS, SGIDS );

    // plane_shared_comm(shared_state, state, 0);


    // for (int k = 0; k < MPSS; k++)
    // {
    //     for (int i = 0; i < NV; i++)
    //     {
    //         shared_state[sgid+i*SGIDS] = state[gid+i*VARS+k*TIMES]; //Current time step
    //     }
    //     __syncthreads(); //Sync threads here to ensure all initial values are copied
    //
    //
    //
    //     // // Solving step function
    //     // if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
    //     // {
    //     //     // step(shared_state,sgid);
    //     //     shared_state[sgid] = 2*shared_state[sgid];
    //     // }
    //     //
    //     // //Update swept bounds
    //     // ux -= OPS;
    //     // uy -= OPS;
    //     // lx += OPS;
    //     // ly += OPS;
    //
    //     // Place values back in original matrix
    //     for (int j = 0; j < NV; j++)
    //     {
    //         state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
    //     }
    // }

}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
Octahedron(float *state)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    //Creating swept boundaries
    int lx = blockDim.x/2; //Lower swept bound
    int ly = blockDim.y/2; //Lower swept bound y
    int ux = blockDim.x/2; //upper x
    int uy = blockDim.y/2;//blockDim.y; //upper y
    //Creating indexing
    int gid = getGlobalIdx_2D_2D();  //Getting 2D global index
    int sgid = gid%(SGIDS); //Shared global index

    //Down Pyramid step of the octahedron
    for (int k = 0; k < MPSS; k++)
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*SGIDS] = state[gid+i*VARS+k*TIMES]; //Current time step
        }
        __syncthreads(); //Sync threads here to ensure all initial values are copied

        //Update swept bounds
        ux += OPS;
        uy += OPS;
        lx -= OPS;
        ly -= OPS;

        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
        {
            step(shared_state,sgid);
        }

        // Place values back in original matrix
        for (int j = 0; j < NV; j++)
        {
            state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
        }
    }

    //Reassign lx and ly for UpPyramid
    lx = 0;
    ly = 0;
    ux = blockDim.x;
    uy = blockDim.y;

    //UpPyramid Step of Octahedron
    for (int k = MPSS; k < MPSS+MPSS; k++)
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*SGIDS] = state[gid+i*VARS+k*TIMES]; //Current time step
        }
        __syncthreads(); //Sync threads here to ensure all initial values are copied

        //Update swept bounds
        ux -= OPS;
        uy -= OPS;
        lx += OPS;
        ly += OPS;
        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
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
DownPyramid(float *state)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    //Creating swept boundaries
    int lx = blockDim.x/2; //Lower swept bound
    int ly = blockDim.y/2; //Lower swept bound y
    int ux = blockDim.x/2; //upper x
    int uy = blockDim.y/2;//blockDim.y; //upper y
    //Creating indexing
    int gid = getGlobalIdx_2D_2D();  //Getting 2D global index
    int sgid = gid%(SGIDS); //Shared global index

    for (int k = 0; k < MPSS; k++)
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*SGIDS] = state[gid+i*VARS+k*TIMES]; //Current time step
        }
        __syncthreads(); //Sync threads here to ensure all initial values are copied

        //Update swept bounds
        ux += OPS;
        uy += OPS;
        lx -= OPS;
        ly -= OPS;
        // Solving step function
        if (threadIdx.x<ux && threadIdx.x>=lx && threadIdx.y<uy && threadIdx.y>=ly)
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
