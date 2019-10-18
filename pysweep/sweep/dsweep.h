
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
//Swept constants
__device__ __constant__ int SGIDS; //shift in shared data
__device__ __constant__ int VARS; //shift in variables in data
__device__ __constant__ int TIMES; //shift in times in data
__device__ __constant__  int MPSS; //max pyramid swept steps
__device__ __constant__  int MOSS; //max octahedron swept steps
__device__ __constant__  int NV; //number of variables
__device__ __constant__ int OPS; //number of atomic operations
__device__ __constant__ int TSO; //Time scheme order
__device__ __constant__ int STS;
//GPU Constants
__device__ __constant__ const int LB_MIN_BLOCKS = 1;    //Launch bounds min blocks
__device__ __constant__ const int LB_MAX_THREADS = 1024; //Launch bounds max threads per block

//!!(@#

/*
    Use this function to get the shared id
*/
__device__ int get_sgid(int tidx, int tidy)
{
    return tidy + (blockDim.y)*tidx;
}
/*
    Use this function to get the global id
*/
__device__ int get_gid()
{
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = bid*(blockDim.x*blockDim.y)+(threadIdx.y*blockDim.x)+threadIdx.x;
    return tid;
}

/*
    Use this function to print the shared array
*/
__device__
void pint(int myInt,bool test_bool)
{
    __syncthreads();
    if (test_bool) {
        printf("%d\n",myInt);
    }
    __syncthreads();
}

__device__
void print_sarr(float * shared_state,bool test_bool,int smod)
{
    int bdx = blockDim.x;
    int bdy = blockDim.y;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int sgid = get_sgid(tidx,tidy)+smod*STS; //Shared global index
    int id;
    __syncthreads();
    if (test_bool)
    {
      printf("%s\n", "---------------------------------------");
        for (int i = 0; i < blockDim.x+2*OPS; i++)
        {   printf("%s","[" );
            for (int j = 0; j < blockDim.y+2*OPS; j++)
            {
                id = sgid+(i-OPS)*(bdx+OPS)+j-OPS;
                printf("%0.1f, ",shared_state[id],id);
            }
            printf("%s\n","]" );
        }
        printf("%s\n", "---------------------------------------");
    }
    __syncthreads();
}

__device__
void print_state(float * state,bool test_bool,int tmod)
{
    int bdy = blockDim.y*(gridDim.y);
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int gid = get_gid()+tmod*TIMES; //global index
    int id;
    __syncthreads();
    if (test_bool)
    {
      printf("%s\n", "---------------------------------------");
        for (int i = 0; i < blockDim.x; i++)
        {   printf("%s","[" );
            for (int j = 0; j < blockDim.y; j++)
            {
                id = gid+i*(bdy)+j;
                printf("%0.1f, ",state[id],id);
            }
            printf("%s\n","]" );
        }
        printf("%s\n", "---------------------------------------");
    }
    __syncthreads();
}

/*
    Use this function to communicate edges more effectively
*/
__device__
void shared_state_fill(float * shared_state, float * state, int smod, int tmod)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int sgid = get_sgid(tidx,tidy)+smod*STS; //Shared global index
    int gid = get_gid()+tmod*TIMES; //global index
    for (int j = 0; j < NV; j++)
    {
        shared_state[sgid+j*SGIDS] = state[gid+j*VARS];
    }
}

/*
    Use this function to clear out former values in the shared array. (Zero them)
*/
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
            shared_state[sgid+i*STS+j*SGIDS] = 0;
        }
    }
}
/*
    Use this function to create and return the GPU UpPyramid
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
UpPrism(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    // Cleaning out shared array
    shared_state_clear(shared_state);
    __syncthreads();
    //Other quantities for indexing
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int gid = get_gid()+(TSO-ONE)*TIMES; //global index
    //Creating swept boundaries
    int lx = OPS; //Lower x swept bound
    int ly = OPS; // Lower y swept bound
    int ux = blockDim.x-OPS; //upper x
    int uy = blockDim.y-OPS; //upper y
    //Communicating interior points for TSO data and calculation data
    //UpPyramid Always starts on predictor so no if needed here
    shared_state_fill(shared_state,state,ZERO,(TSO-ONE));
    shared_state_fill(shared_state,state,ONE,(TSO-ONE));
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    //Swept loop
    for (int k = 0; k < MPSS; k++)
    {
        // Solving step function
        // step(shared_state,sgid,gts);
        //Putting sweep values into next time step
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            // printf("%d,%d\n",tidx,tidy );
            for (int j = 0; j < NV; j++)
            {
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS]+5;
            }
        }
        __syncthreads();
        // for (int j = 0; j < NV; j++)
        // {
        //     //Copy latest step
        //     shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
        //     //Copy TSO Step Down
        //     shared_state[sgid+j*SGIDS-STS] = state[gid+j*VARS+(k-gts%TSO)*TIMES];
        // }
        gts += 1;   //Update gts
        //Update swept bounds
        ux -= OPS;
        uy -= OPS;
        lx += OPS;
        ly += OPS;
        __syncthreads(); //Sync threads here to ensure all initial values are copied
    }
}
/*
    Use this function to create and return the GPU X-Bridge
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
BridgeX(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_clear(shared_state);
    __syncthreads();
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int gid = get_gid()+TSO*TIMES; //global index
    //Creating swept boundaries
    int lx = blockDim.x/2; //Lower x swept bound
    int ly = 2*OPS; // Lower y swept bound
    int ux = lx+2*OPS; //upper x
    int uy = ly+blockDim.y-2*OPS; //upper y
    //Communicating interior points for TSO data and calculation data
    for (int i = 0; i < NV; i++)
    {
        shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-(gts%TSO)*TIMES]; //Initial time step
        shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    for (int k = 0; k < MPSS-ONE; k++)
    {
        step(shared_state,sgid,gts);
        // Solving step function
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            //Ensures steps are done prior to communication
            for (int j = 0; j < NV; j++)
            {
                // Place values back in original matrix
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        __syncthreads();
        for (int j = 0; j < NV; j++)
        {
            //Copy latest step
            shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            //Copy TSO Step Down
            shared_state[sgid+j*SGIDS-STS] = state[gid+j*VARS+(k-gts%TSO)*TIMES];
        }
        gts += 1;   //Update gts
        //Update swept bounds
        lx-=OPS;
        ux+=OPS;
        ly+=OPS;
        uy-=OPS;
        __syncthreads(); //Sync threads here to ensure all initial values are copied
    }

}

/*
    Use this function to create and return the GPU X-Bridge
*/

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
BridgeY(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_clear(shared_state);
    __syncthreads();
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int gid = get_gid()+TSO*TIMES; //global index
    //Creating swept boundaries
    int lx = 2*OPS; // Lower y swept bound
    int ly = blockDim.y/2; //Lower x swept bound
    int ux = lx+blockDim.x-2*OPS; //upper y
    int uy = ly+2*OPS; //upper x
    // bool test_bool = threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x==0 && blockIdx.y==0;
    //Communicating interior points for TSO data and calculation data
    for (int i = 0; i < NV; i++)
    {
        shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-(gts%TSO)*TIMES]; //Initial time step
        shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied

    for (int k = 0; k < MPSS-ONE; k++)
    {
        // Solving step function
        step(shared_state,sgid,gts);
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
          //Ensures steps are done prior to communication
            for (int j = 0; j < NV; j++)
            {
                // Place values back in original matrix
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        __syncthreads(); //Sync threads here to ensure all initial values are copied
        for (int j = 0; j < NV; j++)
        {
            //Copy latest step
            shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            //Copy TSO Step Down
            shared_state[sgid+j*SGIDS-STS] = state[gid+j*VARS+(k-gts%TSO)*TIMES];
        }
        gts += 1;   //Update gts
        //Update swept bounds
        ly-=OPS;
        uy+=OPS;
        lx+=OPS;
        ux-=OPS;
        __syncthreads();
    }
}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
Octahedron(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_clear(shared_state);
    __syncthreads();
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int gid = get_gid()+TSO*TIMES; //global index
    int TOPS = 2*OPS;
    int MDSS = MOSS-MPSS;
     // bool test_bool = threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x==0 && blockIdx.y==0;
    //------------------------DOWNPYRAMID of OCTAHEDRON-----------------------------
    //Creating swept boundaries
    int lx =(blockDim.x+TOPS)/TWO-OPS; //lower x
    int ly = (blockDim.y+TOPS)/TWO-OPS; //lower y
    int ux =(blockDim.x+TOPS)/TWO+OPS; //upper x
    int uy = (blockDim.y+TOPS)/TWO+OPS; //upper y
    for (int i = 0; i < NV; i++)
    {
        shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-(gts%TSO)*TIMES]; //Initial time step
        shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    //Calculate Down Pyramid - Down Step
    for (int k = 0; k <= MDSS; k++)
    {
        step(shared_state,sgid,gts);
        // Solving step function

        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            //Ensures steps are done prior to communication
            for (int j = 0; j < NV; j++)
            {
                // Place values back in original matrix
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        __syncthreads();
        //Necessary communication step
        for (int j = 0; j < NV; j++)
        {
            //Copy latest step
            shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            //Copy TSO Step Down
            shared_state[sgid+j*SGIDS-STS] = state[gid+j*VARS+(k-gts%TSO)*TIMES];
        }
        gts += 1;   //Update gts
        //Update swept bounds
        ux += OPS;
        uy += OPS;
        lx -= OPS;
        ly -= OPS;
        __syncthreads(); //Sync threads here to ensure all initial values are copied
    }

    //------------------------UPPYRAMID of OCTAHEDRON-----------------------------
    //Reassigning swept bounds
    lx = OPS; //Lower x swept bound
    ly = OPS; // Lower y swept bound
    ux = blockDim.x+OPS; //upper x
    uy = blockDim.y+OPS; //upper y
    //Swept loop
    for (int k = MDSS+ONE; k < MOSS; k++)
    {
        //Update swept bounds - It is first here so it does not do the full base
        ux -= OPS;
        uy -= OPS;
        lx += OPS;
        ly += OPS;
        __syncthreads(); //Sync threads here to ensure all initial values are copied
        // Solving step function
        step(shared_state,sgid,gts);
        //Putting sweep values into next time step
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            for (int j = 0; j < NV; j++)
            {
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        __syncthreads();
        for (int j = 0; j < NV; j++)
        {
            //Copy latest step
            shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            //Copy TSO Step Down
            shared_state[sgid+j*SGIDS-STS] = state[gid+j*VARS+(k-gts%TSO)*TIMES];
        }
        gts += 1;   //Update gts
    }
}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
DownPyramid(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_clear(shared_state);
    __syncthreads();
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int gid = get_gid()+TSO*TIMES; //global index
    int TOPS = 2*OPS;
    int MDSS = MOSS-MPSS;
    //------------------------DOWNPYRAMID of OCTAHEDRON-----------------------------

    //Creating swept boundaries
    int lx =(blockDim.x+TOPS)/TWO-OPS; //lower x
    int ly = (blockDim.y+TOPS)/TWO-OPS; //lower y
    int ux =(blockDim.x+TOPS)/TWO+OPS; //upper x
    int uy = (blockDim.y+TOPS)/TWO+OPS; //upper y
    for (int i = 0; i < NV; i++)
    {
        shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-(gts%TSO)*TIMES]; //Initial time step
        shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied
    for (int k = 0; k <= MDSS; k++)
    {
        step(shared_state,sgid,gts);
        // Solving step function

        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            //Ensures steps are done prior to communication
            for (int j = 0; j < NV; j++)
            {
                // Place values back in original matrix
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        __syncthreads();
        //Necessary communication step
        for (int j = 0; j < NV; j++)
        {
            //Copy latest step
            shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            //Copy TSO Step Down
            shared_state[sgid+j*SGIDS-STS] = state[gid+j*VARS+(k-gts%TSO)*TIMES];
        }
        gts += 1;   //Update gts
        //Update swept bounds
        ux += OPS;
        uy += OPS;
        lx -= OPS;
        ly -= OPS;
        __syncthreads(); //Sync threads here to ensure all initial values are copied
    }
}
