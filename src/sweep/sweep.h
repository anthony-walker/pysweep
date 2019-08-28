
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
__device__ __constant__  float SPLITX; //half an octahedron x
__device__ __constant__  float SPLITY; //half an octahedron y
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
void edge_comm(float * shared_state, float * state, int curr_time, int mod_sts)
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
                ntgid = tgid+i*(blockDim.y+TWO*OPS)+j*SGIDS+STS*mod_sts;
                ngid = gid+i*M+j*VARS+curr_time*TIMES;
                shared_state[ntgid] =  state[ngid];
            }
        }
        //Back edge comm
        for (i = blockDim.x+OPS; i < blockDim.x+TWO*OPS; i++)
        {
            for (j = 0; j < NV; j++) {
                ntgid = tgid+i*(blockDim.y+TWO*OPS)+j*SGIDS+STS*mod_sts;
                ngid = gid+i*M+j*VARS+curr_time*TIMES;
                shared_state[ntgid] =  state[ngid];
            }
        }

        if (tgid<OPS || blockDim.x+OPS<=tgid)
        {
            for (i = 0; i < blockDim.x+TWO*OPS; i++) {
                for (j = 0; j < NV; j++) {
                    ntgid = tgid+i*(blockDim.y+TWO*OPS)+j*SGIDS+STS*mod_sts;
                    ngid = gid+i*M+j*VARS+curr_time*TIMES;
                    shared_state[ntgid] =  state[ngid];
                }
            }
        }
    }
    __syncthreads();
}

/*
    Use this function to clear the shared memory array
*/
__device__
void shared_state_zero(float * shared_state)
{
    int M = (gridDim.y*blockDim.y+2*OPS); //Major y axis length
    int tgid = getGlobalIdx_2D_2D()%(blockDim.x*blockDim.y);
    int ntgid;
    if (tgid < blockDim.y+TWO*OPS)
    {
        int ntgid;
        int ngid;
        int i;

        for (i = 0; i < blockDim.x+TWO*OPS; i++)
        {
            for (int j = 0; j < NV; j++) {
                ntgid = tgid+i*(blockDim.y+TWO*OPS)+j*SGIDS;
                shared_state[ntgid] = 0;
            }

        }
    }
    __syncthreads();
}

/*
    Use this function to create and return the GPU UpPyramid
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
UpPyramid(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_zero(shared_state);
    //Other quantities for indexing
    int tidx = threadIdx.x+OPS;
    int tidy = threadIdx.y+OPS;
    int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
    int gid = get_gid()+(TSO-ONE)*TIMES; //global index
    //Creating swept boundaries
    int lx = OPS; //Lower x swept bound
    int ly = OPS; // Lower y swept bound
    int ux = blockDim.x+OPS; //upper x
    int uy = blockDim.y+OPS; //upper y

    //Communicating edge values to shared array for the calculation point (ONE)
    edge_comm(shared_state, state,ZERO,ONE);
    //Communicating interior points for TSO data and calculation data
    if (gts%TSO!=0)
    {
      for (int i = 0; i < NV; i++)
      {
          shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS]; //Initial time step to TSO stage
          shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step to Calc stage
      }
    }


    __syncthreads(); //Sync threads here to ensure all initial values are copied

    for (int k = 0; k < MPSS; k++)
    {
        // Solving step function
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            step(shared_state,sgid,gts);
            //Ensures steps are done prior to communication
            __syncthreads();
            if (gts%TSO==0)
            {
                for (int j = 0; j < NV; j++)
                {
                    // Place values back in original matrix
                    state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
                    //Copy TSO Step Down
                    shared_state[sgid+j*SGIDS-STS] = shared_state[sgid+j*SGIDS];
                }
            }
            else
            {
                for (int j = 0; j < NV; j++)
                {
                    // Place values back in original matrix
                    state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
                }
            }
        }
        gts += 1;   //Update gts
        __syncthreads(); //Sync threads here to ensure all initial values are copied
        //Update swept bounds
        ux -= OPS;
        uy -= OPS;
        lx += OPS;
        ly += OPS;

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
    shared_state_zero(shared_state);

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

    //Communicating edge values to shared array
    edge_comm(shared_state, state,ZERO,ONE);
    //Communicating interior points for TSO data and calculation data
    for (int i = 0; i < NV; i++)
    {
        shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-TIMES]; //Initial time step
        shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied

    for (int k = 0; k < MPSS-ONE; k++)
    {
        // Solving step function
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            step(shared_state,sgid,gts);
            //Ensures steps are done prior to communication
            __syncthreads();
            for (int j = 0; j < NV; j++)
            {
                // Place values back in original matrix
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        //Necessary communication step
        if (gts%TSO==0)
        {
            for (int j = 0; j < NV; j++)
            {
                //Copy latest step
                shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
                //Copy TSO Step Down
                shared_state[sgid+j*SGIDS-STS] = shared_state[sgid+j*SGIDS];
            }
        }
        else
        {
            for (int j = 0; j < NV; j++)
            {
                //Copy latest step
                shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            }
        }
        gts += 1;   //Update gts
        __syncthreads(); //Sync threads here to ensure all initial values are copied
        //Update swept bounds
        lx-=OPS;
        ux+=OPS;
        ly+=OPS;
        uy-=OPS;

    }

}


/*
    Use this function to create and return the GPU UpPyramid
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
BridgeY(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_zero(shared_state);

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


    //Communicating edge values to shared array
    edge_comm(shared_state, state,ZERO,ONE);
    //Communicating interior points for TSO data and calculation data
    for (int i = 0; i < NV; i++)
    {
        shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-TIMES]; //Initial time step
        shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied

    for (int k = 0; k < MPSS-ONE; k++)
    {
        // Solving step function
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            step(shared_state,sgid,gts);
            //Ensures steps are done prior to communication
            __syncthreads();
            for (int j = 0; j < NV; j++)
            {
                // Place values back in original matrix
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        //Necessary communication step
        if (gts%TSO==0)
        {
            for (int j = 0; j < NV; j++)
            {
                //Copy latest step
                shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
                //Copy TSO Step Down
                shared_state[sgid+j*SGIDS-STS] = shared_state[sgid+j*SGIDS];
            }
        }
        else
        {
            for (int j = 0; j < NV; j++)
            {
                //Copy latest step
                shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            }
        }
        gts += 1;   //Update gts
        __syncthreads(); //Sync threads here to ensure all initial values are copied

        //Update swept bounds
        ly-=OPS;
        uy+=OPS;
        lx+=OPS;
        ux-=OPS;

    }

}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
Octahedron(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    shared_state_zero(shared_state);

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

    //Communicating interior points for TSO data and calculation data
    for (int i = 0; i < NV; i++)
    {
        shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-TIMES]; //Initial time step
        shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
    }
    __syncthreads(); //Sync threads here to ensure all initial values are copied

    //Calculate Down Pyramid - Down Step
    for (int k = 0; k <= MDSS-ONE; k++)
    {

        // Solving step function
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            step(shared_state,sgid,gts);
            //Ensures steps are done prior to communication
            __syncthreads();
            for (int j = 0; j < NV; j++)
            {
                // Place values back in original matrix
                state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
            }
        }
        //Necessary communication step
        if (gts%TSO==0)
        {
            for (int j = 0; j < NV; j++)
            {
                //Copy latest step
                shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
                //Copy TSO Step Down
                shared_state[sgid+j*SGIDS-STS] = shared_state[sgid+j*SGIDS];
            }
        }
        else
        {
            for (int j = 0; j < NV; j++)
            {
                //Copy latest step
                shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
            }
        }
        gts += 1;   //Update gts
        __syncthreads(); //Sync threads here to ensure all initial values are copied

        //Update swept bounds
        ux += OPS;
        uy += OPS;
        lx -= OPS;
        ly -= OPS;
    }

    //------------------------UPPYRAMID of OCTAHEDRON-----------------------------
    //Reassigning swept bounds
    lx = OPS; //Lower x swept bound
    ly = OPS; // Lower y swept bound
    ux = blockDim.x+OPS; //upper x
    uy = blockDim.y+OPS; //upper y
    //Communicating edge values to shared array
    edge_comm(shared_state, state,MDSS+ONE,ONE);

    __syncthreads(); //Sync threads here to ensure all initial values are copied
    for (int k = MDSS; k < MOSS; k++)
    {
        // Solving step function
        if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
        {
            step(shared_state,sgid,gts);
            //Ensures steps are done prior to communication
            __syncthreads();
            if (gts%TSO==0)
            {
                for (int j = 0; j < NV; j++)
                {
                    // Place values back in original matrix
                    state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
                    //Copy TSO Step Down
                    shared_state[sgid+j*SGIDS-STS] = shared_state[sgid+j*SGIDS];
                }
            }
            else
            {
                for (int j = 0; j < NV; j++)
                {
                    // Place values back in original matrix
                    state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
                }
            }
        }
        gts += 1;   //Update gts
        __syncthreads(); //Sync threads here to ensure all initial values are copied
        //Update swept bounds
        ux -= OPS;
        uy -= OPS;
        lx += OPS;
        ly += OPS;
    }
}

__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
DownPyramid(float *state, int gts)
{
  //Creating flattened shared array ptr (x,y,v) length
  extern __shared__ float shared_state[];    //Shared state specified externally
  shared_state_zero(shared_state);

  //Other quantities for indexing
  int tidx = threadIdx.x+OPS;
  int tidy = threadIdx.y+OPS;
  int sgid = get_sgid(tidx,tidy)+STS; //Shared global index
  int gid = get_gid()+TSO*TIMES; //global index
  int TOPS = 2*OPS;
  int MDSS = MOSS-MPSS;
  //------------------------DOWNPYRAMID -----------------------------

  //Creating swept boundaries
  int lx =(blockDim.x+TOPS)/TWO-OPS; //lower x
  int ly = (blockDim.y+TOPS)/TWO-OPS; //lower y
  int ux =(blockDim.x+TOPS)/TWO+OPS; //upper x
  int uy = (blockDim.y+TOPS)/TWO+OPS; //upper y

  //Communicating interior points for TSO data and calculation data
  for (int i = 0; i < NV; i++)
  {
      shared_state[sgid+i*SGIDS-STS] = state[gid+i*VARS-TIMES]; //Initial time step
      shared_state[sgid+i*SGIDS] = state[gid+i*VARS]; //Initial time step
  }
  __syncthreads(); //Sync threads here to ensure all initial values are copied

  //Calculate Down Pyramid - Down Step
  for (int k = 0; k <= MDSS; k++)
  {

      // Solving step function
      if (tidx<ux && tidx>=lx && tidy<uy && tidy>=ly)
      {
          step(shared_state,sgid,gts);
          //Ensures steps are done prior to communication
          __syncthreads();
          for (int j = 0; j < NV; j++)
          {
              // Place values back in original matrix
              state[gid+j*VARS+(k+1)*TIMES]=shared_state[sgid+j*SGIDS];
          }
      }
      //Necessary communication step
      if (gts%TSO==0)
      {
          for (int j = 0; j < NV; j++)
          {
              //Copy latest step
              shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
              //Copy TSO Step Down
              shared_state[sgid+j*SGIDS-STS] = shared_state[sgid+j*SGIDS];
          }
      }
      else
      {
          for (int j = 0; j < NV; j++)
          {
              //Copy latest step
              shared_state[sgid+j*SGIDS] = state[gid+j*VARS+(k+1)*TIMES];
          }
      }
      gts += 1;   //Update gts
      __syncthreads(); //Sync threads here to ensure all initial values are copied
      //Update swept bounds
      ux += OPS;
      uy += OPS;
      lx -= OPS;
      ly -= OPS;
  }
}
