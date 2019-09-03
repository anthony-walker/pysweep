
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


/*
    Use this function to get the shared id
*/
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
    Use this function to print the shared array
*/

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
/*
    Use this function to create and return the GPU UpPyramid
*/
__global__ void
__launch_bounds__(LB_MAX_THREADS, LB_MIN_BLOCKS)    //Launch bounds greatly reduce register usage
UpPyramid(float *state, int gts)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    // Cleaning out shared array
    shared_state_clear(shared_state);
    __syncthreads();
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
    //Communicating interior points for TSO data and calculation data
    //UpPyramid Always starts on predictor so no if needed here
    shared_state_fill(shared_state,state,ZERO,(TSO-ONE));
    shared_state_fill(shared_state,state,ONE,(TSO-ONE));
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

    //Communicating edge values to shared array
    // edge_comm(shared_state, state,TSO,ONE);
    __syncthreads();
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
        __syncthreads();
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
    //Communicating edge values to shared array
    // edge_comm(shared_state, state,TSO,ONE);
    __syncthreads();
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
        __syncthreads();
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
    // edge_comm(shared_state, state,MDSS+ONE,ONE);
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
  shared_state_clear(shared_state);
  __syncthreads();
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
