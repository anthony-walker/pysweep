/*
    Programmer: Anthony Walker
    The equation specific functions Unsteady HDE
*/

__device__ __constant__  float DX;
__device__ __constant__  float DY;
__device__ __constant__  float DT;
__device__ __constant__ float DTDX2;
__device__ __constant__ float DTDY2;
__device__ __constant__ float ALPHA;
/*
  Use this function to get point data
*/

__device__
void d2Tdxy(float & dT, float * shared_state, int idx)
{
  //Constants and Initializing
    float cpoint = shared_state[idx];
    float npoint = shared_state[idx+1];
    float spoint = shared_state[idx-1];
    float epoint = shared_state[idx+(2*OPS+blockDim.y)];
    float wpoint = shared_state[idx-(2*OPS+blockDim.y)];
    //Gradients
    dT = ALPHA*DTDX2*(wpoint+epoint-2*cpoint) + ALPHA*DTDY2*(npoint+spoint-2*cpoint);
}

__device__
void step(float *shared_state, int idx, int gts)
{
    float dT;
    d2Tdxy(dT,shared_state,idx);
    __syncthreads();
    if ((gts+1)%TSO==0) //Corrector step
    {
      shared_state[idx]=shared_state[idx-STS]+dT;
    }
    else //Predictor step
    {
      shared_state[idx]=shared_state[idx]+HALF*(dT);
    }
}

/*
    This function is for testing purposes
*/
// __device__
// void step(float *shared_state, int idx, int gts)
// {
//     float TV = (shared_state[idx-(2*OPS+blockDim.y)]+shared_state[idx+(2*OPS+blockDim.y)]+shared_state[idx-1]+shared_state[idx+1])/4;
//       __syncthreads();
//       if ((gts+1)%TSO==0) //Corrector step
//       {
//         shared_state[idx]=shared_state[idx-STS]+2*TV;
//       }
//       else //Predictor step
//       {
//         shared_state[idx]=shared_state[idx]+TV;
//       }
// }
