/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  float DX;
__device__ __constant__  float DY;
__device__ __constant__  float DT;
__device__ __constant__ float DTDX;
__device__ __constant__ float DTDY;
__device__ __constant__ float GAMMA; //Gamma
__device__ __constant__ float GAM_M1;
__device__ __constant__ const int NVC=4; //Number of variables

__device__
void step(float *shared_state, int idx, int gts)
{
  __syncthreads();
  if (gts%TSO!=0)
  {
      for (int i = 0; i < NVC; i++)
      {
          shared_state[idx+i*SGIDS]+=1;
      }
  }
  else
  {
      for (int i = 0; i < NVC; i++)
      {
          // shared_state[idx+i*SGIDS]=7;
          shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS-STS]+2;
      }
  }
}
