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
void getPoint(float * curr_point,float *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
    curr_point[1]=shared_state[idx+SGIDS];
    curr_point[2]=shared_state[idx+SGIDS*2];
    curr_point[3]=shared_state[idx+SGIDS*3];
}

__device__
void step(float *shared_state, int idx, int gts)
{

  float cpoint[NVC];
  getPoint(cpoint,shared_state,idx);
  float epoint[NVC];
  getPoint(epoint,shared_state,idx+blockDim.y);
  float wpoint[NVC];
  getPoint(wpoint,shared_state,idx-blockDim.y);
  float npoint[NVC];
  getPoint(npoint,shared_state,idx+1);
  float spoint[NVC];
  getPoint(spoint,shared_state,idx-1);
  __syncthreads();

  if (gts%TSO!=0)
  {
      for (int i = 0; i < NVC; i++)
      {
          shared_state[idx+i*SGIDS] = (epoint[i]+wpoint[i]+npoint[i]+spoint[i])/4;
      }
  }
  else
  {
      for (int i = 0; i < NVC; i++)
      {
          // shared_state[idx+i*SGIDS]=7;
          shared_state[idx+i*SGIDS]+=1;
      }
  }
}
