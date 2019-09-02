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
  float tval[NVC]={0,0,0,0};
  float cpoint[NVC];
  getPoint(cpoint,shared_state,idx);
  float epoint[NVC];
  getPoint(epoint,shared_state,idx+blockDim.y+2*OPS);
  float wpoint[NVC];
  getPoint(wpoint,shared_state,idx-blockDim.y-2*OPS);
  float npoint[NVC];
  getPoint(npoint,shared_state,idx+1);
  float spoint[NVC];
  getPoint(spoint,shared_state,idx-1);
  // printf("%f,%f,%f,%f\n",wpoint[0],wpoint[0],wpoint[0],wpoint[0]);
  __syncthreads();
  // printf("%f,%f,%f,%f\n",npoint[0],spoint[0],epoint[0],wpoint[0]);
  if (gts%TSO!=0)
  {
      for (int i = 0; i < NVC; i++)
      {
          tval[i] = (epoint[i]+wpoint[i]+npoint[i]+spoint[i])/4;
      }
  }
  else
  {
      for (int i = 0; i < NVC; i++)
      {
          tval[i] = (epoint[i]+wpoint[i]+npoint[i]+spoint[i])/4;
      }
  }

  __syncthreads();
  for (int i = 0; i < NVC; i++)
  {
      shared_state[idx+i*SGIDS]=tval[i];
  }
}
