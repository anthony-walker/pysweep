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
void step(float * shared_state, int idx, int gts)
{
  // printf("%d\n", idx);
  float cpoint[NVC];
  getPoint(cpoint,shared_state,idx);
  float epoint[NVC];
  getPoint(epoint,shared_state,idx+(2*OPS+blockDim.y));
  float wpoint[NVC];
  getPoint(wpoint,shared_state,idx-(2*OPS+blockDim.y));
  float eepoint[NVC];
  getPoint(eepoint,shared_state,idx+2*(2*OPS+blockDim.y));
  float wwpoint[NVC];
  getPoint(wwpoint,shared_state,idx-2*(2*OPS+blockDim.y));
  float npoint[NVC];
  getPoint(npoint,shared_state,idx+1);
  float spoint[NVC];
  getPoint(spoint,shared_state,idx-1);
  float nnpoint[NVC];
  getPoint(nnpoint,shared_state,idx+2);
  float sspoint[NVC];
  getPoint(sspoint,shared_state,idx-2);
  float tval[NVC]={0,0,0,0};

   __syncthreads();

  if ((gts+1)%TSO==0) //Corrector step
  {
      // printf("%s\n", "C");
      for (int i = 0; i < NVC; i++)
      {
          tval[i] = shared_state[idx-STS]+2;
          // tval[i]=2+shared_state[idx-STS];
      }

  }
  else //Predictor
  {
      for (int i = 0; i < NVC; i++)
      {
          // printf("%s\n", "P");
          tval[i] = (epoint[i]+wpoint[i]+npoint[i]+spoint[i])/4+1;
          // tval[i]=1+shared_state[idx];
      }
  }

  __syncthreads();

  for (int i = 0; i < NVC; i++)
  {
      shared_state[idx+i*SGIDS]=tval[i];
  }
}
