/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ double DTDX;
__device__ __constant__ double DTDY;
__device__ __constant__ double GAMMA; //Gamma
__device__ __constant__ double GAM_M1;
__device__ __constant__ const int NVC=4; //Number of variables

__device__
void getPoint(double * curr_point,double *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
    curr_point[1]=shared_state[idx+VARS];
    curr_point[2]=shared_state[idx+VARS*2];
    curr_point[3]=shared_state[idx+VARS*3];
}

__device__
void step(double * shared_state, int idx, int gts)
{
  bool cond = ((gts+1)%TSO==0);
  int coeff = cond ? 1 : 0;

  double cpoint[NVC];
  getPoint(cpoint,shared_state,idx-TIMES*coeff);

  double tval[NVC]={0,0,0,0};
   __syncthreads();


   for (int i = 0; i < NVC; i++)
   {

       tval[i] = cpoint[i]+1;
   }

  __syncthreads();

  for (int i = 0; i < NVC; i++)
  {
      shared_state[idx+TIMES+i*VARS]=tval[i];
  }
}
