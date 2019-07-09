/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  const int SS=5; //stencil size
__device__ __constant__ const int NVC=4;

void  pressureRatio(float *state)
{


}

__device__
float  dfdxy(float *flux, int idx)
{


  return 0.0;
}

  __device__
  void step(float *state, int idx)
  {
      int XS = blockDim.x;
     // Creating local flux vectors
     float *fluxx = new float[SS*NVC];
     float *fluxy = new float[SS*NVC];
     int fidx = 0;
     int idxp = 0;
      for (int i = -2; i <= OPS; i++)
      {
          for (int j = 0; j < NV; j++) {

              idxp = idx+j*VARS;
              // printf("%d,%d\n",idxp+i*XS,idxp);
              // fluxx[fidx] = state[idx+j*VARS+i*XS];
              // fluxy[fidx] = state[idx+j*VARS+i];
              fidx++;
          }
      }
      delete[] fluxx;
      delete[] fluxy;
  }



__device__
void  pressure(float *q)
{



}

__device__
void  direction_flux(float *state, float *Pr, bool xy)
{


}

__device__
void  flimiter(float *qL, float *qR, float *Pr)
{



}

__device__
void  eflux(float *left_state, float *right_state, bool xy)
{


}

__device__
void  espectral(float *left_state, float *right_state, bool xy)
{




}
