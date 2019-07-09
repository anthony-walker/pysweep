/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  const int SS=5; //stencil size

void  pressureRatio(float *state)
{


}

__device__
float  dfdxy(float *state, int idx)
{

  int sgid_shift = blockDim.x*blockDim.y;
  int var_shift = sgid_shift*gridDim.x*gridDim.y;//This is the variable used to shift between values

  return state[idx+1*sgid_shift]*state[idx+1*sgid_shift];
}

  __device__
  void step(float *state, int idx,int k)
  {
      int XS = blockDim.x;
     // Creating flux stencils
      // float **fluxx = new float*[SS];
      // float **fluxy = new float*[SS];
      // for (int i = 0; i < SS; i++)
      // {
      //     fluxx[i] = new float[NV];
      //     fluxy[i] = new float[NV];
      // }
      // float fluxy[ss][NV];
      // for (int i = -2; i <= OPS; i++)
      // {
      //     for (int j = 0; j < NV; j++) {
      //         fluxx[i][j] = state[idx+j*VARS+i*XS];
      //         fluxy[i][j] = state[idx+j*VARS+i];
      //     }
      // }
      // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",state[idx+0*sgid_shift],state[idx+1*sgid_shift],state[idx+2*sgid_shift],state[idx+3*sgid_shift] );
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
