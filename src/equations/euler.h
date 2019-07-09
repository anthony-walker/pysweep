/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/

#define ZERO            0.0
#define QUARTER         0.25
#define HALF            0.5
#define ONE             1.0
#define TWO             2.0

  __device__
  void step(float *state, int idx,int k)
  {
      const int ss = 5; //Stencil size
      const int nv = 4;
      int sgid_shift = blockDim.x*blockDim.y;
      int var_shift = sgid_shift*gridDim.x*gridDim.y;//This is the variable used to shift between values
      float fluxx[ss][nv];
      float fluxy[ss][nv];
      for (int i = 0; i < nv; i++)
      {
          state[idx+i*sgid_shift] *= state[idx+i*sgid_shift];
      }
      printf("%0.2f,%0.2f,%0.2f,%0.2f\n",state[idx+0*sgid_shift],state[idx+1*sgid_shift],state[idx+2*sgid_shift],state[idx+3*sgid_shift] );
  }

__device__
void  dfdxy(float *state, float* idx, const int ts)
  {

  }

__device__
void  pressureRatio(float *state)
{


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
