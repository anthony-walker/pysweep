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
  void step(float *state, int idx)
  {
      float *flux = new float[4];
      for (int i = 0; i < 4; i++)
      {
          // state[idx+i] *= state[idx+i];
          // // printf("%0.2f\n",state[idx+i] );
      }

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
