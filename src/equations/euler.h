/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  const int SS=5; //stencil size
__device__ __constant__ const int NVC=4;

void  pressureRatio(float *state)
{
  const int nr = 3;
  float * Pr[nr];

}

__device__
void dfdx(float *state, int idx)
{
    float dfdx[NVC];
    float left_state[NVC];
    float right_state[NVC];
    pressureRatio(state);


    delete[] left_state;
    delete[] right_state;
}

__device__
void step(float *state, int idx)
{
  dfdx(state,idx);
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
