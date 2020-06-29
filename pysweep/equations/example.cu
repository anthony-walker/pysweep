/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ const int NVC=1; //Number of variables

__device__
void getPoint(double * curr_point,double *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
}

__device__
void step(double * shared_state, int idx, int globalTimeStep)
{
  double cpoint[NVC];
  getPoint(cpoint,shared_state,idx);
  shared_state[idx+TIMES]=cpoint[0]+1;
}
