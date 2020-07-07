/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ int SCHEME; //Determine which scheme to use

__device__
void step(double * shared_state, int idx, int globalTimeStep)
{
  if (SCHEME)
  {
    shared_state[idx+TIMES]=shared_state[idx]+1;
  }
  else
  {
    bool cond = ((globalTimeStep)%ITS==0); //True on complete steps not intermediate
    int timeChange = cond ? 0 : 1; //If true rolls back a time step
    int addition = cond ? 1 : 2; //If true uses 2 instead of 1 for intermediate step
    shared_state[idx+TIMES]=shared_state[idx-TIMES*timeChange]+addition;
  }
}