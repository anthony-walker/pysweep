/*
    Programmer: Anthony Walker
    The equation specific functions for checker test.
*/
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ bool SCHEME; //Determine which scheme to use

__device__ void checkerOneStep(double * shared_state, int idx, int globalTimeStep)
{
  shared_state[idx+TIMES]=(shared_state[idx+1]+shared_state[idx-1]+shared_state[idx+SY]+shared_state[idx-SY])/4;
}


__device__ void checkerTwoStep(double * shared_state, int idx, int globalTimeStep)
{
  shared_state[idx+TIMES]=(shared_state[idx+1]+shared_state[idx-1]+shared_state[idx+SY]+shared_state[idx-SY])/4;
}

__device__
void step(double * shared_state, int idx, int globalTimeStep)
{

 if (SCHEME)
 {
    checkerOneStep(shared_state,idx,globalTimeStep);
 }
 else
 {
     checkerTwoStep(shared_state,idx,globalTimeStep);
 }
}