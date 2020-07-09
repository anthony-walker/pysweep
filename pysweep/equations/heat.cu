/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ double ALPHA; //Thermal diffusivity
__device__ __constant__ int SCHEME; //Determine which scheme to use

__device__ double centralDifference(double * shared_state, int idx)
{

  double derivativeSum;
  derivativeSum = ALPHA*DT*(shared_state[idx+SY]-2*shared_state[idx]+shared_state[idx-SY])/(DX*DX); //X derivative
  derivativeSum += ALPHA*DT*(shared_state[idx+1]-2*shared_state[idx]+shared_state[idx-1])/(DY*DY); //Y derivative
  return derivativeSum;
}


__device__ void forwardEuler(double * shared_state, int idx, int globalTimeStep)
{
  shared_state[idx+TIMES]=shared_state[idx]+centralDifference(shared_state,idx);
}


__device__ void rungeKuttaTwo(double * shared_state, int idx, int globalTimeStep)
{
bool cond = (globalTimeStep%ITS==0); //True on complete steps not intermediate
int timeChange = cond ? 1 : 0; //If true rolls back a time step
int coeff = cond ? 1 : 0.5; //If true uses 1 instead of 0.5 for intermediate step
shared_state[idx+TIMES]=shared_state[idx-TIMES*timeChange]+coeff*centralDifference(shared_state,idx);
}

__device__
void step(double * shared_state, int idx, int globalTimeStep)
{
 if (SCHEME)
 {
    forwardEuler(shared_state,idx,globalTimeStep);
 }
 else
 {
     rungeKuttaTwo(shared_state,idx,globalTimeStep);
 }
 

}
