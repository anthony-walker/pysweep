/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;

/*
    Use this function to check the value of global constants
*/
__device__ void checkHere()
{
    if (threadIdx.x==1 && threadIdx.y==1)
    {
        printf("%s\n","Here");
    }
}

__device__ double centralDifference(double * shared_state, int idx)
{
  double derivativeSum;
  derivativeSum = (shared_state[idx+SY]-2*shared_state[idx]+shared_state[idx-SY])/100.0; //X derivative
  derivativeSum += (shared_state[idx+1]-2*shared_state[idx]+shared_state[idx-1])/100.0; //Y derivative
  return derivativeSum;
}

__device__ void step(double * shared_state, int idx, int globalTimeStep)
{
bool cond = (globalTimeStep%ITS==0); //True on complete steps not intermediate
int timeChange = cond ? 1 : 0; //If true rolls back a time step
double coeff = cond ? 1.0 : 0.5; //If true uses 1 instead of 0.5 for intermediate step
shared_state[idx+TIMES]=shared_state[idx-TIMES*timeChange]+coeff*centralDifference(shared_state,idx);
}
