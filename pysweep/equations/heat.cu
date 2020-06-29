/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ double ALPHA; //Thermal diffusivity
__device__ __constant__ bool SCHEME; //Determine which scheme to use
__device__ __constant__ const int NVC=1; //Number of variables

__device__ void getPoint(double * curr_point,double *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
}

__device__ double centralDifference(double * shared_state, int idx)
{

  double derivativeSum;
  derivativeSum = (shared_state[idx+SY]-2*shared_state[idx]+shared_state[idx-SY])/(DX*DX); //X derivative
  derivativeSum += (shared_state[idx+1]-2*shared_state[idx]+shared_state[idx-1])/(DY*DY); //Y derivative
  return derivativeSum;
}


__device__ void forwardEuler(double * shared_state, int idx, int globalTimeStep)
{
  double cpoint[NVC];
  getPoint(cpoint,shared_state,idx);
  shared_state[idx+TIMES]=shared_state[idx]+ALPHA*DT*centralDifference(shared_state,idx);
}


__device__ void rungeKutta2(double * shared_state, int idx, int globalTimeStep)
{
//     bool cond = ((globalTimeStep+1)%ITS==0);
//   int coeff = cond ? 1 : 0;

//   double cpoint[NVC];
//   getPoint(cpoint,shared_state,idx-TIMES*coeff);

//   double tval[NVC]={0,0,0,0};
//    __syncthreads();


//    for (int i = 0; i < NVC; i++)
//    {

//        tval[i] = cpoint[i]+1;
//    }

//   __syncthreads();

//   for (int i = 0; i < NVC; i++)
//   {
//       shared_state[idx+TIMES+i*VARS]=tval[i];
//   }

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
     rungeKutta2(shared_state,idx,globalTimeStep);
 }
 

}
