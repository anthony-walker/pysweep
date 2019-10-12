/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/

__device__ __constant__ const int NVC=1; //Number of variables

__device__
void getPoint(float * curr_point,float *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
    curr_point[1]=shared_state[idx+SGIDS];
    curr_point[2]=shared_state[idx+SGIDS*2];
    curr_point[3]=shared_state[idx+SGIDS*3];
}

__device__
void step(float * shared_state, int idx, int gts)
{

  float tval[NVC]={0};

   __syncthreads();

  if ((gts+1)%TSO==0)
  {
      tval[0] = shared_state[idx-STS];
  }
  else //Predictor
  {
      tval[0] = shared_state[idx];
  }

  __syncthreads();

  shared_state[idx]=tval[0];

}
