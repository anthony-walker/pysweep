/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
#define SQUAREROOT(x)   sqrt(x)
#define ABSOLUTE(x)   abs(x)
#define MIN(x,y)   min(x,y)
#define ISNAN(x)   isnan(x)

__device__ __constant__  float DX;
__device__ __constant__  float DY;
__device__ __constant__  float DT;
__device__ __constant__ float DTDX;
__device__ __constant__ float DTDY;
__device__ __constant__ float GAMMA=1.4; //Gamma
__device__ __constant__ float GAM_M1=0.4;
__device__ __constant__ const int NVC=4; //Number of variables
/*
  Use this function to get point data
*/
__device__
void getPoint(float * curr_point,float *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
    curr_point[1]=shared_state[idx+SGIDS];
    curr_point[2]=shared_state[idx+SGIDS*2];
    curr_point[3]=shared_state[idx+SGIDS*3];
}

/*
  Use this function to determine the pressure
*/
__device__
float pressure(float *point)
{
  float vss = (point[1]*point[1]+point[2]*point[2])/point[0];
  return GAM_M1*(point[3]-HALF*vss);
}

/*
  This is the minmod flux limiter method for handling discontinuities
*/
__device__
void  flimiter(float *temp_state, float *left_point, float *right_point  ,float num, float den)
{
    // float comp_tol = 1e-5;
    float Pr = num/den;

    if (isnan(Pr) || isinf(Pr) || Pr > 1.0e6 || Pr < 1e-6)
    {
      for (int i = 0; i < NVC; i++)
      {
          temp_state[i] = left_point[i];
      }
    }
    else
    {
        // printf("%d,%d,%d,%d,%f\n",isnan(Pr),isinf(Pr), Pr > 1.0e6,Pr < 1e-6,Pr);
        printf("%f,%f,%f\n",Pr,num,den);
        float coef = HALF*MIN(num/den, ONE);
        for (int j = 0; j < NVC; j++)
        {
            temp_state[j] =  left_point[j]+coef*(left_point[j] - right_point[j]);
        }
    }
    __syncthreads();    //May or may not help thread divergence
}
/*
  Use this function to determine the flux in the x direction
*/

/*
Use this function to obtain the flux for each system variable
*/
__device__
void  efluxx(float *flux, float *left_state, float *right_state,int dir)
{
  float rhoL = left_state[0];
  float rhoR = right_state[0];
  float uL = left_state[1]/rhoL;
  float uR = right_state[1]/rhoR;
  float vL = left_state[2]/rhoL;
  float vR = right_state[2]/rhoR;

  float pL = pressure(left_state);
  float pR = pressure(right_state);

  flux[0] += dir*(left_state[1]+right_state[1]);
  flux[1] += dir*(left_state[1]*uL+pL+right_state[1]*uR+pR);
  flux[2] += dir*(left_state[1]*vL+right_state[1]*vR);
  flux[3] += dir*((left_state[3]+pL)*uL+(right_state[3]+pR)*uR);
 }


__device__
void get_dfdxy(float *dfdx, float *shared_state, int idx,int idmod)
{
  //Constants and Initializing
    float cpoint[NVC];
    getPoint(cpoint,shared_state,idx);
    float epoint[NVC];
    getPoint(epoint,shared_state,idx+idmod);
    float wpoint[NVC];
    getPoint(wpoint,shared_state,idx-idmod);
    float eepoint[NVC];
    getPoint(eepoint,shared_state,idx+2*idmod);
    float wwpoint[NVC];
    getPoint(wwpoint,shared_state,idx-2*idmod);
    float Pv[5]={0};
    float temp_left[NVC]={0};
    float temp_right[NVC]={0};
    int spi = 1;  //spectral radius idx
    // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",cpoint[0],cpoint[1],cpoint[2],cpoint[3]);
    // Pressure ratio
    Pv[0] = pressure(wwpoint);
    Pv[1] = pressure(wpoint);
    Pv[2] = pressure(cpoint);
    Pv[3] = pressure(epoint);
    Pv[4] = pressure(eepoint);
    // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",cpoint[0],cpoint[1],cpoint[2],cpoint[3]);
    // //West
    flimiter(temp_left,wpoint,cpoint,Pv[2]-Pv[1],Pv[1]-Pv[0]); //,num,den
    // flimiter(temp_right,cpoint,wpoint,Pv[2]-Pv[1],Pv[3]-Pv[2]); //,den,num
    // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",temp_left[0],temp_left[1],temp_left[2],temp_left[3]);
    // efluxx(dfdx,temp_left,temp_right,ONE);
   // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",dfdx[0],dfdx[1],dfdx[2],dfdx[3]);
    // espectral(dfdx,temp_left,temp_right,spi,ONE);
    // //East
    // flimiter(temp_left,cpoint,epoint,Pr[1]);
    // flimiter(temp_right,epoint,cpoint,ONE/Pr[2]);
    // efluxx(dfdx,temp_left,temp_right,-1);
    // espectral(dfdx,temp_left,temp_right,spi,-1);
}


__device__
void step(float *shared_state, int idx, int gts)
{
    // float dfdx[NVC]={0,0,0,0};
    // float dfdy[NVC]={0,0,0,0};
    // get_dfdx(dfdx,shared_state,idx);
    // __syncthreads();
    // if ((gts+1)%TSO==0) //Corrector step
    // {
    //     for (int i = 0; i < NVC; i++)
    //     {
    //         shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS-STS]+HALF*(DTDX*dfdx[i]+DTDY*dfdy[i]);
    //     }
    // }
    // else //Predictor step
    // {
    //     for (int i = 0; i < NVC; i++)
    //     {
    //         shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS]+QUARTER*(DTDX*dfdx[i]+DTDY*dfdy[i]);
    //     }
    // }
}
/*
    This function is for testing purposes
*/

__device__
int getGlobalIdx_2D_2D(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}

__global__
void test_step(float *shared_state, int idx, int gts)
{
    float dfdx[NVC]={0,0,0,0};
    float dfdy[NVC]={0,0,0,0};
    // get_dfdy(dfdy,shared_state,idx);
    idx = getGlobalIdx_2D_2D();
    if (threadIdx.x >= 2 && threadIdx.y >= 2 && threadIdx.x < 10 && threadIdx.y < 10)
    {
      get_dfdxy(dfdx,shared_state,idx,(2*OPS+blockDim.y));
    }

    // printf("%d,%f\n", idx, shared_state[idx]);
    // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",dfdx[0],dfdx[1],dfdx[2],dfdx[3]);

    __syncthreads();
    // if ((gts+1)%TSO==0) //Corrector step
    // {
    //     for (int i = 0; i < NVC; i++)
    //     {
    //         shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS-STS]+HALF*(DTDX*dfdx[i]+DTDY*dfdy[i]);
    //     }
    // }
    // else //Predictor step
    // {
    //     for (int i = 0; i < NVC; i++)
    //     {
    //         shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS]+QUARTER*(DTDX*dfdx[i]+DTDY*dfdy[i]);
    //     }
    // }
}
