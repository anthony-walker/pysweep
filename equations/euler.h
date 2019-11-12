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
  Use this function to determine the pressure for spectral function
*/
__device__
float spec_press(float *point)
{
  float rho = point[0];
  float vss = point[1]*point[1]+point[2]*point[2];
  return GAM_M1*(rho*point[3]-HALF*rho*vss);
}

/*
  Use this function to apply the roe average and obtain the spectral radius
*/
__device__
void espectral(float* flux,float *left_state, float *right_state, int dim, int dir)
{
  //Initializing values and determining spectral state
  float spec_state[NVC]={0};
  float rootrhoL = SQUAREROOT(left_state[0]);
  float rootrhoR = SQUAREROOT(right_state[0]);
  spec_state[0] = rootrhoL*rootrhoR;
  float denom = 1/(rootrhoL+rootrhoR);

  for (int i = 1; i < NVC; i++)
  {
    spec_state[i] += rootrhoL*left_state[i]/left_state[0];
    spec_state[i] += rootrhoR*right_state[i]/right_state[0];
    spec_state[i] *= denom; //Puts in flux form to find pressure
  }
  //Updating flux with spectral value
  float rs = (SQUAREROOT(GAMMA*spec_press(spec_state)/spec_state[0])+ABSOLUTE(spec_state[dim]));
  for (int i = 0; i < NVC; i++)
  {
    flux[i] += dir*rs*(left_state[i]-right_state[i]);
  }
  // printf("%f,%f,%f,%f\n", flux[0],flux[1],flux[2],flux[3]);
}

/*
  Use this function to determine the flux in the x direction
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

 /*
 Use this function to obtain the flux for each system variable
 */
 __device__
 void  efluxy(float* flux,float *left_state, float *right_state, int dir)
 {
   //Initializing variables
   float rhoL = left_state[0];
   float rhoR = right_state[0];
   float uL = left_state[1]/rhoL;
   float uR = right_state[1]/rhoR;
   float vL = left_state[2]/rhoL;
   float vR = right_state[2]/rhoR;
   //Determining pressures
   float pL = pressure(left_state);
   float pR = pressure(right_state);
   //Updating fluxes
   flux[0] += dir*(left_state[2]+right_state[2]);
   flux[1] += dir*(left_state[2]*uL+right_state[2]*uR);
   flux[2] += dir*(left_state[2]*vL+pL+right_state[2]*vR+pR);
   flux[3] += dir*((left_state[3]+pL)*vL+(right_state[3]+pR)*vR);
  }

  /*
    This is the minmod flux limiter method for handling discontinuities
  */
  __device__
  void  flimiter(float *temp_state,float *point1,float *point2,float *point3,float num, float den)
  {
      // float comp_tol = 1e-5;
      float Pr = num/den;

      if (isnan(Pr) || isinf(Pr) || Pr > 1.0e6 || Pr < 1e-6)
      {


        for (int i = 0; i < NVC; i++)
        {
            temp_state[i] = point1[i];
        }
      }
      else
      {
          // printf("%f\n",Pr );
          // printf("%d,%d,%d,%d,%f\n",isnan(Pr),isinf(Pr), Pr > 1.0e6,Pr < 1e-6,Pr);
          float coef = HALF*MIN(Pr, ONE);
          // printf("%f\n", coef);
          // printf("%f, %f,%f,%f\n",coef,left_point[0],right_point[0],left_point[0] - right_point[0]);
          for (int j = 0; j < NVC; j++)
          {
              temp_state[j] =  point1[j]+coef*(point2[j] - point3[j]);
          }
      }
      __syncthreads();    //May or may not help thread divergence
  }


__device__
void get_dfdx(float *dfdx, float *shared_state, int idx,int idmod)
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
    // printf("%0.2f,%0.2f,%0.2f,%0.2f,%0.2f\n",wwpoint[0],wpoint[0],cpoint[0],epoint[0],eepoint[0]);
    //West
    flimiter(temp_left,wpoint,cpoint,wpoint,Pv[2]-Pv[1],Pv[1]-Pv[0]); //,num,den
    flimiter(temp_right,cpoint,wpoint,cpoint,Pv[2]-Pv[1],Pv[3]-Pv[2]); //,den,num
    // printf("%f,%f,%d\n",temp_left[0],temp_right[0],idx);
    // printf("%f,%f\n",temp_left[1],temp_right[1]);
    // printf("%f,%f\n",temp_left[2],temp_right[2]);
    // printf("%f,%f\n",temp_left[3],temp_right[3]);
    efluxx(dfdx,temp_left,temp_right,ONE);
    espectral(dfdx,temp_left,temp_right,spi,ONE);
    // printf("%f,%f,%f,%f\n", dfdx[0],dfdx[1],dfdx[2],dfdx[3]);
    // //East
    flimiter(temp_left,cpoint,epoint,cpoint,Pv[3]-Pv[2],Pv[2]-Pv[1]); //,num,den
    flimiter(temp_right,epoint,cpoint,epoint,Pv[3]-Pv[2],Pv[4]-Pv[3]); //,den,num
    // // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",temp_left[0],temp_left[1],temp_left[2],temp_left[3]);
    // // // printf("%0.2f,%0.2f,%0.2f,%0.2f\n",temp_right[0],temp_right[1],temp_right[2],temp_right[3]);
    efluxx(dfdx,temp_left,temp_right,-1);
    espectral(dfdx,temp_left,temp_right,spi,-1);
    // Divide flux in half
    for (int i = 0; i < NVC; i++) {
      dfdx[i]/=2;
    }
    // printf("%f,%f,%f,%f\n", dfdx[0],dfdx[1],dfdx[2],dfdx[3]);
}

__device__
void get_dfdy(float *dfdy, float *shared_state, int idx,int idmod)
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
    int spi = 2;  //spectral radius idx

    // Pressure ratio
    Pv[0] = pressure(wwpoint);
    Pv[1] = pressure(wpoint);
    Pv[2] = pressure(cpoint);
    Pv[3] = pressure(epoint);
    Pv[4] = pressure(eepoint);

    //West
    flimiter(temp_left,wpoint,cpoint,wpoint,Pv[2]-Pv[1],Pv[1]-Pv[0]); //,num,den
    flimiter(temp_right,cpoint,wpoint,cpoint,Pv[2]-Pv[1],Pv[3]-Pv[2]); //,den,num

    efluxy(dfdy,temp_left,temp_right,ONE);
    espectral(dfdy,temp_left,temp_right,spi,ONE);
    // printf("%f\n", dfdy[0]);
    //East
    flimiter(temp_left,cpoint,epoint,cpoint,Pv[3]-Pv[2],Pv[2]-Pv[1]); //,num,den
    flimiter(temp_right,epoint,cpoint,epoint,Pv[3]-Pv[2],Pv[4]-Pv[3]); //,den,num

    efluxy(dfdy,temp_left,temp_right,-1);
    espectral(dfdy,temp_left,temp_right,spi,-1);

    //Divide flux in half
    for (int i = 0; i < NVC; i++) {
      dfdy[i]/=2;
    }
    // printf("%f,%f,%f,%f\n", dfdy[0],dfdy[1],dfdy[2],dfdy[3]);
}

__device__
void step(float *shared_state, int idx, int gts)
{
  float dfdx[NVC]={0,0,0,0};
  float dfdy[NVC]={0,0,0,0};
  get_dfdx(dfdx,shared_state,idx,(2*OPS+blockDim.y)); //Need plus 2 ops in actual setting
  get_dfdy(dfdy,shared_state,idx,(1)); //Need plus 2 ops in actual setting
  //These variables determine predictor or corrector
  bool cond = ((gts+1)%TSO==0);
  int sidx =  cond ? 1 : 0;
  float coeff = cond ? 1.0 : 0.5; //Corrector step
  __syncthreads();
  // printf("%f\n", coeff);
  for (int i = 0; i < NVC; i++)
  {
      shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS-STS*sidx]+coeff*(DTDX*dfdx[i]+DTDY*dfdy[i]);
  }
}

/*
    This function is for testing purposes
*/

__device__
int test_idxs(){
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
return threadId;
}

__global__
void test_step(float *shared_state, int idx, int gts)
{
    float dfdx[NVC]={0,0,0,0};
    float dfdy[NVC]={0,0,0,0};

    idx = test_idxs()+STS;
    // printf("%f\n", shared_state[idx]);
    get_dfdx(dfdx,shared_state,idx,(blockDim.y*gridDim.x)); //Need plus 2 ops in actual setting
    get_dfdy(dfdy,shared_state,idx,(1)); //Need plus 2 ops in actual setting
    // printf("%d\n", gridDim.x);
    // printf("%f,%f,%f,%f\n", dfdy[0],dfdy[1],dfdy[2],dfdy[3]);
    //These variables determine predictor or corrector
    bool cond = ((gts+1)%TSO==0);
    int sidx =  cond ? 1 : 0;
    float coeff = cond ? 1.0 : 0.5; //Corrector step
    __syncthreads();
    for (int i = 0; i < NVC; i++)
    {
        shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS-STS*sidx]+coeff*(0.1*dfdx[i]+0.1*dfdy[i]);
    }
}
