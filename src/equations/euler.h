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
__device__ __constant__ float GAMMA; //Gamma
__device__ __constant__ float GAM_M1;
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
  float rho = point[0];
  float rho_inv = 1/rho;
  float vss = point[1]*point[1]*rho_inv+point[2]*point[2]*rho_inv;
  return GAM_M1*(point[3]-HALF*vss);
}

__device__
float spec_press(float *point)
{
  float rho = point[0];
  float vss = point[1]*point[1]+point[2]*point[2];
  return GAM_M1*(rho*point[3]-HALF*rho*vss);
}

/*
  Use this function to find the pressure ratio around 3 points
*/
__device__
float  pressureRatio(float *wpoint,float *point,float *epoint)
{

  float p = pressure(point);
  float Pr=0.;
  float den=0.;
  Pr = pressure(epoint)-p;
  den = (p-pressure(wpoint));
  //This loop is to handle error
  if (den<1.0e-6)
  {
     den = 0;
  }
  Pr /= den;
  return Pr;
}
/*
  This is the minmod flux limiter method for handling discontinuities
*/
__device__
void  flimiter(float *temp_state, float *left_point, float *right_point , float Pr)
{
    for (int i = 0; i < NVC; i++)
    {
        temp_state[i] = left_point[i];
    }

    if (!(ISNAN(Pr)) && (Pr>1.0e-8))
    {
        float coef = HALF*MIN(Pr, ONE);
        for (int j = 0; j < NVC; j++)
        {

            temp_state[j] +=  coef*(left_point[j] - right_point[j]);
        }
    }
    __syncthreads();    //May or may not help thread divergence
}
/*
  Use this function to apply the roe average and obtain the spectral radius
*/
__device__
void espectral(float* flux,float *left_state, float *right_state, int dir, int dim)
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
  float rs = dir*(SQUAREROOT(GAMMA*spec_press(spec_state)/spec_state[0])+ABSOLUTE(spec_state[dim]));
  for (int i = 0; i < NVC; i++)
  {
    flux[i] += rs*(left_state[i]-right_state[i]);
  }
}
/*
Use this function to obtain the flux for each system variable
*/
__device__
void  efluxx(float *flux, float *left_state, float *right_state, int dir)
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
  Use this function to determine the flux in the x direction
*/
__device__
void get_dfdx(float *dfdx, float *shared_state, int idx)
{
  //Constants and Initializing
    float cpoint[NVC];
    getPoint(cpoint,shared_state,idx);
    float epoint[NVC];
    getPoint(epoint,shared_state,idx+(2*OPS+blockDim.y));
    float wpoint[NVC];
    getPoint(wpoint,shared_state,idx-(2*OPS+blockDim.y));
    float eepoint[NVC];
    getPoint(eepoint,shared_state,idx+2*(2*OPS+blockDim.y));
    float wwpoint[NVC];
    getPoint(wwpoint,shared_state,idx-2*(2*OPS+blockDim.y));
    float Pr[3]={0};
    float temp_left[NVC]={0};
    float temp_right[NVC]={0};
    int spi = 1;  //spectral radius idx

    // Pressure ratio
    Pr[0] = pressureRatio(wwpoint,wpoint,cpoint);
    Pr[1] = pressureRatio(wpoint,cpoint,epoint);
    Pr[2] = pressureRatio(cpoint,epoint,eepoint);

    //West
    flimiter(temp_left,wpoint,cpoint,Pr[0]);
    flimiter(temp_right,cpoint,wpoint,ONE/Pr[1]);
    efluxx(dfdx,temp_left,temp_right,ONE);
    espectral(dfdx,temp_left,temp_right,ONE,spi);
    //East
    flimiter(temp_left,cpoint,epoint,Pr[1]);
    flimiter(temp_right,epoint,cpoint,ONE/Pr[2]);
    efluxx(dfdx,temp_left,temp_right,-ONE);
    espectral(dfdx,temp_left,temp_right,-ONE,spi);
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
  flux[0] += dir*(left_state[1]+right_state[1]);
  flux[1] += dir*(left_state[2]*uL+right_state[2]*uR);
  flux[2] += dir*(left_state[2]*vL+pL+right_state[2]*vR+pR);
  flux[3] += dir*((left_state[3]+pL)*uL+(right_state[3]+pR)*uR);
 }
/*
  Use this function to determine the flux in the x direction
*/
__device__
void get_dfdy(float *dfdy, float *shared_state, int idx)
{
  //Constants and Initializing
    float cpoint[NVC];
    getPoint(cpoint,shared_state,idx);
    float epoint[NVC];
    getPoint(epoint,shared_state,idx+1);
    float wpoint[NVC];
    getPoint(wpoint,shared_state,idx-1);
    float eepoint[NVC];
    getPoint(eepoint,shared_state,idx+2);
    float wwpoint[NVC];
    getPoint(wwpoint,shared_state,idx-2);

    float Pr[3]={0};
    float temp_left[NVC]={0};
    float temp_right[NVC]={0};
    int spi = 2;  //spectral radius idx

    // Pressure ratio
    Pr[0] = pressureRatio(wwpoint,wpoint,cpoint);
    Pr[1] = pressureRatio(wpoint,cpoint,epoint);
    Pr[2] = pressureRatio(cpoint,epoint,eepoint);

    // West
    flimiter(temp_left,wpoint,cpoint,Pr[0]);
    flimiter(temp_right,cpoint,wpoint,ONE/Pr[1]);
    efluxy(dfdy,temp_left,temp_right,ONE);
    espectral(dfdy,temp_left,temp_right,ONE,spi);

    //East
    flimiter(temp_left,cpoint,epoint,Pr[1]);
    flimiter(temp_right,epoint,cpoint,ONE/Pr[2]);
    efluxy(dfdy,temp_left,temp_right,-ONE);
    espectral(dfdy,temp_left,temp_right,-ONE,spi);
}

__device__
void step(float *shared_state, int idx, int gts)
{
    float dfdx[NVC]={0,0,0,0};
    float dfdy[NVC]={0,0,0,0};
    get_dfdy(dfdy,shared_state,idx);
    get_dfdx(dfdx,shared_state,idx);
    __syncthreads();
    if ((gts+1)%TSO==0) //Corrector step
    {
        for (int i = 0; i < NVC; i++)
        {
            shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS-STS]+HALF*(DTDX*dfdx[i]+DTDY*dfdy[i]);
        }
    }
    else //Predictor step
    {
        for (int i = 0; i < NVC; i++)
        {
            shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS]+QUARTER*(DTDX*dfdx[i]+DTDY*dfdy[i]);
        }
    }
}
/*
    This function is for testing purposes
*/
__global__
void test_step(float *shared_state, int idx, int gts)
{
  if (threadIdx.x == 0 && threadIdx.y==0) {
      float dfdx[NVC]={0,0,0,0};
      float dfdy[NVC]={0,0,0,0};
      get_dfdy(dfdy,shared_state,idx);
      get_dfdx(dfdx,shared_state,idx);
      __syncthreads();
      if ((gts+1)%TSO==0) //Corrector step
      {
          for (int i = 0; i < NVC; i++)
          {
              shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS-STS]+HALF*(DTDX*dfdx[i]+DTDY*dfdy[i]);
          }
      }
      else //Predictor step
      {
          for (int i = 0; i < NVC; i++)
          {
              shared_state[idx+i*SGIDS]=shared_state[idx+i*SGIDS]+QUARTER*(DTDX*dfdx[i]+DTDY*dfdy[i]);
          }
      }
    }
}
