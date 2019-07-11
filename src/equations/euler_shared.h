/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
#define SQUAREROOT(x)   sqrt(x)
#define ABSOLUTE(x)   abs(x)

__device__ __constant__  const int SS=5; //stencil size
__device__ __constant__ const int NVC=4; //Number of variables
__device__ __constant__ const float GAMMA=1.4; //Gamma
__device__ __constant__ const float GAM_M1=GAMMA-1;
__device__ __constant__ const int NR = 3;
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
float pressure(float *shared_state, int idx)
{
  float rho = shared_state[idx];
  float rho_inv = 1/rho;
  float vss = shared_state[idx+1*SGIDS]*shared_state[idx+1*SGIDS]*rho_inv+shared_state[idx+2*SGIDS]*shared_state[idx+2*SGIDS]*rho_inv;
  return GAM_M1*(point[3]-HALF*rho*vss);
}

/*
  Use this function to find the pressure ratio around 3 points
*/
__device__
float  pressureRatio(float *shared_state, int idx, int shift)
{
  float Pr;
  float p = pressure(point);
  Pr = pressure(epoint)-p;
  Pr /= p-pressure(wpoint);
  return Pr;
}
/*
  This is the minmod flux limiter method for handling discontinuities
*/
__device__
void  flimiter(float *temp_state, float *left_point, float *right_point , float Pr)
{

  if (isnan(Pr) || (Pr<1.0e-8)){
    for (int i = 0; i < NVC; i++)
    {
        temp_state[i] = left_point[i];
    }
  }
  else
  {
    for (int j = 0; j < NVC; j++)
    {
        temp_state[j] = (left_point[j] + HALF*min(Pr, ONE)*(left_point[j] - right_point[j]));
    }
  }
}

/*
Use this function to obtain the flux for each system variable
*/
__device__
void  efluxx(float* flux, float *left_state, float *right_state, int dir)
{
  float rhoL = left_state[0];
  float rhoR = right_state[0];
  float uL = left_state[1]/rhoL;
  float uR = right_state[1]/rhoR;
  float vL = left_state[2]/rhoL;
  float vR = right_state[2]/rhoR;
  // __syncthreads();
  float pL = pressure(left_state);
  float pR = pressure(right_state);
  flux[0] += dir*(left_state[1]+right_state[1]);
  flux[1] += dir*(left_state[1]*uL+pL+right_state[1]*uR+pR);
  flux[2] += dir*(left_state[1]*vL+right_state[1]*vR);
  flux[3] += dir*((left_state[3]+pL)*uL+(right_state[3]+pR)*uR);
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
  float rootrhoR = SQUAREROOT(left_state[1]);
  spec_state[0] = rootrhoL*rootrhoR;
  float denom = 1/(spec_state[0]);
  for (int i = 1; i < NVC; i++)
  {
    spec_state[i] += rootrhoL*left_state[i]/left_state[0];
    spec_state[i] += rootrhoR*right_state[i]*right_state[0];
    spec_state[i] *= spec_state[0]*denom; //Puts in flux form to find pressure
  }
  //Updating flux with spectral value
  float rs = dir*(SQUAREROOT(GAMMA*pressure(spec_state)/spec_state[0])+ABSOLUTE(spec_state[dim]/spec_state[0]));
  for (int i = 0; i < NVC; i++)
  {
    flux[i] += rs*(left_state[i]-right_state[i]);
  }
}

/*
  Use this function to determine the flux in the x direction
*/
__device__
void get_dfdx(float *shared_state, int idx)
{
  //Constants and Initializing
    float cpoint[NVC];
    getPoint(cpoint,shared_state,idx);
    float epoint[NVC];
    getPoint(epoint,shared_state,idx+blockDim.y);
    float wpoint[NVC];
    getPoint(wpoint,shared_state,idx-blockDim.y);
    float eepoint[NVC];
    getPoint(eepoint,shared_state,idx+2*blockDim.y);
    float wwpoint[NVC];
    getPoint(wwpoint,shared_state,idx-2*blockDim.y);
    float Pr[NR]={0};
    float temp_left[NVC]={0};
    float temp_right[NVC]={0};
    int spi = 1;  //spectral radius idx
    //Pressure ratio
    Pr[0] = pressureRatio(wwpoint,wpoint,cpoint);
    Pr[1] = pressureRatio(wpoint,cpoint,epoint);
    Pr[2] = pressureRatio(cpoint,epoint,eepoint);

    // //West
    // flimiter(temp_left,wpoint,cpoint,Pr[0]);
    // flimiter(temp_right,cpoint,wpoint,ONE/Pr[1]);
    // efluxx(dfdx,temp_left,temp_right,ONE);
    // espectral(dfdx,temp_left,temp_right,ONE,spi);
    // //
    // //East
    // flimiter(temp_left,cpoint,epoint,Pr[1]);
    // flimiter(temp_right,epoint,cpoint,ONE/Pr[2]);
    // efluxx(dfdx,temp_left,temp_right,-ONE);
    // espectral(dfdx,temp_left,temp_right,ONE,spi);
}
/*
Use this function to obtain the flux for each system variable
*/
__device__
void  efluxy(float* flux,float *left_state, float *right_state, int dir)
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
    float Pr[NR]={0};
    float temp_left[NVC]={0};
    float temp_right[NVC]={0};
    int spi = 2;  //spectral radius idx
    //Pressure ratio
    Pr[0] = pressureRatio(wwpoint,wpoint,cpoint);
    Pr[1] = pressureRatio(wpoint,cpoint,epoint);
    Pr[2] = pressureRatio(cpoint,epoint,eepoint);

    //West
    flimiter(temp_left,wpoint,cpoint,Pr[0]);
    flimiter(temp_right,cpoint,wpoint,ONE/Pr[1]);
    efluxy(dfdy,temp_left,temp_right,ONE);
    espectral(dfdy,temp_left,temp_right,ONE,spi);

    //East
    flimiter(temp_left,cpoint,epoint,Pr[1]);
    flimiter(temp_right,epoint,cpoint,ONE/Pr[2]);
    efluxy(dfdy,temp_left,temp_right,-ONE);
    espectral(dfdy,temp_left,temp_right,ONE,spi);
}

__device__
void step(float *shared_state, int idx)
{
// printf("%d\n",sizeof(shared_state));
  // float flux[NVC]={0};
  // get_dfdy(dfdxy,shared_state,idx);
  get_dfdx(shared_state,idx);

}
