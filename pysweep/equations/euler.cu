/*
    Programmer: Anthony Walker
    The equation specific functions (2D Euler).
*/
#define SQUAREROOT(x)   sqrt(x)
#define ABSOLUTE(x)   abs(x)
#define MIN(x,y)   min(x,y)
#define ISNAN(x)   isnan(x)
#define ISINF(x)   isinf(x)

__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ double DTDX;
__device__ __constant__ double DTDY;
__device__ __constant__ double GAMMA; //Gamma
__device__ __constant__ double GAM_M1;
__device__ __constant__ const int NVC=4; //Number of variables
/*
  Use this function to get point data
*/
__device__
void getPoint(double * curr_point,double *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
    curr_point[1]=shared_state[idx+VARS];
    curr_point[2]=shared_state[idx+VARS*2];
    curr_point[3]=shared_state[idx+VARS*3];
}

/*
  Use this function to determine the pressure
*/
__device__
double pressure(double *point)
{
  double vss = (point[1]*point[1]+point[2]*point[2])/point[0];
  return GAM_M1*(point[3]-0.5*vss);
}
/*
  Use this function to determine the pressure for spectral function
*/
__device__
double spec_press(double *point)
{
  double rho = point[0];
  double vss = point[1]*point[1]+point[2]*point[2];
  return GAM_M1*(rho*point[3]-0.5*rho*vss);
}

/*
  Use this function to apply the roe average and obtain the spectral radius
*/
__device__
void espectral(double* flux,double *left_state, double *right_state, int dim, int dir)
{
  //Initializing values and determining spectral state
  double spec_state[NVC]={0};
  double rootrhoL = SQUAREROOT(left_state[0]);
  double rootrhoR = SQUAREROOT(right_state[0]);
  spec_state[0] = rootrhoL*rootrhoR;
  double denom = 1/(rootrhoL+rootrhoR);

  for (int i = 1; i < NVC; i++)
  {
    spec_state[i] += rootrhoL*left_state[i]/left_state[0];
    spec_state[i] += rootrhoR*right_state[i]/right_state[0];
    spec_state[i] *= denom; //Puts in flux form to find pressure
  }

  //Updating flux with spectral value
  double rs = (SQUAREROOT(GAMMA*spec_press(spec_state)/spec_state[0])+ABSOLUTE(spec_state[dim]));
  rs = ISNAN(rs) ? 0 : rs;
  for (int i = 0; i < NVC; i++)
  {
    flux[i] += dir*rs*(left_state[i]-right_state[i]);
  }
}

/*
  Use this function to determine the flux in the x direction
*/
__device__
void  efluxx(double *flux, double *left_state, double *right_state,int dir)
{
  double rhoL = left_state[0];
  double rhoR = right_state[0];
  double uL = left_state[1]/rhoL;
  double uR = right_state[1]/rhoR;
  double vL = left_state[2]/rhoL;
  double vR = right_state[2]/rhoR;
  double pL = pressure(left_state);
  double pR = pressure(right_state);
  flux[0] += dir*(left_state[1]+right_state[1]);
  flux[1] += dir*(left_state[1]*uL+pL+right_state[1]*uR+pR);
  flux[2] += dir*(left_state[1]*vL+right_state[1]*vR);
  flux[3] += dir*((left_state[3]+pL)*uL+(right_state[3]+pR)*uR);
 }

 /*
 Use this function to obtain the flux for each system variable
 */
 __device__
 void  efluxy(double* flux,double *left_state, double *right_state, int dir)
 {
   //Initializing variables
   double rhoL = left_state[0];
   double rhoR = right_state[0];
   double uL = left_state[1]/rhoL;
   double uR = right_state[1]/rhoR;
   double vL = left_state[2]/rhoL;
   double vR = right_state[2]/rhoR;
   //Determining pressures
   double pL = pressure(left_state);
   double pR = pressure(right_state);
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
  void  flimiter(double *temp_state,double *point1,double *point2,double *point3,double num, double den)
  {

      double Pr = num/den; //Neeed the extra comparison here for large numbers
      Pr = (ISNAN(Pr) || ISINF(Pr) || Pr > 1e6 || Pr < 0.0) ? 0.0 : Pr;
      Pr = MIN(Pr, 1.0);
      for (int j = 0; j < NVC; j++)
      {
        temp_state[j] =  point1[j]+0.5*Pr*(point2[j] - point3[j]);
      }
  }


__device__
void get_dfdx(double *dfdx, double *shared_state, int idx,int idmod)
{
  //Constants and Initializing
    double cpoint[NVC];
    getPoint(cpoint,shared_state,idx);
    double epoint[NVC];
    getPoint(epoint,shared_state,idx+idmod);
    double wpoint[NVC];
    getPoint(wpoint,shared_state,idx-idmod);
    double eepoint[NVC];
    getPoint(eepoint,shared_state,idx+2*idmod);
    double wwpoint[NVC];
    getPoint(wwpoint,shared_state,idx-2*idmod);
    double Pv[5]={0};
    double temp_left[NVC]={0};
    double temp_right[NVC]={0};

    // Pressure ratio
    Pv[0] = pressure(wwpoint);
    Pv[1] = pressure(wpoint);
    Pv[2] = pressure(cpoint);
    Pv[3] = pressure(epoint);
    Pv[4] = pressure(eepoint);

    //West
    flimiter(temp_left,wpoint,cpoint,wpoint,Pv[2]-Pv[1],Pv[1]-Pv[0]); //,num,den
    flimiter(temp_right,cpoint,wpoint,cpoint,Pv[2]-Pv[1],Pv[3]-Pv[2]); //,den,num
    efluxx(dfdx,temp_left,temp_right,1.0);
    espectral(dfdx,temp_left,temp_right,1,1.0);

    //East
    flimiter(temp_left,cpoint,epoint,cpoint,Pv[3]-Pv[2],Pv[2]-Pv[1]); //,num,den
    flimiter(temp_right,epoint,cpoint,epoint,Pv[3]-Pv[2],Pv[4]-Pv[3]); //,den,num
    efluxx(dfdx,temp_left,temp_right,-1.0);
    espectral(dfdx,temp_left,temp_right,1,-1.0);

    // Divide flux in half
    for (int i = 0; i < NVC; i++) {
      dfdx[i]/=2;
    }
}

__device__
void get_dfdy(double *dfdy, double *shared_state, int idx,int idmod)
{
  //Constants and Initializing
    double cpoint[NVC];
    getPoint(cpoint,shared_state,idx);
    double epoint[NVC];
    getPoint(epoint,shared_state,idx+idmod);
    double wpoint[NVC];
    getPoint(wpoint,shared_state,idx-idmod);
    double eepoint[NVC];
    getPoint(eepoint,shared_state,idx+2*idmod);
    double wwpoint[NVC];
    getPoint(wwpoint,shared_state,idx-2*idmod);
    double Pv[5]={0};
    double temp_left[NVC]={0};
    double temp_right[NVC]={0};
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

    efluxy(dfdy,temp_left,temp_right,1.0);
    espectral(dfdy,temp_left,temp_right,spi,1.0);
    //East
    flimiter(temp_left,cpoint,epoint,cpoint,Pv[3]-Pv[2],Pv[2]-Pv[1]); //,num,den
    flimiter(temp_right,epoint,cpoint,epoint,Pv[3]-Pv[2],Pv[4]-Pv[3]); //,den,num

    efluxy(dfdy,temp_left,temp_right,-1);
    espectral(dfdy,temp_left,temp_right,spi,-1);

    //Divide flux in half
    for (int i = 0; i < NVC; i++) {
      dfdy[i]/=2;
    }
}

__device__
void step(double *state, int idx, int globalTimeStep)
{
  double dfdx[NVC]={0,0,0,0};
  double dfdy[NVC]={0,0,0,0};
  get_dfdx(dfdx,state,idx,(SY)); //Need plus 2 ops in actual setting
  get_dfdy(dfdy,state,idx,(1));
  //These variables determine predictor or corrector
  bool cond = ((globalTimeStep)%ITS==0); //True on complete steps not intermediate
  int timeChange = cond ? 1 : 0; //If true rolls back a time step
  double coeff = cond ? 1 : 0.5; //If true uses 1 instead of 0.5 for intermediate step
  for (int i = 0; i < NVC; i++)
  {
      state[idx+i*VARS+TIMES]=state[idx+i*VARS-TIMES*timeChange]+coeff*(DTDX*dfdx[i]+DTDY*dfdy[i]);
  }
}

__global__
void stepTestInterface(double *state, int tidx, int globalTimeStep)
{
  double dfdx[NVC]={0,0,0,0};
  double dfdy[NVC]={0,0,0,0};
  int idx = threadIdx.y+OPS+(threadIdx.x+OPS)*SY+tidx*TIMES; //This is for test function only
  get_dfdx(dfdx,state,idx,(SY)); //Need plus 2 ops in actual setting
  // printf("%0.8e,%0.8e\n",dfdx[0],dfdy[0]);
  get_dfdy(dfdy,state,idx,(1));
  //These variables determine predictor or corrector
  bool cond = ((globalTimeStep)%ITS==0); //True on complete steps not intermediate
  int timeChange = cond ? 1 : 0; //If true rolls back a time step
  double coeff = cond ? 1 : 0.5; //If true uses 1 instead of 0.5 for intermediate step
  for (int i = 0; i < NVC; i++)
  {
      state[idx+i*VARS+TIMES]=state[idx+i*VARS-TIMES*timeChange]+coeff*(DTDX*dfdx[i]+DTDY*dfdy[i]);
  }
}
