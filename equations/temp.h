

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
void espectral(float* flux,float *left_state, float *right_state, int dim,int dir)
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
}
