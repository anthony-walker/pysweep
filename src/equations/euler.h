//Programmer: Anthony Walker
/**
    The equation specific functions.
*/

/**
    The equations specific global variables and function prototypes.
*/

// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_runtime_api.h>
// // #include <device_functions.h>
// #include <mpi.h>
// #include <string>
// #include <vector>
// #include <algorithm>
// #include<array>
// #include <iostream>
// #include <fstream>
// #include <ostream>
// #include <istream>
// #include <cstdio>
// #include <cstdlib>
// #include <cmath>


#define ZERO            0.0
#define QUARTER         0.25
#define HALF            0.5
#define ONE             1.0
#define TWO             2.0


// using namespace std;

struct element //A struct for storing the flux data at any given point in time or space
{
    float rho;
    float rhou;
    float rhov;
    float rhoe;
};

  // __device__ __host__
  void step(float *state, float *iidx, const int ts)
  {


  }

// __device__ __host__
void  dfdxy(float *state, float* idx, const int ts)
  {

  }

// __device__ __host__
void  pressureRatio(float *state)
{




}

// __device__ __host__
void  pressure(float *q)
{




}

// __device__ __host__
void  direction_flux(float *state, float *Pr, bool xy)
{




}

// __device__ __host__
void  flimiter(float *qL, float *qR, float *Pr)
{




}

// __device__ __host__
void  eflux(float *left_state, float *right_state, bool xy)
{




}

// __device__ __host__
void  espectral(float *left_state, float *right_state, bool xy)
{




}



// int main(int argc, char const *argv[])
// {
//     const int d1 = 10;
//     const int d2 = 10;
//     const int d3 = 5;
//     const int d4 = 4;
//
//     element myElement;
//     myElement.rho = 1.01;
//     myElement.rhou = 2.01;
//     myElement.rhov= 3.01;
//     myElement.rhoe = 4.01;
//
//
//     // float *y0 = new float[d1][d2][d3];
//     //
//     // y0[1][1][1] = 100.0;
//     //
//     // for (int i = 0; i < d1; i++) {
//     //     for (int j = 0; j < d2; j++) {
 /*//     //        printf("%f\n",y0[i][j][0]);*/
//     //     }
//     // }
//
//
//
//
//
//
//     return 0;
// }
