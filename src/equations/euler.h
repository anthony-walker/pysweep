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
//
// #include <iostream>
// #include <fstream>
// #include <ostream>
// #include <istream>
//
// #include <cstdio>
// #include <cstdlib>
// #include <cmath>
// #include <string>
// #include <vector>
// #include <algorithm>

// #include "myVectorTypes.h"
// #include "json/json.h"

#define ZERO            0.0
#define QUARTER         0.25
#define HALF            0.5
#define ONE             1.0
#define TWO             2.0

  __device__ __host__
  void step(float *state, float *iidx, const int ts)
  {


  }

__device__ __host__
void  dfdxy(float *state, float* idx, const int ts)
  {

  }

__device__ __host__
void  pressureRatio(float *state)
{




}

__device__ __host__
void  pressure(float *q)
{




}

__device__ __host__
void  direction_flux(float *state, float *Pr, bool xy)
{




}

__device__ __host__
void  flimiter(float *qL, float *qR, float *Pr)
{




}

__device__ __host__
void  eflux(float *left_state, float *right_state, bool xy)
{




}

__device__ __host__
void  espectral(float *left_state, float *right_state, bool xy)
{




}
