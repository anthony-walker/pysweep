//Programmer: Anthony Walker
//This file contains cuda/c++ source code for launching the pycuda swept rule.

//Constant Memory Values
__device__ __constant__  int mss;

__device__ __constant__  int nx;

__device__ __constant__  int ny;

__device__ __constant__  int nt;

__device__ __constant__  int nv;

__device__ __constant__  int nd;

__device__ __constant__  float dt;

__device__ int getGlobalIdx_2D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y)+ (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

/*
    This is a temporary function for testing step
*/
__device__ void step(float ***shared_state)
{   for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nv; k++)
            {
                // printf("%f\n",shared_state[i][j][k]);
            }
        }
    }
}
/*
    Creating shared array (4 Dimensions (time, x , y, variables))
*/
__device__ float ****txyvArray()
{
    float ****shared_state = new float***[nt];
    for(int i =0; i<nt; i++){
       shared_state[i] = new float**[nx];
       for(int j =0; j<nx; j++){
           shared_state[i][j] = new float*[ny];
           for(int k = 0; k<ny;k++){
              shared_state[i][j][k] = new float[nv];
              for(int h = 0; h<nv;h++){
                 shared_state[i][j][k][h] = 0.0;
              }
           }
       }
    }
    return shared_state;
}

/*
    Use this function to create and return the GPU UpPyramid
*/
__global__ void UpPyramid(float *state)
{
//Creating shared array ptr
__shared__ float ****shared_state;
shared_state = txyvArray();
//Creating indexing
int gid = getGlobalIdx_2D_2D()*nv;  //Getting 2D index and adjusting it for number of variables
int shift = nx*ny*nv; //Use this variable to shift in time
int idxx = gid/(ny*nv); //Getting x index from global, Using type int truncates appropriately
int idxy = threadIdx.y; //Getting y index from global,
int ct = 0; //temporary current time variable
//Filling shared state array with variables initial variables
printf("%d, %d, %d, %d, %d\n",gid,ct,idxx,idxy,0);
// printf("%d, %d, %d, %d, %d\n",gid,ct,idxx,idxy,1);
// printf("%d, %d, %d, %d, %d\n",gid,ct,idxx,idxy,2);
// printf("%d, %d, %d, %d, %d\n",gid,ct,idxx,idxy,3);
__syncthreads();

for (int i = 0; i < nv; i++)
{
    shared_state[ct][idxx][idxy][i]=state[gid+i];
    // printf("%0.2f, %0.2f\n",shared_state[ct][idxx][idxy][i],state[gid+i]);
}
__syncthreads(); //Sync threads here to prevent race conditions

}


// __device__ void step(float shared_state[nx][ny][nv])
// {
//     for (int i = 0; i < nv; i++) {
//         printf("%d, %f\n",i,shared_state[0][0][i]);
//     }
// }
//
//
// __global__ void UpPyramid(float *state)
// {
// __shared__ float shared_state[nt][nx][ny][nv];    //The values here are created at run time via python
//
// float ****ss4 = new float***[nd];
//
// int gid = getGlobalIdx_2D_2D()*nv;  //Getting 2D index and adjusting it for number of variables
// int shift = nx*ny*nv; //Use this variable to shift in time
// int idxx = gid/(ny*nv); //Getting x index from global, Using type int truncates appropriately
// int idxy = threadIdx.y; //Getting y index from global,
//
// int ct = 0; //temporary current time variable
// //Filling shared state array with variables initial variables
// for (int i = 0; i < nv; i++) {
//     shared_state[ct][idxx][idxy][i]=state[gid+i];
// }
// __syncthreads(); //Sync threads here to prevent race conditions
// printf("%d, %d, %d, %d, %f\n",ct,idxx,idxy,0,shared_state[ct][idxx][idxy][0]);
// step(shared_state[0]);
// }
