__device__ __constant__  int nx;

__device__ __constant__  int ny;

__device__ __constant__  int nt;

__device__ __constant__  int nd;
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
              // for(int h = 0; h<nv;h++){
              //    shared_state[i][j][k][h] = 0.0;
              // }
           }
       }
    }
    return shared_state;
}


/*
    Creating shared array (3 Dimensions (x , y, variables))
*/
__device__ float ***Array3D(int nv)
{
    float ***shared_state = new float**[blockDim.x];
    for(int i =0; i<blockDim.x; i++){
       shared_state[i] = new float*[blockDim.y];
       for(int j =0; j<blockDim.y; j++){
           shared_state[i][j] = new float[nv];
       }
    }
    return shared_state;
}

__device__ float **Array2D(int nv)
{
    int arr_size = blockDim.x*blockDim.y;
    float **shared_array = new float*[arr_size];
    for (int i = 0; i < arr_size; i++)
    {
        shared_array[i] = new float[nv];
    }
    return shared_array;
}


// int idxy = gid%(blockDim.y); //Getting y index from global,
// int idxx = gid/(blockDim.y); //Getting x index from global, Using type int truncates appropriately


__global__ void UpPyramid(float *state)
{
    //Creating flattened shared array ptr (x,y,v) length
    extern __shared__ float shared_state[];    //Shared state specified externally
    //Creating indexing
    int gid = getGlobalIdx_2D_2D();  //Getting 2D global index
    int sgid_shift = blockDim.x*blockDim.y;
    int var_shift = sgid_shift*gridDim.x*gridDim.y;//This is the variable used to shift between values
    int time_shift = var_shift*NV; //Use this variable to shift in time and for sgid
    int sgid = gid%(sgid_shift); //Shared global index
    // Filling shared state array with variables initial variables
    __syncthreads(); //Sync threads here to ensure all initial values are copied

    for (int k = 0; k < 2; k++)
    {
        for (int i = 0; i < NV; i++)
        {
            shared_state[sgid+i*sgid_shift] = state[gid+i*var_shift+k*time_shift];
        }
        __syncthreads(); //Sync threads here to ensure all initial values are copied
        // Solving step function

        step(shared_state,sgid,k);

        __syncthreads();   //Sync threads after solving
        // Place values back in original matrix
        for (int j = 0; j < NV; j++)
        {
            state[gid+j*var_shift+(k+1)*time_shift]=shared_state[sgid+j*sgid_shift];
        }
    }

}


// float **fluxx = new float*[SS];
// float **fluxy = new float*[SS];
// for (int i = 0; i < SS; i++)
// {
//     fluxx[i] = new float[NV];
//     fluxy[i] = new float[NV];
// }
// float fluxy[ss][NV];
