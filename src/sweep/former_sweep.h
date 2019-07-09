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
