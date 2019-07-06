
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
__device__ float ***xyvArray()
{
    float ***shared_state = new float**[nx];
    for(int i =0; i<nx; i++){
       shared_state[i] = new float*[ny];
       for(int j =0; j<ny; j++){
           shared_state[i][j] = new float[nv];
       }
    }
    return shared_state;
}
