# PySweep

This is a package containing the functions to implement the swept space time decomposition rule in 2 dimensions on
heterogeneous computing architecture.

### Installation

The plan is to make this pip and conda installable.

#### Dependencies

PySweep depends heavily on HDF5, h5py, MPI, and mpi4py. HDF5 has to be built in parallel to function properly. First install MPI/mpi4py through whatever means you prefer, e.g., conda. I personally built mpich ([guide](https://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2-installguide.pdf)) and mpi4py from source.

I installed mpich from the source directory as
```shell
./configure -prefix=/opt/mpich --with-device=ch4:ofi --disable-fortran |& tee c.txt
make 2>&1 | tee m.txt
make install |& tee mi.txt
```

I installed mpi4py from the source directory as
```shell
python setup.py build --mpicc=/opt/mpich/bin/mpicc
python setup.py install
```

Then get the source distribution of HDF5 and h5py. Install HDF5 first via
```shell
export CC=/opt/mpich/bin/mpicc #export CC path - change as is needed
./configure --enable-parallel --enable-shared --prefix=/opt/hdf5-1.12.0/ #configure from hdf5 source dir
make
make check
make install #You may or may not need sudo for this depending on your prefix
```
Next, install h5py from the source directory ([git](https://github.com/h5py/h5py)) via
```shell
export CC=/opt/mpich/bin/mpicc #doesn't need repeated if done in previous step and the same shell
export HDF5_MPI="ON"
python setup.py configure --hdf5=/opt/hdf5-1.12.0
python setup.py build
python setup.py install
```


# Constraints
- The grid used is uniform and rectangular.
- block_size should be 2^n and constrained by your GPU. It is also constrained by the relation mod(b,2*k)==0.
- A total of three functions must be named accordingly and take specific arguments.
- This code is currently limited to periodic boundary conditions


The first two functions are python functions, a `step` function and a `set_globals` function. These functions will be used as they are named to take a step and to set global variables for the simulation; simple examples are provided below.

```python

#Step function in example.py

def step(state,iidx,arrayTimeIdx,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIdx - index to be used to update the state array appropriately
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    vSlice = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (ts+1,vSlice,idx,idy)  #next step index
        cidx = (ts,vSlice,idx,idy)
        state[ntidx] = state[cidx]+1
    return state

```

```python
#set_globals in example.py

def set(state,iidx,arrayTimeIdx,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    arrayTimeIdx - index to be used to update the state array appropriately
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    vSlice = slice(0,state.shape[1],1)
    for idx,idy in iidx:
        ntidx = (ts+1,vSlice,idx,idy)  #next step index
        cidx = (ts,vSlice,idx,idy)
        state[ntidx] = state[cidx]+1
    return state
```

The final function is a CUDA function for the GPU.
```cpp
//code in example.cu
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ const int NVC=1; //Number of variables

__device__
void getPoint(double * curr_point,double *shared_state, int idx)
{
    curr_point[0]=shared_state[idx];
}

__device__
void step(double * shared_state, int idx, int globalTimeStep)
{
  double cpoint[NVC];
  getPoint(cpoint,shared_state,idx);
  shared_state[idx+TIMES]=cpoint[0]+1;
}

```

# Usage of the code

The main entry point for the code is a script paired with an input yaml file. Staying with the current example, `example.yaml` could look like this

```yaml
  swept: True
  filename: "example.hdf5"
  verbose: True
  blocksize: 8
  share: 0.5
  dtype: float64
  globals:
    - 0
    - 10
    - 0.1
    - 0.1
    - 0.1
  intermediate_steps: 1
  operating_points: 1
  cpu: ./pysweep-git/pysweep/equations/example.py
  gpu: ./pysweep-git/pysweep/equations/example.cu
```

The associated script portion would look like
```python
    filename = pysweep.equations.example.createInitialConditions(1,384,384)
    yfile = os.path.join(path,"inputs")
    yfile = os.path.join(yfile,"example.yaml")
    testSolver = pysweep.Solver(filename,yfile)
    testSolver.simulation=True
    testSolver()
```

The output of calling the solver object will look something like this

```shell
Setting up processes...

Creating time step data...

Creating output file...

Creating shared memory arrays and process functions...

Cleaning up solver...

Running simulation...

Pysweep simulation properties:
	simulation type: swept
	number of processes: 2
	gpu source: ./pysweep-git/pysweep/equations/example.cu
	cpu source: /home/anthony-walker/nrg-swept-project/pysweep-git/pysweep/equations/example.py
	output file: example.hdf5
	verbose: True
	Array shape: (103, 1, 384, 384)
	dtype: float64
	share: 0.5
	blocksize: 8
	stencil size: 3
	intermediate time steps: 1
	time data (t0,tf,dt): (0,10,0.1)

Cleaning up processes...

Done in 7.3623316287994385 seconds...
```

### General Approach

This code is intended to implement the swept-time rule in two dimensions and is paired with a standard solver for comparison. 