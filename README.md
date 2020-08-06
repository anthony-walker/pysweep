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
- block_size should be 2^n and constrained by your GPU. It is also constrained by the relation mod(b,2*k)==0 where k is the number of points on each side of the operating point, e.g., 2 for a 5 point stencil.
- A total of three functions must be named accordingly and take specific arguments.
- This code is currently limited to periodic boundary conditions


The first two functions are python functions, a `step` function and a `set_globals` function. These functions will be used as they are named to take a step and to set global variables for the simulation; simple examples are provided below.

```python

#Step function in example.py
def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy,scheme #true for one step
    t0,tf,dt,dx,dy,scheme = args
    if source_mod is not None:
        keys = "DT","DX","DY"
        nargs = args[2:]
        fc = lambda x:numpy.float32(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
        ckey,_ = source_mod.get_global("SCHEME")
        cuda.memcpy_htod(ckey,numpy.intc(scheme))
```

```python
#set_globals in example.py
def set_globals(*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dtdx,dtdy,gamma,gM1
    t0,tf,dt,dx,dy,gamma = args
    dtdx = dtdy = dt/dx
    gM1 = gamma-1
    if source_mod is not None:
        keys = "DT","DX","DY","GAMMA","GAM_M1","DTDX","DTDY"
        nargs = args[2:]+(gamma-1,dt/dx,dt/dy)
        fc = lambda x:numpy.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
```

The final function is a CUDA function for the GPU.
```cpp
//code in example.cu
__device__ __constant__  double DX;
__device__ __constant__  double DY;
__device__ __constant__  double DT;
__device__ __constant__ int SCHEME; //Determine which scheme to use

__device__
void step(double * shared_state, int idx, int globalTimeStep)
{
  if (SCHEME)
  {
    shared_state[idx+TIMES]=shared_state[idx]+1;
  }
  else
  {
    bool cond = ((globalTimeStep)%ITS==0); //True on complete steps not intermediate
    int timeChange = cond ? 1 : 0; //If true rolls back a time step
    int addition = cond ? 2 : 1; //If true uses 2 instead of 1 for intermediate step
    shared_state[idx+TIMES]=shared_state[idx-TIMES*timeChange]+addition;
  }
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

### Commandline interface
This code is setup to run two examples from the command line. Running `pysweep -h` will give the full list of options. To run an example with MPI use something like 
shell
```
mpiexec -n 2 pysweep -f heat -nx 16 -nt 10 -b 8 -s 1 --swept --verbose
```


### General Approach

This code is intended to implement the swept-time rule in two dimensions and is paired with a standard solver for comparison. 