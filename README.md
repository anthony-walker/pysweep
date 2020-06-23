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

<!--
# Constraints
- The grid used is uniform and rectangular.
- block_size should be (2^n,2^n,1) and constrained by your GPU. Note the block_size should have the same x and y dimension.
- A total of three functions must be named accordingly and take specific arguments.
- This code is currently limited to periodic boundary conditions

### General Approach

#### 1.) Code Inputs

#### 2.) Data Decomposition
    - The code first splits the data to the nearest column based on block_size between the CPUs and GPUs. The number of CPU cores and GPUs is not taken into account in this split. Note, that the provided affinity will be adjusted if it cannot evenly split the data by column.
    - Each region after the affinity split is then split amongst the number of ranks for each architecture. Note, that each GPU will occupy 1 rank (e.g.
    running the code with 16 MPI processes and two GPUs will result in 14 CPU ranks).
    - If there is not sufficient data for the number of ranks supplied, those ranks will be terminated.

### Swept Steps
#### 1.) UpPyramid
    - The "UpPyramid" is the first swept calculation and it creates a pyramid of data in time. At the end of this calculation, the data is written into its standard position for the next step.
    - The appropriate edges are communicated for the shift calculation.
#### 2.) Bridge
    - The "Bridge" is the next step required.
    - The appropriate edges are communicated for the shift calculation.

#### 3.) Standard Octahedron
    - The "Octahedron" is the next calculation after the bridge.

#### 4.) Reverse Bridge
    - The "Bridge" is the next step required.
    - The appropriate edges are communicated for the shift calculation.

#### 5.) Shift Octahedron
    - The "Octahedron" is the next calculation after the bridge.

#### 6. DownPyramid
    This is the closing pyramid of the swept rule. -->

    The latex for the paper outlining the work done in this code can be found [here](https://github.com/Niemeyer-Research-Group/2019-walker-2dswept-camwa).
