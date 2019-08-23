# PySweep

This is a package containing the functions to implement the swept space time decomposition rule in 2 dimensions on
heterogeneous computing architecture.

# Constraints
- The grid used is uniform and rectangular.
- block_size should be (2^n,2^n,1) and constrained by your GPU. Note the block_size should have the same x and y dimension.


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
    This is the closing pyramid of the swept rule.
