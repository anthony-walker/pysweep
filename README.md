# PySweep

This is a package containing the functions to implement the swept space time decomposition rule.


# Approach

### 1.) Determine System Characteristics
    This includes the number of CPUs and GPUs in system which will dictate how work is divided up.

### 2.) MPI Initialization
    This step is basically setting up MPI data required for parallelism on a cluster.

### 3.) Divide up based on the system and send data to all CPU cores/GPUs.
    In this step it would probably be beneficial to have a rank that does asynchronous calculations on the GPU and it's own set of calculations.

### 4.) OpenPyramid Calculation
    Run Pyramid Calculation.

### 5. Cycle: Communicate and Octahedron Calculation
    This should cycle between Octahedron and communicating

### 6. Run ClosePyramid
    This is the closing pyramid of the swept rule.
