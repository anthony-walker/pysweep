# PySweep

This is a package containing the functions to implement the swept space time decomposition rule.


# Approach

### 1.) MPI Initialization

    This step is basically setting up MPI data required for parallelism.

### 2.) GPU Device Querying



### Step 1: Split the data up among the ranks.

    The data is currently split along the axis with the most points.

### Step 2: Check if the rank has a GPU.
