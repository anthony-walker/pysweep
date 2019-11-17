
# PySweep

This is a package containing the functions to implement the swept space-time decomposition rule for solving unsteady PDEs in 2 dimensions on
heterogeneous computing architecture.

# Constraints
- The grid used is structured.
- block_size is constrained to  <em>b = (2nf)<sup>2</sup></em> and constrained by your GPU. Note the block_size should have the same x and y dimension.
- A total of three functions must be named accordingly and take specific arguments.
- This code is currently limited to periodic boundary conditions.

### Using PySweep

PySweep requires 3 specifically named functions be created and the file path to these functions be given. Python functions should be put into 1 function and CUDA functions should be put into the other.

The Python functions should look something like this:

```python
    def step(state,iidx,ts,gts):
        """This is the method that will be called by the swept solver.
        state - 4D numpy array(t,v,x,y (v is variables length))
        iidx -  an iterable of indexs
        ts - the current time step
        gts - a step counter that allows implementation of the scheme
        """
        half = 0.5
        TSO = 2
        vSlice = slice(0,state.shape[1],1)
        for idx,idy in iidx:
            ntidx = (ts+1,vSlice,idx,idy)  #next step index
            cidx = (ts,vSlice,idx,idy)
            pidx = (ts-1,vSlice,idx,idy) #next step index
            if (gts+1)%TSO==0: #Corrector
                state[ntidx] = (state[pidx])+2
            else: #Predictor
                state[ntidx] = (state[cidx])+1
        return state

    def set_globals(gpu,source_mod,*args):
        """Use this function to set cpu and gpu global variables"""
        t0,tf,dt,dx,dy,gam = args
        if gpu:
            keys = "DT","DX","DY","GAMMA","GAM_M1"
            nargs = args[2:]+(gam-1,)
            fc = lambda x:np.float32(x)
            for i,key in enumerate(keys):
                ckey,_ = source_mod.get_global(key)
                cuda.memcpy_htod(ckey,fc(nargs[i]))
        else:
            global dtdx
            dtdx = dt/dx
            global dtdy
            dtdy = dt/dy
            global gamma
            gamma = gam
```  
The function names and inputs should be identical to this.

The CUDA kernel should look something like this:

```c++
    __device__
    void step(float * shared_state, int idx, int gts)
    {

      float tval[NVC]={0,0,0,0};
       __syncthreads();

      if ((gts+1)%TSO==0) //Corrector step
      {
          for (int i = 0; i < NVC; i++)
          {
              tval[i] = shared_state[idx-STS]+2;
          }
      }
      else //Predictor
      {

          for (int i = 0; i < NVC; i++)
          {
              tval[i] = shared_state[idx]+1;

          }
      }
      __syncthreads();

      for (int i = 0; i < NVC; i++)
      {
          shared_state[idx+i*SGIDS]=tval[i];
      }
    }

```

After your functions are created, you can set up your problem. It will look something like this.

```python
  import numpy as np
  from pysweep.distributed.sweep.distsweep import dsweep
  from pysweep.distributed.decomposition.ddecomp import ddecomp
  from pysweep.analytical.ahde import TIC

  if __name__ == "__main__":
      #Creating input array
      nx = ny = 120
      arr = np.ones((1,nx,ny))
      #Spatial Inputs
      X = Y = 10
      dx = X/nx
      dy = Y/ny
      #Numerical inputs
      tso = 2 #Number of steps taken by temporal scheme
      ops = 1 #Number of points on one side of the current point being solved
      #GPU inputs
      bs = 12 #Block size
      aff = 0.9 #GPU affinity
      #Heat Diffusion Stability Considerations
      alpha = 5
      Fo = 0.24
      #Time Variables
      t0 = 0 #Initial time
      dt = Fo*(X/nx)**2/alpha
      tf = dt*500
      #Filling input array with initial conditions
      arr = TIC(nx,ny,X,Y,1,373,298)[0,:,:,:]
      #Arguments for solvers
      gargs = (t0,tf,dt,dx,dy,alpha)
      swargs = (tso,ops,bs,aff,"./pysweep/equations/hde.h","./pysweep/equations/hde.py")
      #This will solve using the swept solver
      dsweep(arr,gargs,swargs,filename="test",exid=[])
      #This will solve using the standard
      ddecomp(arr,gargs,swargs,filename="test",exid=[])
```

You can also modify the pst.py file in Pysweep to execute from the command-line.
One pre-programmed example looks like this:

```
  mpiexec -n 1 python ./pysweep/pst.py swept_vortex -b 12 -a 0.5 --hdf5 \"MyResults\"

```

There is also solvers under `pysweep.node` that use different strategies on a single node but these solvers have not been fully tested or validated. The performance of the distributed solver tends to be better anyways so it is the recommended package to use.

PySweep was validated using the Euler Vortex problem in 2 dimensions. There are tests in `pysweep.tests` that can demonstrate this if desired.
