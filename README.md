
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
        """Use this function to set cpu global variables"""
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
