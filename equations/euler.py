#Programmer: Anthony Walker
#This file contains functions to solve the eulers equations in 2 dimensions with
#the swept rule or in a standard way



import numpy as np
import sympy,sys,itertools
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except Exception as e:
    pass
#Testing imports
if __name__ == "__main__":
    import pycuda.autoinit
    import test_euler as e1d
    import warnings, os
    fp = os.path.abspath(__file__)
    path = os.path.dirname(fp)
    sys.path.insert(0, path[:-len('equations')])
    import distributed.sweep.ccore.source as source
    # warnings.filterwarnings('error')
sfs = "%0.2e"
#Testing
def print_line(line):
    sys.stdout.write('[')
    for item in line:
        sys.stdout.write(sfs%item+", ")
    sys.stdout.write(']\n')
#----------------------------------Globals-------------------------------------#
gamma = 0
dtdx = 0
dtdy = 0
gM1 = 0
#----------------------------------End Globals-------------------------------------#

def step(state,iidx,ts,gts):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    gts - a step counter that allows implementation of the scheme
    """
    half = 0.5
    ops=2
    vs = slice(0,state.shape[1],1)
    sidx = ts-1 if (gts+1)%2==0 else ts #scheme index
    coef = 1 if (gts+1)%2==0 else 0.5 #scheme index
    #Making pressure vector
    l1,l2 = tuple(zip(*iidx))
    l1 = range(min(l1)-ops,max(l1)+ops+1,1)
    l2 = range(min(l2)-ops,max(l2)+ops+1,1)
    P = np.zeros(state[0,0,:,:].shape)

    for idx,idy in itertools.product(l1,l2):
        P[idx,idy] = pressure(state[ts,vs,idx,idy])

    for idx,idy in iidx:
        dfdx,dfdy = dfdxy(state,(ts,vs,idx,idy),P)
        state[ts+1,vs,idx,idy] = state[sidx,vs,idx,idy]+coef*(dtdx*dfdx+dtdy*dfdy)
    return state

def set_globals(gpu,*args,source_mod=None):
    """Use this function to set cpu global variables"""
    t0,tf,dt,dx,dy,gam = args
    if gpu:
        keys = "DT","DX","DY","GAMMA","GAM_M1","DTDX","DTDY"
        nargs = args[2:]+(gam-1,dt/dx,dt/dy)
        fc = lambda x:np.float64(x)
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
        global gM1
        gM1 = gam-1.0

#---------------------------------------------Solving functions
def dfdxy(state,idx,P):
    """This method is a five point finite volume method in 2D."""
    #Five point finite volume method
    #Creating indices from given point (idx)
    ops = 2 #number of atomic operations
    i1,i2,i3,i4 = idx
    idxx=(i1,i2,slice(i3-ops,i3+ops+1,1),i4)
    idxy=(i1,i2,i3,slice(i4-ops,i4+ops+1,1))
    #Finding spatial derivatives
    dfdx = direction_flux(state[idxx],True,P[idxx[2:]])
    dfdy = direction_flux(state[idxy],False,P[idxy[2:]])
    return dfdx, dfdy

def direction_flux(state,xy,P):
    """Use this method to determine the flux in a particular direction."""
    ONE = 1    #Constant value of 1
    idx = 2     #This is the index of the point in state (stencil data)
    #Initializing Flux
    flux = np.zeros(len(state[:,idx]))

    #Atomic Operation 1
    tsl = flux_limiter(state,idx-1,idx,P[idx]-P[idx-1],P[idx-1]-P[idx-2])
    tsr = flux_limiter(state,idx,idx-1,P[idx]-P[idx-1],P[idx+1]-P[idx])
    flux += eflux(tsl,tsr,xy)
    flux += espectral(tsl,tsr,xy)
    #Atomic Operation 2
    tsl = flux_limiter(state,idx,idx+1,P[idx+1]-P[idx],P[idx]-P[idx-1])
    tsr = flux_limiter(state,idx+1,idx,P[idx+1]-P[idx],P[idx+2]-P[idx+1])
    flux -= eflux(tsl,tsr,xy)
    flux -= espectral(tsl,tsr,xy)
    return flux*0.5

def flux_limiter(state,idx1,idx2,num,den):
    """This function computers the minmod flux limiter based on pressure ratio"""
    dec = 15
    num = round(num,dec)
    den = round(den,dec)
    if (num > 0 and den > 0) or (num < 0 and den < 0):
        return state[:,idx1]+min(num/den,1)/2*(state[:,idx2]-state[:,idx1])
    else:
        return state[:,idx1]

def pressure(q):
    """Use this function to solve for pressure of the 2D Eulers equations.
    q is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    P = (GAMMA-1)*(rho*e-(1/2)*(rho*u^2+rho*v^2))
    """
    return gM1*(q[3]-(q[1]*q[1]+q[2]*q[2])/(2*q[0]))

def eflux(left_state,right_state,xy):
    """Use this method to calculation the flux.
    q (state) is set up as:
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    #Pressures
    PL = pressure(left_state)
    PR = pressure(right_state)
    #Unpacking
    _,uL,vL,eL = left_state/left_state[0]
    _,uR,vR,eR = right_state/right_state[0]
    #Calculating flux
    #X or Y split
    if xy:  #X flux
        return np.add([left_state[1],left_state[1]*uL+PL,left_state[1]*vL,(left_state[3]+PL)*uL],[right_state[1],right_state[1]*uR+PR,right_state[1]*vR,(right_state[3]+PR)*uR])
    else: #Y flux
        return np.add([left_state[2],left_state[2]*uL,left_state[2]*vL+PL,(left_state[3]+PL)*vL],[right_state[2],right_state[2]*uR,right_state[2]*vR+PR,(right_state[3]+PR)*vR])

def espectral(left_state,right_state,xy):
    """Use this method to compute the Roe Average.
    q(state)
    q[0] = rho
    q[1] = rho*u
    q[2] = rho*v
    q[3] = rho*e
    """
    # print(left_state,right_state)
    spec_state = np.zeros(len(left_state))
    rootrhoL = np.sqrt(left_state[0])
    rootrhoR = np.sqrt(right_state[0])
    tL = left_state/left_state[0] #Temporary variable to access e, u, v, and w - Left
    tR = right_state/right_state[0] #Temporary variable to access e, u, v, and w -  Right
    #Calculations
    denom = 1/(rootrhoL+rootrhoR)
    spec_state[0] = rootrhoL*rootrhoR
    spec_state[1] = (rootrhoL*tL[1]+rootrhoR*tR[1])*denom
    spec_state[2] = (rootrhoL*tL[2]+rootrhoR*tR[2])*denom
    spec_state[3] = (rootrhoL*tL[3]+rootrhoR*tR[3])*denom
    spvec = (spec_state[0],spec_state[0]*spec_state[1],spec_state[0]*spec_state[2],spec_state[0]*spec_state[3])
    P = pressure(spvec)
    dim = 1 if xy else 2    #if true provides u dim else provides v dim
    return (np.sqrt(gamma*P/spec_state[0])+abs(spec_state[dim]))*(left_state-right_state) #Returns the spectral radius *(dQ)

if __name__ == "__main__":
    #Debugging functions
    #Boundary copy
    def bcpy(idx,state,ops):
        state[idx,:,:ops,:] = state[idx,:,-2*ops:-ops,:]
        state[idx,:,-ops:,:] = state[idx,:,ops:2*ops,:]
        state[idx,:,:,:ops] = state[idx,:,:,-2*ops:-ops]
        state[idx,:,:,-ops:] = state[idx,:,:,ops:2*ops]

    def bcpy1D(idx,state,ops):
        state[idx,:,:ops] = state[idx,:,-2*ops:-ops]
        state[idx,:,-ops:] = state[idx,:,ops:2*ops]

    #Printer function
    def pm(arr,i,iv=0,ps="%d"):
        for item in arr[i,iv,:,:]:
            sys.stdout.write("[ ")
            for si in item:
                sys.stdout.write(ps%si+", ")
            sys.stdout.write("]\n")


    def test_euler_shock():
        """This function tests the implemented manufactured solution."""
        #Space arguments
        X = Y = 5
        nx = ny = 10
        halfx = int(nx/2)
        halfy = int(ny/2)
        dx = X/(nx-1)
        dy = Y/(ny-1)
        xVec = np.linspace(0,X,nx)
        yVec = np.linspace(0,Y,ny)
        ops = 2
        tso = 2
        #Time arguments
        t0 = 0
        tf = 10
        dt = 0.01
        times = np.arange(t0,tf+dt,dt)
        nt = len(times)
        #properties
        gamma = 1.4
        leftBC = (1,0,0,2.5)
        lbc1 = np.asarray(leftBC[:2]+(leftBC[-1],))
        rightBC = (0.125,0,0,.25)
        rbc1 = np.asarray(rightBC[:2]+(rightBC[-1],))
        #globals for each funciton
        set_globals(False,None,tf,t0,dt,dx,dy,gamma)
        e1d.set_globals(gamma,dt,dx)
        #------------------------------TESTING X DIRECTION----------------------#
        #Make analytical solution
        shock2D = np.zeros((2*nt,4,nx+2*ops,ny+2*ops))
        shock1D = np.zeros((2*nt,3,nx+2*ops))

        for x in range(halfx):
            shock1D[0,:,x+ops] = lbc1
            shock1D[0,:,x+halfx+ops] = rbc1
        bcpy1D(0,shock1D,ops)

        for y in range(ny+2*ops):
            shock2D[0,0,:,y]=shock1D[0,0,:]
            shock2D[0,1,:,y]=shock1D[0,1,:]
            shock2D[0,2,:,y]=shock1D[0,1,:]
            shock2D[0,3,:,y]=shock1D[0,2,:]
        #Creating array
        iidx = [(x+ops,y+ops)for x,y in np.ndindex(shock2D.shape[2]-2*ops,shock2D.shape[3]-2*ops)]
        #First step
        ps = "%0.3f"
        shock1D[1,:,ops:-ops] = e1d.step(shock1D[0],shock1D[0],0)[:,ops:-ops]
        bcpy1D(1,shock1D,ops)
        step(shock2D,iidx,0,0)
        bcpy(1,shock2D,ops)
        # Iteration
        err = []
        for i in range(1,2*nt-1):
            #1D
            id1 = i-1 if (i+1)%2==0 else i
            shock1D[i+1,:,ops:-ops] = e1d.step(shock1D[id1],shock1D[i],i)[:,ops:-ops]
            bcpy1D(i+1,shock1D,ops)
            #2D
            step(shock2D,iidx,i,i)
            bcpy(i+1,shock2D,ops)
            err.append(shock2D[i,(0,1,3),:,halfy]-shock1D[i,:,:])
            assert np.allclose(shock2D[i,(0,1,3),:,halfy],shock1D[i,:,:])
        # pm(shock2D,0,0,"%0.3f")
        print("CPU Max X Error: ",np.amax(err))

        #------------------------------TESTING Y DIRECTION----------------------#
        #Make analytical solution
        shock2D = np.zeros((2*nt,4,nx+2*ops,ny+2*ops))
        shock1D = np.zeros((2*nt,3,nx+2*ops))

        for x in range(halfx):
            shock1D[0,:,x+ops] = lbc1
            shock1D[0,:,x+halfx+ops] = rbc1
        bcpy1D(0,shock1D,ops)

        for x in range(nx+2*ops):
            shock2D[0,0,x,:]=shock1D[0,0,:]
            shock2D[0,1,x,:]=shock1D[0,1,:]
            shock2D[0,2,x,:]=shock1D[0,1,:]
            shock2D[0,3,x,:]=shock1D[0,2,:]
        #Creating array
        iidx = [(x+ops,y+ops)for x,y in np.ndindex(shock2D.shape[2]-2*ops,shock2D.shape[3]-2*ops)]
        #First step
        ps = "%0.3f"
        shock1D[1,:,ops:-ops] = e1d.step(shock1D[0],shock1D[0],0)[:,ops:-ops]
        bcpy1D(1,shock1D,ops)
        step(shock2D,iidx,0,0)
        bcpy(1,shock2D,ops)
        # Iteration
        err = []
        for i in range(1,2*nt-1):
            #1D
            id1 = i-1 if (i+1)%2==0 else i
            shock1D[i+1,:,ops:-ops] = e1d.step(shock1D[id1],shock1D[i],i)[:,ops:-ops]
            bcpy1D(i+1,shock1D,ops)
            #2D
            step(shock2D,iidx,i,i)
            bcpy(i+1,shock2D,ops)
            err.append(shock2D[i,(0,2,3),halfx,:]-shock1D[i,:,:])
            assert np.allclose(shock2D[i,(0,2,3),halfx,:],shock1D[i,:,:])
        # pm(shock2D,0,0,"%0.3f")
        print("CPU Max Y Error: ",np.amax(err))

    def test_euler_shock_gpu_X():
            #Space arguments
            X = Y = 5
            nx = ny = 10
            BS = 10
            halfx = int(nx/2)
            halfy = int(ny/2)
            dx = X/(nx-1)
            dy = Y/(ny-1)
            xVec = np.linspace(0,X,nx)
            yVec = np.linspace(0,Y,ny)
            ops = 2
            tso = 2
            #Time arguments
            t0 = 0
            tf = 10
            dt = 0.1
            times = np.arange(t0,tf+dt,dt)
            nt = len(times)
            #properties
            gamma = 1.4
            leftBC = (1,0,0,2.5)
            lbc1 = np.asarray(leftBC[:2]+(leftBC[-1],))
            rightBC = (0.125,0,0,.25)
            rbc1 = np.asarray(rightBC[:2]+(rightBC[-1],))
            #globals for each funciton
            set_globals(False,None,tf,t0,dt,dx,dy,gamma)
            e1d.set_globals(gamma,dt,dx)
            #------------------------------TESTING X DIRECTION----------------------#
            #Make analytical solution
            shock2D = np.zeros((2*nt,4,nx+2*ops,ny+2*ops))
            shock1D = np.zeros((2*nt,3,nx+2*ops))

            for x in range(halfx):
                shock1D[0,:,x+ops] = lbc1
                shock1D[0,:,x+halfx+ops] = rbc1
            bcpy1D(0,shock1D,ops)

            for y in range(ny+2*ops):
                shock2D[0,0,:,y]=shock1D[0,0,:]
                shock2D[0,1,:,y]=shock1D[0,1,:]
                shock2D[0,2,:,y]=shock1D[0,1,:]
                shock2D[0,3,:,y]=shock1D[0,2,:]
            #Creating array
            iidx = [(x+ops,y+ops)for x,y in np.ndindex(shock2D.shape[2]-2*ops,shock2D.shape[3]-2*ops)]
            #-----------------------------GPU STUFF----------------------------#
            block_shape = (BS,BS,1)
            GRD = (int(nx/BS),int(ny/BS))   #Grid size
            #Creating constants
            NV = shock2D.shape[1]
            SGIDS = (nx+2*ops)*(ny+2*ops)
            STS = SGIDS*NV #Shared time shift
            VARS =  (nx*+2*ops)*(ny*+2*ops)
            TIMES = VARS*NV
            MPSS = 2
            MOSS = 4
            OPS = ops
            TSO = 2
            const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})

            #Source Modules
            SM = source.build_gpu_source(os.path.join(path,'euler.h'))
            source.swept_constant_copy(SM,const_dict)
            arr_gpu = cuda.mem_alloc(shock2D.nbytes)
            set_globals(True,SM,tf,t0,dt,dx,dy,gamma)
            #First step
            ps = "%0.3f"
            shock1D[1,:,ops:-ops] = e1d.step(shock1D[0],shock1D[0],0)[:,ops:-ops]
            bcpy1D(1,shock1D,ops)
            #GPU 1st step
            cuda.memcpy_htod(arr_gpu,shock2D)
            SM.get_function('test_step')(arr_gpu,np.int32(0),np.int32(0),grid=GRD,block=(BS,BS,1))
            cuda.Context.synchronize()
            cuda.memcpy_dtoh(shock2D,arr_gpu)
            bcpy(1,shock2D,ops)
            #Iteration
            err=[shock2D[1,(0,1,3),:,halfy]-shock1D[1,:,:],]
            for i in range(1,2*nt-1):
                #1D
                id1 = i-1 if (i+1)%2==0 else i
                shock1D[i+1,:,ops:-ops] = e1d.step(shock1D[id1],shock1D[i],i)[:,ops:-ops]
                bcpy1D(i+1,shock1D,ops)
                #2D GPU
                cuda.memcpy_htod(arr_gpu,shock2D)
                SM.get_function('test_step')(arr_gpu,np.int32(0),np.int32(i),grid=GRD,block=(BS,BS,1))
                cuda.Context.synchronize()
                cuda.memcpy_dtoh(shock2D,arr_gpu)
                bcpy(i+1,shock2D,ops)
                assert np.allclose(shock2D[i,(0,1,3),:,halfy],shock1D[i,:,:])
                err.append(shock2D[i+1,(0,1,3),:,halfy]-shock1D[i+1,:,:])
            print("GPU Max X Error: ",np.amax(err))

    def test_euler_shock_gpu_Y():
            #Space arguments
            X = Y = 5
            nx = ny = 10
            BS = 10
            halfx = int(nx/2)
            halfy = int(ny/2)
            dx = X/(nx-1)
            dy = Y/(ny-1)
            xVec = np.linspace(0,X,nx)
            yVec = np.linspace(0,Y,ny)
            ops = 2
            tso = 2
            #Time arguments
            t0 = 0
            tf = 10
            dt = 0.01
            times = np.arange(t0,tf+dt,dt)
            nt = len(times)
            #properties
            gamma = 1.4
            leftBC = (1,0,0,2.5)
            lbc1 = np.asarray(leftBC[:2]+(leftBC[-1],))
            rightBC = (0.125,0,0,.25)
            rbc1 = np.asarray(rightBC[:2]+(rightBC[-1],))
            #globals for each funciton
            set_globals(False,None,tf,t0,dt,dx,dy,gamma)
            e1d.set_globals(gamma,dt,dx)
            #------------------------------TESTING X DIRECTION----------------------#
            #Make analytical solution
            shock2D = np.zeros((2*nt,4,nx+2*ops,ny+2*ops))
            shock1D = np.zeros((2*nt,3,nx+2*ops))

            for x in range(halfx):
                shock1D[0,:,x+ops] = lbc1
                shock1D[0,:,x+halfx+ops] = rbc1
            bcpy1D(0,shock1D,ops)

            for x in range(nx+2*ops):
                shock2D[0,0,x,:]=shock1D[0,0,:]
                shock2D[0,1,x,:]=shock1D[0,1,:]
                shock2D[0,2,x,:]=shock1D[0,1,:]
                shock2D[0,3,x,:]=shock1D[0,2,:]
            #Creating array
            iidx = [(x+ops,y+ops)for x,y in np.ndindex(shock2D.shape[2]-2*ops,shock2D.shape[3]-2*ops)]
            #-----------------------------GPU STUFF----------------------------#
            block_shape = (BS,BS,1)
            GRD = (int(nx/BS),int(ny/BS))   #Grid size
            #Creating constants
            NV = shock2D.shape[1]
            SGIDS = (nx+2*ops)*(ny+2*ops)
            STS = SGIDS*NV #Shared time shift
            VARS =  (nx*+2*ops)*(ny*+2*ops)
            TIMES = VARS*NV
            MPSS = 2
            MOSS = 4
            OPS = ops
            TSO = 2
            const_dict = ({"NV":NV,"SGIDS":SGIDS,"VARS":VARS,"TIMES":TIMES,"MPSS":MPSS,"MOSS":MOSS,"OPS":OPS,"TSO":TSO,"STS":STS})

            #Source Modules
            SM = source.build_gpu_source(os.path.join(path,'euler.h'))
            source.swept_constant_copy(SM,const_dict)
            arr_gpu = cuda.mem_alloc(shock2D.nbytes)
            set_globals(True,SM,tf,t0,dt,dx,dy,gamma)
            #First step
            ps = "%0.3f"
            shock1D[1,:,ops:-ops] = e1d.step(shock1D[0],shock1D[0],0)[:,ops:-ops]
            bcpy1D(1,shock1D,ops)
            #GPU 1st step
            cuda.memcpy_htod(arr_gpu,shock2D)
            SM.get_function('test_step')(arr_gpu,np.int32(0),np.int32(0),grid=GRD,block=(BS,BS,1))
            cuda.Context.synchronize()
            cuda.memcpy_dtoh(shock2D,arr_gpu)
            bcpy(1,shock2D,ops)
            #Iteration
            err=[shock2D[1,(0,2,3),halfx,:]-shock1D[1,:,:],]
            for i in range(1,2*nt-1):
                #1D
                id1 = i-1 if (i+1)%2==0 else i
                shock1D[i+1,:,ops:-ops] = e1d.step(shock1D[id1],shock1D[i],i)[:,ops:-ops]
                bcpy1D(i+1,shock1D,ops)
                #2D GPU
                cuda.memcpy_htod(arr_gpu,shock2D)
                SM.get_function('test_step')(arr_gpu,np.int32(0),np.int32(i),grid=GRD,block=(BS,BS,1))
                cuda.Context.synchronize()
                cuda.memcpy_dtoh(shock2D,arr_gpu)
                bcpy(i+1,shock2D,ops)
                assert np.allclose(shock2D[i,(0,2,3),halfx,:],shock1D[i,:,:])
                err.append(shock2D[i+1,(0,2,3),halfx,:]-shock1D[i+1,:,:])
            print("GPU Max Y Error: ",np.amax(err))

    test_euler_shock()
    test_euler_shock_gpu_X()
    test_euler_shock_gpu_Y()
