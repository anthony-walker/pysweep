#Programmer: Anthony Walker
#This file contains functions to solve the eulers equations in 2 dimensions with
#the swept rule or in a standard way


import numpy as np
import sympy,sys,itertools
#Testing imports
if __name__ == "__main__":
    import test_euler as e1d
    import warnings, os
    fp = os.path.abspath(__file__)
    path = os.path.dirname(fp)
    sys.path.insert(0, path[:-len('equations')])
    import distributed.sweep.ccore.source as source
    # warnings.filterwarnings('error')

#----------------------------------Globals-------------------------------------#
dx = dy = dt = conductivity = 0
rho = cp = 0
eqns = None
srcs = None
#----------------------------------End Globals-------------------------------------#

def step(state,iidx,ts,globalTimeStep):
    """This is the method that will be called by the swept solver.
    state - 4D numpy array(t,v,x,y (v is variables length))
    iidx -  an iterable of indexs
    ts - the current time step
    globalTimeStep - a step counter that allows implementation of the scheme
    """
    statements = {0:statement0,1:statement1}
    # return statements[globalTimeStep%2](state,iidx,ts)
    return statementf(state,iidx,ts)

def statementf(state,iidx,ts):
    """Use this funciton to complete the first step."""
    vs = slice(0,state.shape[1],1)
    ops=2
    for idx,idy in iidx:
        # print(source((idx-ops)*dx,(idy-ops)*dy,(ts)*dt-dt/2))
        # print(derivatives(state[ts],idx,idy))
        # input()
        state[ts+1,:,idx,idy] = state[ts,:,idx,idy]+dt*(derivatives(state[ts],idx,idy)+source((idx-ops)*dx,(idy-ops)*dy,(ts)*dt))
    return state

def statement0(state,iidx,ts):
    """Use this funciton to complete the first step."""
    vs = slice(0,state.shape[1],1)
    ops=2
    for idx,idy in iidx:
        # print(source((idx-ops)*dx,(idy-ops)*dy,(ts)*dt-dt/2))
        # print(derivatives(state[ts],idx,idy))
        # input()
        state[ts+1,:,idx,idy] = state[ts,:,idx,idy]+0.5*dt*(derivatives(state[ts],idx,idy)+source((idx-ops)*dx,(idy-ops)*dy,(ts)*dt/2))
    return state

def statement1(state,iidx,ts):
    """Use this function to complete the second step of the splitting"""
    vs = slice(0,state.shape[1],1)
    ops=2
    for idx,idy in iidx:
        state[ts+1,:,idx,idy] = state[ts-1,:,idx,idy]+dt*(derivatives(state[ts],idx,idy)+source((idx-ops)*dx,(idy-ops)*dy,(ts)*dt/2))
    return state


def derivatives(state,idx,idy):
    """Use this function to determine the first derivatives in space"""
    dervs = np.zeros(state[:,idx,idy].shape)
    #Second derivatives
    #Equation 1
    dervs[0] += conductivity/rho/cp*(state[0,idx+1,idy]-2*state[0,idx,idy]+state[0,idx-1,idy])/(dx*dx)
    #Equation 2
    dervs[0] += conductivity/rho/cp*(state[0,idx,idy+1]-2*state[0,idx,idy]+state[0,idx,idy-1])/(dx*dx)
    return dervs

def set_globals(gpu,*args,source_mod=None):
    """Use this function to set cpu global variables"""
    global dt,dx,dy,conductivity
    t0,tf,dt,dx,dy,conductivity,rho,cp = args
    if gpu:
        keys = "DT","DX","DY"
        nargs = args[2:]+(gam-1,dt/dx,dt/dy)
        fc = lambda x:np.float64(x)
        for i,key in enumerate(keys):
            ckey,_ = source_mod.get_global(key)
            cuda.memcpy_htod(ckey,fc(nargs[i]))
    else:
        solutions()
        x,y,t = sympy.symbols('x y t')
        global srcs
        srcs = tuple()
        srcs += sympy.lambdify((x,y,t),rho*cp*sympy.diff(eqns[0],t)-conductivity*sympy.diff(eqns[0],x,x)-conductivity*sympy.diff(eqns[0],y,y)),

def source(x,y,t):
    """This function finds the source term"""
    return np.asarray([sc(x,y,t)/rho/cp for sc in srcs])

def solutions():
    """Use this function to create the solutions"""
    global eqns
    eqns = tuple()
    x,y,t = sympy.symbols('x y t')
    TPI =2*np.pi
    eqns+=(100*sympy.sin(TPI*x)*sympy.sin(TPI*y/2)*sympy.exp(-t),)
    # eqns+=(2*sympy.sin(TPI*x)*sympy.exp(t),)
    # eqns+=(4*t*sympy.sin(TPI*y),)
    # eqns+=(2*t*sympy.sin(TPI*y),)

if __name__ == "__main__":
    #Debugging functions
    #Boundary copy
    #Space arguments
    def bcpy(idx,state,ops):
        state[idx,:,:ops,:] = state[idx,:,-2*ops-1:-ops-1,:]
        state[idx,:,-ops:,:] = state[idx,:,1+ops:2*ops+1,:]
        state[idx,:,:,:ops] = state[idx,:,:,-2*ops-1:-ops-1]
        state[idx,:,:,-ops:] = state[idx,:,:,1+ops:1+2*ops]

    X = Y = 1
    nx = ny = 11
    dx = X/(nx-1)
    dy = Y/(ny-1)
    xVec = np.linspace(0,X,nx)
    yVec = np.linspace(0,Y,ny)
    ops = 2
    tso = 2
    #Time arguments
    t0 = 0
    tf = 0.1
    dt = tf/1000
    times = np.arange(t0,tf+dt,dt)
    nt = len(times)
    rho = 8933
    cp = 385
    cond = 401
    set_globals(False,t0,tf,dt,dx,dy,cond,rho,cp)
    arr2D = np.zeros((nt,1,nx+2*ops,ny+2*ops))
    iidx = [(x+ops,y+ops)for x,y in np.ndindex(arr2D.shape[2]-2*ops,arr2D.shape[3]-2*ops)]
    #Create initial conditions
    x,y,t = sympy.symbols('x y t')
    Teqn = sympy.lambdify((x,y,t),eqns[0])
    for idx,idy in iidx:
        cx = (idx-ops)*dx
        cy = (idy-ops)*dy
        arr2D[0,0,idx,idy] = Teqn(cx,cy,t0)
    #Analytical solution
    analyt2D = np.zeros((nt,1,nx+2*ops,ny+2*ops))
    for i,t in enumerate(times):
        for idx,idy in iidx:
            cx = (idx-ops)*dx
            cy = (idy-ops)*dy
            analyt2D[i,0,idx,idy] = Teqn(cx,cy,t)
        bcpy(i,analyt2D,ops)
        # print('-------------------------'+str(i+1)+'--------------------------')
        # for row in arr2D[i+1,0,:,:]-analyt2D[i,0,:,:]:
        #     sys.stdout.write('[')
        #     for col in row:
        #         sys.stdout.write("%0.3f"%col+", ")
        #     sys.stdout.write("]\n")
        # input()

    # Solving
    ct = 0
    for i in range(nt-1):
        bcpy(i,arr2D,ops)
        step(arr2D,iidx,i,i)
    print(np.amax(arr2D[:,0,ops:-ops,ops:-ops]-analyt2D[:,0,ops:-ops,ops:-ops]))
