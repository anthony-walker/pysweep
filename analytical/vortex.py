#Programmer: Anthony Walker
#This file contains the analytical solution to the euler vortex
#problem in two dimensions.

#Import statements
import numpy as np
import os
import h5py
#Plottign
# import matplotlib as mpl
# mpl.use("Tkagg")
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from collections.abc import Iterable
# import matplotlib.animation as animation

#Indicies
pid = 0
did = 1
uid = 2
vid = 3
epsilon = 2e-20

def vortex(cvics,npx,npy,times=(0,),x0=0,y0=0):
    """This is the primary method to solve the euler vortex that is centered at the origin with periodic boundary conditions
    properties are obtained from the vics object, cvics.
       The center can be changed with x0 and y0.
       the time step or steps can be changed with times but it must be a tuple.
       X & Y are dimensions of the domain.
       npx, and npy are the number of points the in respective direction

       This solution was obtained from

    """

    # Persson, P. O., Bonet, J., & Peraire, J. (2009).
    # Discontinuous Galerkin solution of the Navier–Stokes equations on deformable domains.
    # Computer Methods in Applied Mechanics and Engineering, 198(17-20), 1585-1595.

    alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, c, sigma, beta, r_c, L = cvics.get_args()
    PI = np.pi
    assert epsilon >= L/r_c*np.exp(-L*L/(2*r_c*r_c*sigma*sigma))
    X = L
    Y = L
    #Universal gas Constant
    R_univ = 8.314 #J/molK
    if rho_inf is None:
        rho_inf = gamma*P_inf/c**2
    #Getting Freestream velocity
    V_inf = M_inf*c
    #Getting velocity components
    u_bar = V_inf*np.cos(alpha)
    v_bar = V_inf*np.sin(alpha)
    #Creating grid
    xpts = np.linspace(-X,X,npx,dtype=np.float64)
    ypts = np.linspace(-Y,Y,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')
    state = np.zeros(np.shape(times)+(4,)+xgrid.shape,dtype=np.float64) #Initialization of state
    idxs = tuple(np.ndindex(xgrid.shape))  #Indicies of the array
    #Solving times
    for i in range(len(times)):
        for idx in idxs:
            #getting x and y location
            x = xgrid[idx]
            y = ygrid[idx]

            #differences from origin
            dx0 = (x-x0)
            dy0 = (y-y0)
            #Common terms
            uterm = (dx0-u_bar*times[i])
            vterm = (dy0-v_bar*times[i])
            pterm = beta*beta*(gamma-1)*M_inf*M_inf/(8*PI*PI)

            #function calculation f(x,y,t)
            fx = uterm*uterm
            fy = vterm*vterm
            f = (1-fx-fy)/(r_c*r_c)

            #Finding state variables
            state[(i,pid)+idx] = P_inf*(1-pterm*np.exp(f))**(gamma/(gamma-1)) #pressure
            state[(i,did)+idx] = rho_inf*(1-pterm*np.exp(f))**(1/(gamma-1)) #density
            state[(i,uid)+idx] = V_inf*(np.cos(alpha)-beta*vterm/(2*PI*r_c)*np.exp(f/2)) #x velocity
            state[(i,vid)+idx] = V_inf*(np.sin(alpha)-beta*uterm/(2*PI*r_c)*np.exp(f/2)) #y velocity
    return state


def create_vortex_data(cvics,npx,npy, times=(0,), x0=0, y0=0, filepath = "./vortex/",filename = "vortex",fdb=True):
    """Use this function to create vortex data from the vortex funciton.
    Note, this function will create a directory called "vortex#" unless otherwise specified.
    An hdf5 file will be created with the groups pressure, density, x-velocity, and y-velocity.
    files in these directories will be labeled in time order e.g. density1.txt, density2.txt etc.
    the time will be stored in each file as well.
    """
    #Args tuple
    state = vortex(cvics,npx,npy, times, x0=x0, y0=y0)
    #Making directories
    if filepath == "./vortex/":
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
    #Making new file
    ctr=0
    hdf_end = ".hdf5"
    tpath = filepath+filename+str(ctr)
    while os.path.isfile(tpath+hdf_end):
        tpath=tpath[:-len(str(ctr))]
        tpath+=str(ctr+1)
        ctr+=1
    file = h5py.File(tpath+".hdf5","w")
    if not fdb:
        pressure = file.create_dataset("pressure",(len(times),1,npx,npy))
        density = file.create_dataset("density",(len(times),1,npx,npy))
        x_velocity = file.create_dataset("x-velocity",(len(times),1,npx,npy))
        y_velocity = file.create_dataset("y-velocity",(len(times),1,npx,npy))
    else:
        data = file.create_dataset("data",(len(times),4,npx,npy))

    cvic_args = [item  if item is not None else 1e-32 for item in cvics.get_args()]
    ic_data = file.create_dataset("ic",np.shape(cvic_args))
    ic_data[:]=cvic_args[:]
    origin_data = file.create_dataset("origin",(2,))
    origin_data[:] = (x0,y0)
    dim_data = file.create_dataset("dimensions",(4,))
    dim_data[:] = (cvics.L,cvics.L,npx,npy)
    #Creating state data

    if not fdb:
        pressure[:,0,:,:] = state[:,pid,:,:]
        density[:,0,:,:] = state[:,did,:,:]
        x_velocity[:,0,:,:] =state[:,uid,:,:]
        y_velocity[:,0,:,:] =state[:,vid,:,:]
    else:
        flux = convert_to_flux(state,cvics.gamma)
        data[:,pid,:,:] = flux[:,pid,:,:]
        data[:,did,:,:] = flux[:,did,:,:]
        data[:,uid,:,:] =flux[:,uid,:,:]
        data[:,vid,:,:] =flux[:,vid,:,:]
    return state

def convert_to_flux(vortex_data,gamma):
    """Use this function to convert a vortex to flux data."""
    flux = np.zeros(vortex_data.shape)
    P = vortex_data[:,pid,:,:]
    rho = vortex_data[:,did,:,:]
    u = vortex_data[:,uid,:,:]
    v = vortex_data[:,vid,:,:]
    flux[:,0,:,:] = rho
    flux[:,1,:,:] = rho*u
    flux[:,2,:,:] = rho*v
    rhoe = P/(gamma-1)+rho*u*u/2+rho*v*v/2
    flux[:,3,:,:] = rhoe
    return flux

def vortex_plot(filename,property,time,xs=None,ys=None,levels=10,savepath = "./vortex_plot"):
    """Use this function to plot a property with a given time."""
    file = h5py.File(filename,"r")
    data = file[property]
    #Dimensions
    dims = file['dimensions']
    X = dims[0]
    Y = dims[1]
    npx = dims[2]
    npy = dims[3]
    #Meshgrid
    xpts = np.linspace(-X,X,npx,dtype=np.float64)
    ypts = np.linspace(-Y,Y,npy,dtype=np.float64)
    xgrid,ygrid = np.meshgrid(xpts,ypts,sparse=False,indexing='ij')

    fig, ax =plt.subplots()
    ax.set_ylim(-Y, Y)
    ax.set_xlim(-X, X)
    ax.set_title(property)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    # pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')
    fig.colorbar(cm.ScalarMappable(cmap=cm.inferno),ax=ax,boundaries=np.linspace(-1,1,10))
    animate = lambda i: ax.contourf(xgrid,ygrid,data[i,:,:,:][0],levels=levels,cmap=cm.inferno)

    if isinstance(time,Iterable):
        frames = len(tuple(time))
        anim = animation.FuncAnimation(fig,animate,frames)
        anim.save(savepath+".gif",writer="imagemagick")
    else:
        animate(time)
        fig.savefig(savepath+".png")
        plt.show()



class vics(object):
    """This class contains functions for vortex initial conditions
    All information contained in this class was found via:

    Spiegel, S. C., Huynh, H. T., & DeBonis, J. R. (2015).
    A survey of the isentropic euler vortex problem using high-order methods.
    In 22nd AIAA Computational Fluid Dynamics Conference (p. 2444).

    Call set or an Initializer function to create the state (a meshgrid of data) and arguments
    the generate function is called by default but can be redefined by calling generate with the desired arguments
    """
    def __init__(self):
        """Creates empty vics object."""
        self.initialized = False

    #----------------------------vics Initializer functions

    def Shu(self,gamma,npts = None):
        """ Initializer function that uses initial conditions from

        Shu, C.-W., “Essentially Non-oscillatory and Weighted Essentially Non-oscillatory Schemes for Hyperbolic Conservation
        Laws,” Advanced Numerical Approximation of Nonlinear Hyperbolic Equations , edited by A. Quarteroni, Vol. 1697 of Lecture
        Notes in Mathematics, Springer Berlin Heidelberg, 1998, pp. 325–432.

        User must specify: gamma
        """
        #Dimensions
        self.L = 10   #Half of the computational domain length and height
        self.r_c = 1  #Vortex Radius
        #Gas properties
        self.gamma = gamma
        self.R_gas = None  #J/kgK
        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = np.pi/4  #Angle of attack
        self.M_inf = np.sqrt(2/self.gamma)    #Mach number
        self.P_inf = 1  #pressure kPa
        self.T_inf = 1 #Temperature K
        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = self.M_inf*5*np.sqrt(2)/(4*np.pi)*np.exp(0.5) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c if npts is None else npts
        self.state = vortex(self,npts,npts)
        self.flux = convert_to_flux(self.state,self.gamma)
        return self

    def Vincent(self,gamma,npts = None):
        """ Initializer function that uses initial conditions from

        Vincent, P., Castonguay, P., and Jameson, A., “Insights from von Neumann Analysis of High-Order Flux Reconstruction
        Schemes,” Journal of Computational Physics, Vol. 230, No. 22, 2011, pp. 8134–8154.

        User must specify: gamma
        """
        #Dimensions
        self.L = 20   #Half of the computational domain length and height
        self.r_c = 3/2  #Vortex Radius

        #Gas properties
        self.gamma = gamma
        self.R_gas = None  #J/kgK

        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = np.pi/2  #Angle of attack
        self.M_inf = 0.4    #Mach number
        self.P_inf = 1/(self.gamma*self.M_inf*self.M_inf)  #Pressure kPa
        self.T_inf = 1 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = self.M_inf*27/(4*np.pi)*np.exp(2/9) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c if npts is None else npts
        self.state = vortex(self,npts,npts)
        self.flux = convert_to_flux(self.state,self.gamma)
        return self

    def HestWar(self,gamma,npts = None):
        """ Initializer function that uses initial conditions from

        Hesthaven, J. S. and Warburton, T., Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications ,
        Vol. 54 of Texts in Applied Mathematics, Springer New York, 2008.

        User must specify: gamma
        """
        #Dimensions
        self.L = 5  #Half of the computational domain length and height
        self.r_c = np.sqrt(0.5)  #Vortex Radius

        #Gas properties
        self.gamma = gamma
        self.R_gas = None  #J/kgK

        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = 0  #Angle of attack
        self.M_inf = np.sqrt(1/self.gamma)    #Mach number
        self.P_inf = 1  #Pressure kPa
        self.T_inf = 1 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = self.M_inf*5/(2*np.pi)*np.exp(1) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c if npts is None else npts
        self.state = vortex(self,npts,npts)
        self.flux = convert_to_flux(self.state,self.gamma)
        return self

    def HOWF(self,gamma,npts = None):
        """ Initializer function that uses initial conditions from (FAST)

            Wang, Z., Fidkowski, K., Abgrall, R., Bassi, F., Caraeni, D., Cary, A., Deconinck, H., Hartmann, R., Hillewaert, K.,
            Huynh, H., Kroll, N., May, G., Persson, P.-O., van Leer, B., and Visbal, M., “High-order CFD Methods: Current Status and
            Perspective,” International Journal for Numerical Methods in Fluids, Vol. 72, No. 8, 2013, pp. 811–845

            User must specify: gamma
        """
        #Dimensions
        self.L = 0.05   #Half of the computational domain length and height
        self.r_c = 0.005  #Vortex Radius

        #Gas properties
        self.gamma = 1.4
        self.R_gas = 287.15  #J/kgK

        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = 0  #Angle of attack
        self.M_inf = 0.5    #Mach number
        self.P_inf = 100  #Pressure kPa
        self.T_inf = 300 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = 1/5 #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c if npts is None else npts
        self.state = vortex(self,npts,npts)
        self.flux = convert_to_flux(self.state,self.gamma)
        return self

    def HOWS(self,gamma,npts = None):
        """ Initializer function that uses initial conditions from (SLOW)

            Wang, Z., Fidkowski, K., Abgrall, R., Bassi, F., Caraeni, D., Cary, A., Deconinck, H., Hartmann, R., Hillewaert, K.,
            Huynh, H., Kroll, N., May, G., Persson, P.-O., van Leer, B., and Visbal, M., “High-order CFD Methods: Current Status and
            Perspective,” International Journal for Numerical Methods in Fluids, Vol. 72, No. 8, 2013, pp. 811–845

            User must specify: gamma
        """
        #Dimensions
        self.L = 0.05   #Half of the computational domain length and height
        self.r_c = 0.005  #Vortex Radius

        #Gas properties
        self.gamma = 1.4
        self.R_gas = 287.15  #J/kgK

        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = 0  #Angle of attack
        self.M_inf = 0.05    #Mach number
        self.P_inf = 100  #Pressure kPa
        self.T_inf = 300 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = 1/50 #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c if npts is None else npts
        self.state = vortex(self,npts,npts)
        self.flux = convert_to_flux(self.state,self.gamma)
        return self

    def Spiegel(self,gamma,npts = None):
        """ Initializer function that uses initial conditions from

            Spiegel, S. C., Huynh, H. T., & DeBonis, J. R. (2015).
            A survey of the isentropic euler vortex problem using high-order methods.
            In 22nd AIAA Computational Fluid Dynamics Conference (p. 2444).

            User must specify: gamma
        """
        #Dimensions
        self.L = 10   #Half of the computational domain length and height
        self.r_c = 1  #Vortex Radius

        #Gas properties
        self.gamma = gamma
        self.R_gas = None  #J/kgK

        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = 0  #Angle of attack
        self.M_inf = 0    #Mach number
        self.P_inf = 1  #Pressure kPa
        self.T_inf = 1 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = 5/(2*np.pi*np.sqrt(self.gamma))*np.exp(0.5) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c if npts is None else npts
        self.state = vortex(self,npts,npts)
        self.flux = convert_to_flux(self.state,self.gamma)
        return self

    def set(args,npts = None):
        """Use this function to set an alternate condition.
        args = (alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, sigma, beta, R, L)
        Values not used can be set to None, note speed of sound is computed with rho_inf or R_gas.
        So one is required
        #Dimensions
        self.L = L   #Half of the computational domain length and height
        self.r_c = r_c  #Vortex Radius

        #Gas properties
        self.gamma = gamma
        self.R_gas = R_gas  #J/kgK

        #Freestream variables
        self.rho_inf = rho_inf    #Denisty
        self.alpha = alpha  #Angle of attack
        self.M_inf = M_inf    #Mach number
        self.P_inf = P_inf  #Pressure kPa
        self.T_inf = T_inf #Temperature K

        #Gaussian Variables
        self.sigma = sigma  #Standard deviation
        self.beta = beta #Maximum perturbation strength
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.sigma, self.beta, self.r_c, self.L)
        """
        (alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, sigma, beta, r_c, L) = args
        #Dimensions
        self.L = L   #Half of the computational domain length and height
        self.r_c = r_c  #Vortex Radius

        #Gas properties
        self.gamma = gamma
        self.R_gas = R_gas  #J/kgK

        #Freestream variables
        self.rho_inf = rho_inf    #Denisty
        self.alpha = alpha  #Angle of attack
        self.M_inf = M_inf    #Mach number
        self.P_inf = P_inf  #Pressure kPa
        self.T_inf = T_inf #Temperature K

        #Gaussian Variables
        self.sigma = sigma  #Standard deviation
        self.beta = beta #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c if npts is None else npts
        self.state = vortex(self,npts,npts)
        self.flux = convert_to_flux(self.state,self.gamma)
        return self



    #-----------------------Functions used by Initializers-----------------------

    def generate(self,X,Y,npx,npy,t=0,x0=0,y0=0):
        """
        This is a method found in

        Spiegel, S. C., Huynh, H. T., & DeBonis, J. R. (2015).
        A survey of the isentropic euler vortex problem using high-order methods.
        In 22nd AIAA Computational Fluid Dynamics Conference (p. 2444).

        to impose a perturbation and generate initial conditions

        """
        alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, c, sigma, beta, r_c, L = self.get_args() #unpacking args
        x = np.linspace(-X,X,npx,dtype=np.float64)
        y = np.linspace(-Y,Y,npy,dtype=np.float64)
        xgrid,ygrid = np.meshgrid(x,y,sparse=False,indexing='ij')
        f = np.zeros(xgrid.shape,dtype=np.float64)
        f = -1/(2*sigma**2)*((xgrid/r_c)**2+(ygrid/r_c)**2)
        #Gaussian function
        Omega = beta*np.exp(f)
        #Perturbations
        du = -ygrid/r_c*Omega
        dv = xgrid/r_c*Omega
        dT = -(gamma-1)/2*Omega**2
        #Initial conditions
        state = np.zeros(xgrid.shape+(4,))
        #THESE ARE NONDIMENSIONAL - NEED INTERNET
        state[:,:,0] = (1+dT)**(1/(gamma-1))    #Initial density
        state[:,:,1] = M_inf*np.cos(alpha)+du   #Initial u
        state[:,:,2] = M_inf*np.cos(alpha)+du   #Initial v
        state[:,:,3] = 1/gamma*(1+dT)**(gamma/(1-gamma))   #Initial pressure
        self.state = state

    def speed_of_sound(self):
        """Use this funciton to calculate the speed of sound."""
        try:
            if self.rho_inf is not None:
                self.c = np.sqrt(self.gamma*self.P_inf/self.rho_inf)
            elif self.R_gas is not None:
                self.MolarMass = self.R_univ/self.R_gas
                self.c = np.sqrt(self.gamma*self.R_univ*self.T_inf/self.MolarMass)
            else:
                warn = """
                Warning: speed of sound could not be calculated with given variables. See below:

                if rho_inf is not None:
                    c = np.sqrt(gamma*P_inf/rho_inf)
                elif R_gas is not None:
                    MolarMass = R_univ/R_gas
                    c = np.sqrt(gamma*R_univ*T_inf/MolarMass)
                """
                raise Exception(warn)
        except Exception as e:
            print(e)

    #------------------------------Functions to get vics data---------------------
    def get_args(self):
        """Use this function to get arguments."""
        try:
            if not hasattr(self, 'args'):
                warn = """ERROR: vics object does not have args, returning None. Please initialize
                with properties before getting args e.g. myVics.Shu() or myVics.set(myProps) for custom properties.
                """
                raise Exception(warn)
            else:
                return self.args
        except Exception as e:
            print(e)

    def get_state(self):
        """Use this function to get arguments."""
        try:
            if not hasattr(self, 'state'):
                warn = """ERROR: vics object does not have state, returning None. Please initialize
                with properties before getting args e.g. myVics.Shu() or myVics.set(myProps) for custom properties.
                """
                raise Exception(warn)
            else:
                return self.state
        except Exception as e:
            print(e)
