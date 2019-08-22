#Programmer: Anthony Walker
#This file contains the analytical solution to the euler vortex
#problem in two dimensions.

#Import statements
import numpy as np
import os
import h5py

def vortex(cvics,X,Y,npx,npy,times=(0,),x0=0,y0=0):
    """This is the primary method to solve the euler vortex that is centered at the origin with periodic boundary conditions
    properties are obtained from the vics object, cvics.
       The center can be changed with x0 and y0.
       the time step or steps can be changed with times but it must be a tuple.
       X & Y are dimensions of the domain.
       npx, and npy are the number of points the in respective direction

       This solution was obtained from

       Persson, P. O., Bonet, J., & Peraire, J. (2009).
       Discontinuous Galerkin solution of the Navier–Stokes equations on deformable domains.
       Computer Methods in Applied Mechanics and Engineering, 198(17-20), 1585-1595.

    """
    alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, c, sigma, beta, r_c, L = cvics.get_args()
    PI = np.pi
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
    state = np.zeros(xgrid.shape+np.shape(times)+(4,),dtype=np.float64) #Initialization of state
    idxs = np.ndindex(xgrid.shape)  #Indicies of the array
    #Solving times
    for t in times:
        for idx in idxs:
            #getting x and y location
            x = xgrid[idx]
            y = ygrid[idx]

            #differences from origin
            dx0 = (x-x0)
            dy0 = (y-y0)

            #Common terms
            uterm = (dx0-u_bar*t)
            vterm = (dy0-v_bar*t)
            pterm = beta*beta*(gamma-1)*M_inf*M_inf/(8*PI*PI)

            #function calculation f(x,y,t)
            fx = uterm*uterm
            fy = vterm*vterm
            f = (1-fx-fy)/(r_c*r_c)

            #Finding state variables
            state[idx+(t,0)] = rho_inf*(1-pterm*np.exp(f))**(1/(gamma-1)) #density
            state[idx+(t,1)] = V_inf*(np.cos(alpha)-beta*vterm/(2*PI*r_c)*np.exp(f/2)) #x velocity
            state[idx+(t,2)] = V_inf*(np.sin(alpha)-beta*uterm/(2*PI*r_c)*np.exp(f/2)) #y velocity
            state[idx+(t,3)] = P_inf*(1-pterm*np.exp(f))**(gamma/(gamma-1)) #pressure
    return state



def create_vortex_data(cvics,X,Y,npx,npy, times=(0,), x0=0, y0=0, dirpath = "./vortex"):
    """Use this function to create vortex data from the vortex funciton.
    Note, this function will create a directory called "vortex#" unless otherwise specified.
    Subdirectories will be created for density, x-velocity, y-velocity, and pressure.
    files in these directories will be labeled in time order e.g. density1.txt, density2.txt etc.
    the time will be stored in each file as well.
    """
    #Making directories
    if os.path.isdir:
        ctr = 0
        dirpath+=str(ctr)
        ctr+=1
        while os.path.isdir(dirpath):
            dirpath=dirpath[:-len(str(ctr-1))]
            dirpath+=str(ctr)
            ctr+=1
        rpath = dirpath+"/density/"
        xpath = dirpath+"/x-velocity/"
        ypath = dirpath+"/y-velocity/"
        ppath = dirpath+"/pressure/"
        os.mkdir(dirpath)
        os.mkdir(rpath)
        os.mkdir(xpath)
        os.mkdir(ypath)
        os.mkdir(ppath)
        str_array = [rpath+"density",xpath+"x-velocity",ypath+"y-velocity",ppath+"pressure"]
        text_end = ".txt"
    state = vortex(cvics,X,Y,npx,npy, times, x0=x0, y0=y0)
    states = np.array_split(state,len(times),axis=2)
    tsc = 0
    for temp_state in states:
        temp_states = np.array_split(temp_state,4,axis=3)
        fc = 0
        for prop_state in temp_states:
            temp_prop_state = np.reshape(prop_state,prop_state.shape[:2])
            np.savetxt(str_array[fc]+str(tsc),temp_prop_state,fmt="%.10f",delimiter=",",header="time_step: "+str(times[tsc]))
            fc+=1
        tsc+=1
    return state

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
        self.L = 5   #Half of the computational domain length and height
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
        npts = 2*self.L/self.r_c
        self.vortex_args = self.L,self.L,npts,npts
        self.state = vortex(self,self.L,self.L,npts,npts)

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
        npts = 2*self.L/self.r_c
        self.vortex_args = self.L,self.L,npts,npts
        self.state = vortex(self,self.L,self.L,npts,npts)

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
        npts = 2*self.L/self.r_c
        self.vortex_args = self.L,self.L,npts,npts
        self.state = vortex(self,self.L,self.L,npts,npts)

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
        npts = 2*self.L/self.r_c
        self.vortex_args = self.L,self.L,npts,npts
        self.state = vortex(self,self.L,self.L,npts,npts)

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
        npts = 2*self.L/self.r_c
        self.vortex_args = self.L,self.L,npts,npts
        self.state = vortex(self,self.L,self.L,npts,npts)

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
        npts = 2*self.L/self.r_c
        self.vortex_args = self.L,self.L,npts,npts
        self.state = vortex(self,self.L,self.L,npts,npts)

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
        npts = 2*self.L/self.r_c
        self.vortex_args = self.L,self.L,npts,npts
        self.state = vortex(self,self.L,self.L,npts,npts)



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


if __name__ == "__main__":
    #HighOrder Workshop Fast
    gamma = 1.4
    times = np.linspace(1,5,10)
    cvics = vics()  #Creating vortex ics object
    cvics.Shu(gamma) #Initializing with Shu parameters
    X,Y,npx,npy = cvics.vortex_args
    #Calling analytical solution
    create_vortex_data(cvics,X,Y,npx,npy,times=(0,0.1))
