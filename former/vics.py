#Programmer: Anthony Walker
#This fill contains classes with specific sets of initial conditions
import math
import numpy as np

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
        pass

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
        self.alpha = math.pi/4  #Angle of attack
        self.M_inf = math.sqrt(2/self.gamma)    #Mach number
        self.P_inf = 1  #pressure kPa
        self.T_inf = 1 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = self.M_inf*5*math.sqrt(2)/(4*math.pi)*math.exp(0.5) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c
        self.generate(self,self.L,self.L,npts,npts)

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
        self.alpha = math.pi/2  #Angle of attack
        self.M_inf = 0.4    #Mach number
        self.P_inf = 1/(self.gamma*self.M_inf*self.M_inf)  #Pressure kPa
        self.T_inf = 1 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = self.M_inf*27/(4*math.pi)*math.exp(2/9) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c
        self.generate(self,self.L,self.L,npts,npts)

    def HestWar(self,gamma,npts = None):
        """ Initializer function that uses initial conditions from

        Hesthaven, J. S. and Warburton, T., Nodal Discontinuous Galerkin Methods: Algorithms, Analysis, and Applications ,
        Vol. 54 of Texts in Applied Mathematics, Springer New York, 2008.

        User must specify: gamma
        """
        #Dimensions
        self.L = 5  #Half of the computational domain length and height
        self.r_c = math.sqrt(0.5)  #Vortex Radius

        #Gas properties
        self.gamma = gamma
        self.R_gas = None  #J/kgK

        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = 0  #Angle of attack
        self.M_inf = math.sqrt(1/self.gamma)    #Mach number
        self.P_inf = 1  #Pressure kPa
        self.T_inf = 1 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = self.M_inf*5/(2*math.pi)*math.exp(1) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c
        self.generate(self,self.L,self.L,npts,npts)

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
        self.generate(self,self.L,self.L,npts,npts)

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
        self.generate(self,self.L,self.L,npts,npts)

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
        self.beta = 5/(2*math.pi*math.sqrt(self.gamma))*math.exp(0.5) #Maximum perturbation strength
        self.speed_of_sound()
        self.args = (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.c, self.sigma, self.beta, self.r_c, self.L)
        npts = 2*self.L/self.r_c
        self.generate(self,self.L,self.L,npts,npts)

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
        self.generate(self,self.L,self.L,npts,npts)



    #-----------------------Functions used by Initializers-----------------------

    def generate(self,X,Y,npx,npy,t=0,x0=0,y0=0):
        """Use this function to generate the grid from the selected inital conditions."""
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
            if rho_inf is not None:
                self.c = np.sqrt(self.gamma*self.P_inf/self.rho_inf)
            elif R_gas is not None:
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
