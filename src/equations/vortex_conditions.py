#Programmer: Anthony Walker
#This fill contains classes with specific sets of initial conditions
import math

class Shu(object):
    """This class uses initial conditions in Shu's work."""
    def __init__(self):
        #Dimensions
        self.L = 5   #Half of the computational domain length and height
        self.R = 1  #Characteristic length of the grid

        #Gas properties
        self.gamma = 1.4
        self.R_gas = None  #J/kgK

        #Freestream variables
        self.rho_inf = 1    #Denisty
        self.alpha = math.pi  #Angle of attack
        self.M_inf = math.sqrt(2/self.gamma)    #Mach number
        self.P_inf = 1  #pressure kPa
        self.T_inf = 1 #Temperature K

        #Gaussian Variables
        self.sigma = 1  #Standard deviation
        self.beta = self.M_inf*5*math.sqrt(2)/(4*math.pi)*math.exp(0.5) #Maximum perturbation strength

    def get(self):
        """Use this function to return a tuple of these conditions.
        (alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, sigma, beta, R, L)
        """
        return (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.sigma, self.beta, self.R, self.L)

class HOWF(object):
    """This class uses initial conditions in HighOrder Workshop (Fast) work."""
    def __init__(self):
        #Dimensions
        self.L = 0.05   #Half of the computational domain length and height
        self.R = 0.005  #Characteristic length of the grid

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

    def get(self):
        """Use this function to return a tuple of these conditions.
        (alpha, M_inf, P_inf, rho_inf, T_inf, gamma, R_gas, sigma, beta, R, L)
        """
        return (self.alpha, self.M_inf, self.P_inf, self.rho_inf, self.T_inf, self.gamma, self.R_gas, self.sigma, self.beta, self.R, self.L)
