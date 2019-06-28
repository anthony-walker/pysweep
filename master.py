#Programmer: Anthony Walker
#This is the main file for running and testing the swept solver
from src import *

#Properties
gamma = 1.4
times = np.linspace(1,5,10)

#Analytical properties
avics = vics()  #Creating vortex ics object
avics.Shu(gamma) #Initializing with Shu parameters

#Calling analytical solution

def analytical():
    """Use this funciton to solve the analytical euler vortex."""
    create_vortex_data(cvics,X,Y,npx,npy,times=(0,0.1))


if __name__ == "__main__":
    analytical()
