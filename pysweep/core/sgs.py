#Programmer: Anthony Walker
#This file is for global variable initialization
def initializeGlobals():
    """Use this function to initialize globals."""
    #Setting global variables
    global CPUArray,cpu
    CPUArray,cpu = None,None

def setGlobalModule(solver):
    global cpu
    cpu = solver.cpu
