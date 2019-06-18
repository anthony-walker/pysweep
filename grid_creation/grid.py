#Programmer: Anthony Walker
#README: This file deals with creation and decomposition of the computational grid
#The dimensions of the computational grid are 2 times the number of
#spatial dimensions. This is because PDEs also change in time and each location
#multiple values such as temperature pressure and etc...

#imports
import numpy as np

class grid(np.ndarray):
    """Grid is a simple wrapper for a np.ndarray that can handle boundary conditions."""
    def __new__(cls,input_array,cbc=True): #shape, dtype=float, buffer=None, offset=0, strides=None, order=None, cbc=True):
        # obj = super(grid, subtype).__new__(subtype,shape, dtype, buffer, offset, strides, order)
        obj = np.asarray(input_array).view(cls)
        obj.cbc = cbc
        return obj

    def __array_finalize__(self, obj):
        """this method initializes the new object."""
        self.cbc = getattr(obj,'cbc',None)
        self.start = np.ones(self.shape)
        self.end = np.shape-1

    def __getitem__(self,index):
        """Overloading of __getitem__."""
        return super(grid,self).__getitem__(index)

    def __iter__(self):
        """Overloading iter for ndarray."""
        super(grid,self).__iter__()
        return self

    def __next__(self):
        """Overloading the next function for ndarray."""
        pass

if __name__ == "__main__":
    y0 = np.zeros((10,10,7))+1
    t0 = 0
    t_b = 1
    dt = 0.1
    gridTest = grid(y0)
    # print()
    # mySplit = np.split(gridTest,2)
    # print(type(gridTest))
    # print(mySplit)
    for g in gridTest:
        for x in g:
            print(x)
