#Programmer: Anthony Walker
#README:
#This is a function an example of the required setup for a space solver
#Arguments:
# arr - this is an ar

def ex_dfdt(arr,d,shape):
    """This is an example function for implementation."""
    ddt = np.zeros(arr.shape())
    for x in shape[0]:
        for y in shape[1]:
            ddt[x,y] = (arr[x+1]+arr[x])/d[0]+(arr[y+1]+arr[y])/d[1]
    return ddt
