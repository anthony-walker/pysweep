#Programmer: Anthony Walker
#This fill contains multiple methods for development testing
from .pysweep_lambda import sweep_lambda
import pycuda.driver as cuda

def create_blocks(arr,block_size):
    """Use this function to create blocks from an array based on the given blocksize."""
    sx = arr.shape[1]/block_size[0] #split x dimensions
    sy = arr.shape[2]/block_size[1] #split y dimension
    num_blocks = sx*sy  #number of blocks
    hcs = lambda x,s: np.array_split(x,s,axis=x.shape.index(max(x.shape[1:])))   #Lambda function for creating blocks for cpu testing
    blocks =  [item for subarr in hcs(arr,sx) for item in hcs(subarr,sy)]    #applying lambda function
    return blocks

def cpu_speed(arr,sweep_fcn,cpu_fcn,block_size,ops,num_tries=1):
    """This function compares the speed of a block calculation to determine the affinity."""
    sx = int(arr.shape[2]/block_size[0]) #split x dimensions
    sy = int(arr.shape[3]/block_size[1]) #split y dimension

    num_blocks = sx*sy  #number of blocks
    hcs = lambda x,s: np.array_split(x,s,axis=x.shape.index(max(x.shape)))   #Lambda function for creating blocks for cpu testing
    cpu_test_blocks =  [item for subarr in hcs(arr,sx) for item in hcs(subarr,sy)]    #applying lambda function
    cpu_test_fcn = sweep_lambda((sweep_fcn,cpu_fcn,0,ops)) #Creating a function that can be called with only the block list

    #------------------------Testing CPU Performance---------------------------#
    pool = mp.Pool(cores)   #Pool allocation
    cpu_performance = 0
    for i in range(num_tries):
        start_cpu = time.time()
        cpu_res = pool.map_async(cpu_test_fcn,cpu_test_blocks)
        nts = cpu_res.get()
        stop_cpu = time.time()
        cpu_performance += np.prod(cpu_test_blocks[0].shape)*num_blocks/(stop_cpu-start_cpu)
    #-------------------------Ending CPU Performance Testing--------------------#
    pool.close()
    pool.join()
    cpu_performance /= num_tries
    # print("Average CPU Performance:", cpu_performance)
    return cpu_performance

def gpu_speed(arr,source_mod,cpu_fcn,block_size,ops,num_tries=1):
    """Use this function to measure the gpu's performance."""
    #Creating Execution model parameters
    grid_size = (int(arr.shape[2]/block_size[0]),int(arr.shape[3]/block_size[1]))   #Grid size
    shared_size = (arr[0,:,:block_size[0],:block_size[1]].nbytes) #Creating size of shared array

    #Creating events to record
    start_gpu = cuda.Event()
    stop_gpu = cuda.Event()

    #Making sure array is correct type
    arr = arr.astype(np.float32)
    arr_gpu = cuda.mem_alloc(arr.nbytes)
    cuda.memcpy_htod(arr_gpu,arr)

    #Getting GPU Function
    gpu_fcn = source_mod.get_function("UpPyramid")
    # print(gpu_fcn.num_regs)

    #------------------------Testing GPU Performance---------------------------#
    gpu_performance = 0 #Allocating gpu performance
    for i in range(num_tries):
        start_gpu.record()
        gpu_fcn(arr_gpu,grid=grid_size, block=block_size,shared=shared_size)
        stop_gpu.record()
        stop_gpu.synchronize()
        gpu_performance += np.prod(grid_size)*np.prod(block_size)/(start_gpu.time_till(stop_gpu)*1e-3 )
    gpu_performance /= num_tries    #finding average by division of number of tries
    # print("Average GPU Performance:", gpu_performance)
    return gpu_performance
    #-------------------------Ending GPU Performance Testing--------------------#

def getDeviceAttrs(devNum=0,print_device = False):
    """Use this function to get device attributes and print them"""
    device = cuda.Device(devNum)
    dev_name = device.name()
    dev_pci_bus_id = device.pci_bus_id()
    dev_attrs = device.get_attributes()
    dev_attrs["DEVICE_NAME"]=dev_name
    if print_device:
        for x in dev_attrs:
            print(x,": ",dev_attrs[x])
    return dev_attrs
