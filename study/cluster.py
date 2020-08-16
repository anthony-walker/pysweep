import numpy


def generateArraySizes():
    """Use this function to generate array sizes based on block sizes."""
    blocksizes = [8, 12, 16, 24, 32] #blocksizes with most options
    arraystart = 0
    divisible =[False,False]
    arraysizes = []
    for i in range(5):
        arraystart += 32*20
        while not numpy.all(divisible):
            arraystart+=blocksizes[-1]
            divisible =[arraystart%bs==0 for bs in blocksizes]
        arraysizes.append(arraystart)
    return arraysizes

if __name__ == "__main__":
    asizes = generateArraySizes()
    print(asizes)