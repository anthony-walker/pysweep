NVCC=nvcc
NVCCFLAGS=-Xcompiler -fPIC -shared
PSLIB=./pysweep/lib/
PS=./pysweep/

dsweep: ./pysweep/sweep/ccore/dsweep.cu
	$(NVCC) $(NVCCFLAGS) -o $(PSLIB)libdsweep.so ./pysweep/sweep/ccore/dsweep.cu
