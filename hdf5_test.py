from mpi4py import MPI
import h5py

rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('velocity', (4,4), dtype='i')

for i in range(4):
    for j in range(4):
        dset[i,j] = rank*i*j

f.close()
