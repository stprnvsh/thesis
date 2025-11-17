# minimal_allreduce_test.py
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

local_value = np.array([rank], dtype=np.float64) # Example local value
print(f"Rank {rank}: Before allreduce, local_value = {local_value[0]}")

reduced_value = np.array([1.0]) # Initialize for result
comm.Allreduce(MPI.IN_PLACE, reduced_value, op=MPI.SUM) # In-place allreduce

print(f"Rank {rank}: After allreduce, reduced_value = {reduced_value[0]}")
comm.Barrier() # Barrier after to ensure all ranks finish printing
if rank == 0:
    print("Minimal allreduce test completed.")