import esr.generation.duplicate_checker as duplicate_checker 
import time
from mpi4py import MPI
from memory_profiler import memory_usage

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

runname = 'core_maths'
compl = 7
nrun = 10

# TIMING
start = time.time()
for _ in range(nrun):
    duplicate_checker.main(runname, compl)
end = time.time()

# MEMORY USAGE IN MB (https://pypi.org/project/memory-profiler/)
mem_usage = memory_usage((duplicate_checker.main, (runname, compl),))

comm.Barrier()
if rank == 0:
    total = end - start
    per_run = total / nrun

    stars = '\n' + ''.join(['*']*35) + '\n'
    print(stars)
    print("RUN:", runname)
    print("COMPLEXITY:", compl)
    print("CORES:", size)
    print("NRUN:", nrun)
    print("ALL RUNS:", total)
    print("PER RUN:", per_run)
    print("MAX MEMORY USAGE: %s MB"%max(mem_usage))
    print(stars)
