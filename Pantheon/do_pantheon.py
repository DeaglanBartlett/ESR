import sys
from mpi4py import MPI

import test_all
import test_all_Fisher
import match
import combine_DL
import plot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def print_text(text):
    if rank != 0:
        return
    stars = ["*" * 20]
    print('\n')
    print(*stars)
    print('\n')
    print(text)
    print(*stars)
    print('\n')
    return
    
compl = int(sys.argv[1])
tmax = 5
    
print_text('COMPLEXITY = %i'%compl)
#test_all.main(compl, tmax=tmax)
comm.Barrier()

print_text('test_all_Fisher')
#test_all_Fisher.main(compl, tmax=tmax)
comm.Barrier()

print_text('match')
match.main(compl, tmax=tmax)
comm.Barrier()

print_text('combine_DL')
combine_DL.main(compl)
comm.Barrier()

print_text('plot')
if rank == 0:
    plot.main(compl, tmax=tmax)
