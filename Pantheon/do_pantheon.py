import sys
from mpi4py import MPI

import test_all
import test_all_Fisher
import match
import combine_DL
import plot

from likelihood import CCLikelihood, PanthLikelihood

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def print_text(text):
    if rank != 0:
        return
    stars = ["*" * 20]
    print('\n')
    print(*stars)
    print(text)
    print(*stars)
    print('\n')
    return
    
comp = int(sys.argv[1])
tmax = 5

print_text('COMPLEXITY = %i'%comp)

print_text('Loading data')
#likelihood = CCLikelihood() 
likelihood = PanthLikelihood()

print_text('test_all')
test_all.main(comp, likelihood, tmax=5)
comm.Barrier()

print_text('test_all_Fisher')
test_all_Fisher.main(comp, likelihood, tmax=tmax)
comm.Barrier()

print_text('match')
match.main(comp, likelihood, tmax=tmax)
comm.Barrier()

print_text('combine_DL')
combine_DL.main(comp, likelihood)
comm.Barrier()

print_text('plot')
if rank == 0:
    plot.main(comp, likelihood, tmax=tmax)
