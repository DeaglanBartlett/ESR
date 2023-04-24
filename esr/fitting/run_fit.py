import sys
from mpi4py import MPI

import esr.fitting.test_all as test_all
import esr.fitting.test_all_Fisher as test_all_Fisher
import esr.fitting.match as match
import esr.fitting.combine_DL as combine_DL
import esr.fitting.plot as plot

from esr.fitting.likelihood import CCLikelihood, PanthLikelihood, MockLikelihood, SimpleLikelihood

try_integration=False

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
likelihood = CCLikelihood()
#likelihood = PanthLikelihood()
#nz = 3200 # 320, 640, 800, 1000, 3200
#yfracerr = 0.2  # 0.01, 0.05, 0.1, 0.2
#likelihood = MockLikelihood(nz, yfracerr)
#likelihood = SimpleLikelihood("feynman_I_6_2a.tsv")

print_text('test_all')
test_all.main(comp, likelihood, tmax=5, try_integration=try_integration)
comm.Barrier()

print_text('test_all_Fisher')
test_all_Fisher.main(comp, likelihood, tmax=tmax, try_integration=try_integration)
comm.Barrier()

print_text('match')
match.main(comp, likelihood, tmax=tmax, try_integration=try_integration)
comm.Barrier()

print_text('combine_DL')
combine_DL.main(comp, likelihood)
comm.Barrier()

print_text('plot')
if rank == 0:
    plot.main(comp, likelihood, tmax=tmax, try_integration=try_integration)

