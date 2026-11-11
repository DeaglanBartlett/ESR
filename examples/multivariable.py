"""
Run ESR on mock data from a function of two variables, f(x, y) = 2 x y.

The catalogue for the variables x and y is generated into
``function_library/core_maths_x_y`` and every function up to complexity 5 is
fitted to the data with a Gaussian likelihood. The true function is recovered
at complexity 5, with its parameter snapped to the integer 2.

Run with e.g. ``mpirun -n 4 python multivariable.py``.
"""

import os

import numpy as np
from mpi4py import MPI

import esr.fitting.combine_dl
import esr.fitting.match
import esr.fitting.plot
import esr.fitting.test_all
import esr.fitting.test_all_fisher
import esr.generation.duplicate_checker
from esr.fitting.likelihood import GaussLikelihood

variables = ["x", "y"]
max_comp = 5

# Mock data: columns are x, y, f, f_err
data_dir = os.getcwd()
if MPI.COMM_WORLD.Get_rank() == 0:
    rng = np.random.default_rng(0)
    n = 60
    xv = rng.uniform(0.5, 3, n)
    yv = rng.uniform(0.5, 3, n)
    err = 0.05 * np.ones(n)
    f = 2.0 * xv * yv + err * rng.standard_normal(n)
    np.savetxt(f"{data_dir}/multivariable_data.txt", np.c_[xv, yv, f, err])
MPI.COMM_WORLD.Barrier()

likelihood = GaussLikelihood(
    "multivariable_data.txt",
    "multivariable_example",
    data_dir=data_dir,
    variables=variables,
)

for comp in range(1, max_comp + 1):
    esr.generation.duplicate_checker.main("core_maths", comp, variables=variables)
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_dl.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)
