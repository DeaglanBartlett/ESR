# Copyright 2023 Deaglan J. Bartlett
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.

"""
Script to run the examples given in the documenation (https://esr.readthedocs.io/en/latest/)
"""

import numpy as np
import os
import sys
from mpi4py import MPI

import esr.generation.duplicate_checker
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.likelihood import CCLikelihood, Likelihood, PoissonLikelihood
from esr.fitting.fit_single import single_function
import esr.plotting.plot

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate functions needed for examples
# runname = 'core_maths'
# for comp in range(1, 6):
#     esr.generation.duplicate_checker.main(runname, comp)

# Fit the CC data with complexity 5 functions
# comp = 5
# likelihood = CCLikelihood()
# esr.fitting.test_all.main(comp, likelihood)
# esr.fitting.test_all_Fisher.main(comp, likelihood)
# esr.fitting.match.main(comp, likelihood)
# esr.fitting.combine_DL.main(comp, likelihood)
# esr.fitting.plot.main(comp, likelihood)

# Define a custom likelihood class
class GaussLikelihood(Likelihood):

    def __init__(self, data_file, run_name, data_dir=None):
        """Likelihood class used to fit a function directly using a Gaussian likelihood
 
        """
 
        super().__init__(data_file, data_file, run_name, data_dir=data_dir)
        self.ylabel = r'$y$'    # for plotting
        self.xvar, self.yvar, self.yerr = np.loadtxt(self.data_file, unpack=True)

    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function.
 
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y
     
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
 
 
        """

        ypred = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(ypred)):
            return np.inf
        nll = np.sum(0.5 * (ypred - self.yvar) ** 2 / self.yerr ** 2 + 0.5 * np.log(2 * np.pi) + np.log(self.yerr))
        if np.isnan(nll):
            return np.inf
        return nll
   
# Run the Gaussian example
np.random.seed(123)
if rank == 0:
    x = np.random.uniform(0.1, 5, 100)
    y = 0.5 * x ** 2
    yerr = np.full(x.shape, 1.0)
    y = y + yerr * np.random.normal(size=len(x))
    np.savetxt('data.txt', np.array([x, y, yerr]).T)
sys.stdout.flush()
comm.Barrier()
likelihood = GaussLikelihood('data.txt', 'gauss_example', data_dir=os.getcwd())
comm.Barrier()
for comp in range(1, 6):
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

# Run the Poisson examples
np.random.seed(123)
if rank == 0:
    x = np.random.uniform(0.1, 5, 100)
    y = 0.5 * x ** 2
    yerr = np.full(x.shape, 0.1)
    y = np.random.poisson(y)
    np.savetxt('data.txt', np.array([x, y]).T)
sys.stdout.flush()
comm.Barrier()

likelihood = PoissonLikelihood('data.txt', 'poisson_example', data_dir=os.getcwd())
for comp in range(1, 6):
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

# Plot the pareto front for the Poisson example
if rank == 0:
    print(likelihood.out_dir)
    esr.plotting.plot.pareto_plot(likelihood.out_dir, 'pareto.png', do_DL=True, do_logL=True)
    
# Fit the CC data with a single function (LCDM)
if rank == 0:
    likelihood = CCLikelihood()
    labels = ["+", "a0", "*", "a1", "pow", "x", "3"]
    basis_functions = [["x", "a"],  # type0
                    ["inv"],  # type1
                    ["+", "*", "-", "/", "pow"]]  # type2

    logl_lcdm_cc, dl_lcdm_cc = single_function(labels,
                                                    basis_functions,
                                                    likelihood,
                                                    verbose=True)

