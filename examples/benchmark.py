import esr
import esr.generation.duplicate_checker
from esr.fitting.likelihood import Likelihood
import esr.fitting.test_all
import matplotlib.pyplot as plt
import os
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

max_comp = 7
runname = 'core_maths'
rerun_fit = False

# First generate the equations to fit
esr_dir = os.path.dirname(esr.__file__)
for comp in range(1, max_comp + 1):
    fname = os.path.join(esr_dir, 'function_library', runname,
                         f'compl_{comp}', f'all_equations_{comp}.txt')
    if os.path.exists(fname):
        if rank == 0:
            print(f"File exists: {os.path.basename(fname)}", flush=True)
    else:
        esr.generation.duplicate_checker.main(runname, comp)

# Setup likelihood object


class MSE(Likelihood):
    def __init__(self,):
        super().__init__('feynman_I_6_2a.tsv', 'None', runname)
        self.xvar, self.yvar = np.loadtxt(
            'feynman_I_6_2a.tsv', unpack=True, delimiter='\t', skiprows=1)
        self.is_mse = True  #  Warning to not use MSE for DL

    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function. Here it is (y-ypred)^2
        Note that this is technically not a log-likelihood, but the function
        name is required to be accessed by other functions.

        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y

        Returns:
            :nll (float): - log(likelihood) for this function and parameters

        """

        ypred = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(ypred)):
            return np.inf
        nll = np.mean((ypred - self.yvar) ** 2)
        if np.isnan(nll):
            return np.inf
        return nll


# Run the fitting
likelihood = MSE()
for comp in range(1, max_comp + 1):
    fname = os.path.join(likelihood.out_dir, f'negloglike_comp{comp}.dat')
    if os.path.exists(fname) and not rerun_fit:
        if rank == 0:
            print(f"File exists: {os.path.basename(fname)}", flush=True)
    else:
        if rank == 0:
            print(f"Running fitting for complexity {comp}", flush=True)
        esr.fitting.test_all.main(comp, likelihood, Niter_params=[
                                  10], Nconv_params=[3])

comm.Barrier()  # Ensure all processes have completed before proceeding

if rank == 0:
    # Plot the Pareto front
    all_logL = np.empty(max_comp)
    for comp in range(1, max_comp + 1):
        fname = os.path.join(likelihood.out_dir, f'negloglike_comp{comp}.dat')
        logL = np.loadtxt(fname)[:, 0]
        i = np.nanargmin(logL)
        all_logL[comp - 1] = logL[i]
        fname = os.path.join(esr_dir, 'function_library', runname,
                             f'compl_{comp}', f'unique_equations_{comp}.txt')
        print('\n', comp, i)
        with open(fname, 'r') as f:
            for j, line in enumerate(f):
                if j == i:
                    print(line.strip())
                    break
    all_comp = np.arange(1, max_comp + 1)
    plt.plot(all_comp, all_logL, marker='o')
    plt.xlabel('Complexity')
    plt.ylabel('MSE')
    plt.title('Pareto front for core maths')
    plt.yscale('log')
    plt.xticks(all_comp)
    plt.tight_layout()
    plt.show()
