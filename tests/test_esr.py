import numpy as np
import os
import sys
import matplotlib.pyplot as plt

import esr.generation.duplicate_checker
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
import esr.fitting.plot
from esr.fitting.likelihood import CCLikelihood, Likelihood, PoissonLikelihood
from esr.fitting.fit_single import single_function
import esr.plotting.plot

def test_cc():

    comp = 5
    likelihood = CCLikelihood()
    esr.generation.duplicate_checker.main('core_maths', comp)
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

    # Test results match Table 1 of arXiv:2211.11461
    assert os.path.exists(likelihood.out_dir)
    fname = os.path.join(likelihood.out_dir,f'final_{comp}.dat')
    with open(fname, 'r') as f:
        best = f.readline().split(';')
    assert int(best[0]) == 0  # Rank
    assert best[1] == 'a0*x**2'  # best function
    assert np.isclose(float(best[2]), 16.39, atol=2e-2)  # logL
    assert np.isclose(float(best[4]), 8.36, atol=2e-2)   # Residuals
    assert np.isclose(float(best[5]), 2.53, atol=2e-2)   # Parameter
    assert np.isclose(float(best[6]), 5.49, atol=2e-2)   # Function
    assert np.isclose(float(best[7]), 3883.44, atol=10)  # Best-fit a0
    assert np.all(np.array(best[8:], dtype=float) == 0)  # Other parameters

    print(likelihood.out_dir)

    likelihood = CCLikelihood()
    labels = ["+", "a0", "*", "a1", "pow", "x", "3"]
    basis_functions = [["x", "a"],  # type0
                    ["inv"],  # type1
                    ["+", "*", "-", "/", "pow"]]  # type2

    single_function(labels,
                basis_functions,
                likelihood,
                verbose=True)

    return


def test_gaussian(monkeypatch): 

    monkeypatch.setattr(plt, 'show', lambda: None)

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
    x = np.random.uniform(0.1, 5, 100)
    y = 0.5 * x ** 2
    yerr = np.full(x.shape, 1.0)
    y = y + yerr * np.random.normal(size=len(x))
    np.savetxt('data.txt', np.array([x, y, yerr]).T)
    sys.stdout.flush()
    likelihood = GaussLikelihood('data.txt', 'gauss_example', data_dir=os.getcwd())
    comp = 5
    esr.generation.duplicate_checker.main('core_maths', comp)
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

    return


def test_poisson(monkeypatch):

    monkeypatch.setattr(plt, 'show', lambda: None)

    # Run the Poisson examples
    np.random.seed(123)
    x = np.random.uniform(0.1, 5, 100)
    y = 0.5 * x ** 2
    y = np.random.poisson(y)
    np.savetxt('data.txt', np.array([x, y]).T)
    sys.stdout.flush()

    likelihood = PoissonLikelihood('data.txt', 'poisson_example', data_dir=os.getcwd())
    comp = 5
    esr.generation.duplicate_checker.main('core_maths', comp)
    esr.fitting.test_all.main(comp, likelihood)
    esr.fitting.test_all_Fisher.main(comp, likelihood)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    esr.fitting.plot.main(comp, likelihood)

    # Plot the pareto front for the Poisson example
    esr.plotting.plot.pareto_plot(likelihood.out_dir, 'pareto.png', do_DL=True, do_logL=True)

    return


def test_function_making():

    for basis_set in ['core_maths', 'ext_maths', 'keep_duplicates', 'osc_maths']:
        for comp in range(1, 5):
            esr.generation.duplicate_checker.main(basis_set, comp)

    return

test_cc()