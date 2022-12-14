import sys

from test_all import optimise_fun
from test_all_Fisher import convert_params

sys.path.insert(0, '../generation/')
import generator
import simplifier

def single_function(labels, basis_functions, likelihood, pmin=0, pmax=5, tmax=5, try_integration=False, verbose=False):
    """Run end-to-end fitting of function for a single function
    
    Args:
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :pmin (float, default=0.): minimum value for each parameter to consider when generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to consider when generating initial guess
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :verbose (bool, default=True): Whether to print results (True) or not (False)
    
    Returns:
         :negloglike (float): the minimum value of -log(likelihood) (corresponding to the maximum likelihood)
         :DL (float): the description length of this function
    
    """

    # (1) Convert the string to a sympy function
    s = generator.labels_to_shape(labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    fstr = generator.node_to_string(0, tree, labels)
    max_param = simplifier.get_max_param([fstr], verbose=verbose)
    fstr, fsym = simplifier.initial_sympify([fstr], max_param, parallel=False, verbose=verbose)
    fstr = fstr[0]
    fsym = fsym[fstr]

    # (2) Fit this function to the data
    chi2, params = optimise_fun(fstr,
                            likelihood,
                            tmax,
                            pmin,
                            pmax,
                            try_integration=try_integration)

    # (3) Obtain the Fisher matrix for this function
    fcn, eq, integrated = likelihood.run_sympify(fstr,
                                            tmax=tmax, 
                                            try_integration=try_integration)
    params, negloglike, deriv, codelen = convert_params(fcn, eq, integrated, params, likelihood, chi2) 
    if verbose:
        print('theta_ML:', params)
        print('Residuals:', negloglike, chi2)
        print('Parameter:', codelen)

    # (4) Get the functional complexity
    param_list = ['a%i'%j for j in range(max_param)]
    aifeyn = generator.aifeyn_complexity(labels, param_list)
    if verbose:
        print('Function:', aifeyn)

    # (5) Combine to get description length
    DL = negloglike + codelen + aifeyn
    if verbose:
        print('\nDescription length:', DL)

    return negloglike, DL

