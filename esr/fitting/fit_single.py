import sys

from esr.fitting.test_all import optimise_fun
from esr.fitting.test_all_Fisher import convert_params

import esr.generation.generator as generator
import esr.generation.simplifier as simplifier

def is_float(string):
    """Determine whether a string is a float or not
    
    Args:
        :string (str): The string to check
        
    Returns:
        bool: Whether the string is a float (True) or not (False).
    
    """
    try:
        float(string)
        return True
    except ValueError:
        return False

def single_function(labels, basis_functions, likelihood, pmin=0, pmax=5, tmax=5, try_integration=False, verbose=False, Niter=30, Nconv=5, log_opt=False):
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
        :Niter (int, default=30): Maximum number of parameter optimisation iterations to attempt.
        :Nconv (int, default=5): If we find Nconv solutions for the parameters which are within a logL of 0.5 of the best, we say we have converged and stop optimising parameters
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in log space
    
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
                            try_integration=try_integration,
                            max_param=max_param,
                            Niter=Niter,
                            Nconv=Nconv,
                            log_opt=log_opt)
                            
    if likelihood.is_mse:
        print('Not computing DL as using MSE')
        DL = np.nan
        negloglike = chi2
    else:
        # (3) Obtain the Fisher matrix for this function
        fcn, eq, integrated = likelihood.run_sympify(fstr,
                                                tmax=tmax,
                                                try_integration=try_integration)
        params, negloglike, deriv, codelen = convert_params(fcn, eq, integrated, params, likelihood, chi2, max_param=max_param)
        if verbose:
            print('\ntheta_ML:', params)
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
    
    
def fit_from_string(fun, basis_functions, likelihood, pmin=0, pmax=5, tmax=5, try_integration=False, verbose=False, Niter=30, Nconv=5, maxvar=20, log_opt=False):
    """Run end-to-end fitting of function for a single function, given as a string. Note that this is not guaranteed to find the optimimum representation as a tree, so there could be a lower description-length representation of the function
    
    Args:
        :fun (str): String representation of the function to be fitted
        :basis_functions (list): list of lists basis functions. basis_functions[0] are nullary, basis_functions[1] are unary and basis_functions[2] are binary operators
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :pmin (float, default=0.): minimum value for each parameter to consider when generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to consider when generating initial guess
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :verbose (bool, default=True): Whether to print results (True) or not (False)
        :Niter (int, default=30): Maximum number of parameter optimisation iterations to attempt.
        :Nconv (int, default=5): If we find Nconv solutions for the parameters which are within a logL of 0.5 of the best, we say we have converged and stop optimising parameters
        :maxvar (int): The maximum number of variables which could appear in the function
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in log space
    
    Returns:
         :negloglike (float): the minimum value of -log(likelihood) (corresponding to the maximum likelihood)
         :DL (float): the description length of this function
         :labels (list): list of strings giving node labels of tree
    
    """

    expr, nodes, complexity = generator.string_to_node(fun, basis_functions, evalf=True)
    labels = nodes.to_list(basis_functions)
    
    # Prepare to get parents
    new_labels = [None] * len(labels)
    for j, lab in enumerate(labels):
        if lab == 'Mul':
            new_labels[j] = '*'
            labels[j] = '*'
        elif lab == 'Add':
            new_labels[j] = '+'
            labels[j] = '+'
        else:
            new_labels[j] = lab.lower()
            labels[j] = lab.lower()
    param_idx = [j for j, lab in enumerate(new_labels) if is_float(lab)]
    assert len(param_idx) <= maxvar
    for k, j in enumerate(param_idx):
        new_labels[j] = f'a{k}'

    # Get parent operators
    s = generator.labels_to_shape(new_labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    parents = [None] + [labels[p.parent] for p in tree[1:]]
    
    # Replace floats with symbols (except exponents)
    param_idx = [j for j, lab in enumerate(labels) if is_float(lab) and not (parents[j].lower() =='pow')]
    for k, j in enumerate(param_idx):
        labels[j] = f'a{k}'
    fstr = generator.node_to_string(0, tree, labels)
    print(labels)
    negloglike, DL = single_function(
            labels,
            basis_functions,
            likelihood,
            pmin=pmin,
            pmax=pmax,
            tmax=tmax,
            try_integration=try_integration,
            verbose=verbose,
            Niter=Niter,
            Nconv=Nconv,
            log_opt=log_opt,
    )

    return negloglike, DL, labels

