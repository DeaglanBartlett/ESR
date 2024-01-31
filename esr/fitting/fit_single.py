import sys

from esr.fitting.test_all import optimise_fun
from esr.fitting.test_all_Fisher import convert_params

import esr.generation.generator as generator
import esr.generation.simplifier as simplifier

def single_function(labels, basis_functions, likelihood, pmin=0, pmax=5, tmax=5,
    try_integration=False, verbose=False, Niter=30, Nconv=5, log_opt=False,
    return_params=False):
    """Run end-to-end fitting of function for a single function
    
    Args:
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are
            binary operators
        :likelihood (fitting.likelihood object): object containing data, likelihood
            functions and file paths
        :pmin (float, default=0.): minimum value for each parameter to consider when
            generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to consider when
            generating initial guess
        :tmax (float, default=5.): maximum time in seconds to run any one part of
            simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral,
            whether to try to analytically integrate (True) or just numerically
            integrate (False)
        :verbose (bool, default=True): Whether to print results (True) or not (False)
        :Niter (int, default=30): Maximum number of parameter optimisation iterations
            to attempt.
        :Nconv (int, default=5): If we find Nconv solutions for the parameters which are
            within a logL of 0.5 of the best, we say we have converged and stop
            optimising parameters
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in
            log space
        :return_params (bool, default=False): whether to return the parameters of the
            maximum likelihood point
    
    Returns:
         :negloglike (float): the minimum value of -log(likelihood) (corresponding to
            the maximum likelihood)
         :DL (float): the description length of this function
         :params (optional, list): the maximum likelihood parameters. Only returned if
            `return_params` is true
    
    """

    # (1) Convert the string to a sympy function
    s = generator.labels_to_shape(labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    fstr = generator.node_to_string(0, tree, labels)
    max_param = simplifier.get_max_param([fstr], verbose=verbose)
    fstr, fsym = simplifier.initial_sympify(
        [fstr], max_param, parallel=False, verbose=verbose)
    fstr = fstr[0]
    fsym = fsym[fstr]
    print(fstr)
    # (2) Fit this function to the data
    chi2, params = optimise_fun(fstr,
                            likelihood,
                            tmax,
                            pmin,
                            pmax,
                            try_integration=try_integration,
                            max_param=max_param,
                            Niter_params=[Niter],
                            Nconv_params=[Nconv],
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
        params, negloglike, deriv, codelen = convert_params(
            fcn, eq, integrated, params, likelihood, chi2, max_param=max_param)
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
            
    if return_params:
        return negloglike, DL, params

    return negloglike, DL
    
    
def fit_from_string(fun, basis_functions, likelihood, pmin=0, pmax=5, tmax=5,
    try_integration=False, verbose=False, Niter=30, Nconv=5, maxvar=20,
    log_opt=False, replace_floats=False, return_params=False):
    """Run end-to-end fitting of function for a single function, given as a string.
    Note that this is not guaranteed to find the optimimum representation as a tree,
    so there could be a lower description-length representation of the function
    
    Args:
        :fun (str): String representation of the function to be fitted
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are binary
            operators
        :likelihood (fitting.likelihood object): object containing data, likelihood
            functions and file paths
        :pmin (float, default=0.): minimum value for each parameter to consider when
            generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to consider when
            generating initial guess
        :tmax (float, default=5.): maximum time in seconds to run any one part of
            simplification procedure for a given function
        :try_integration (bool, default=False): when likelihood requires integral,
            whether to try to analytically integrate (True) or just numerically
            integrate (False)
        :verbose (bool, default=True): Whether to print results (True) or not (False)
        :Niter (int, default=30): Maximum number of parameter optimisation iterations
            to attempt.
        :Nconv (int, default=5): If we find Nconv solutions for the parameters which
            are within a logL of 0.5 of the best, we say we have converged and stop
            optimising parameters
        :maxvar (int): The maximum number of variables which could appear in the
            function
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in
            log space
        :replace_floats (bool, default=False): whether to replace any numbers found in
            the function with variables to optimise
        :return_params (bool, default=False): whether to return the parameters of the
            maximum likelihood point
    
    Returns:
         :negloglike (float): the minimum value of -log(likelihood) (corresponding to
            the maximum likelihood)
         :DL (float): the description length of this function
         :labels (list): list of strings giving node labels of tree
         :params (optional, list): the maximum likelihood parameters. Only returned if
            `return_params` is true
    
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
        elif lab == 'Div':
            new_labels[j] = '/'
            labels[j] = '/'
        elif lab == 'Sub':
            new_labels[j] = '-'
            labels[j] = '-'
        else:
            new_labels[j] = lab.lower()
            labels[j] = lab.lower()
    param_idx = [j for j, lab in enumerate(new_labels) if generator.is_float(lab) or (lab.startswith('a') and generator.is_float(lab[1:]))]
    assert len(param_idx) <= maxvar
    for k, j in enumerate(param_idx):
        new_labels[j] = f'a{k}'
        
    # Get parent operators
    s = generator.labels_to_shape(new_labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    parents = [None] + [labels[p.parent] for p in tree[1:]]
    
    # Replace floats with symbols (except exponents)
    if replace_floats:
        param_idx = [j for j, lab in enumerate(labels) if (generator.is_float(lab) and not (parents[j].lower() =='pow')) or (lab.startswith('a') and generator.is_float(lab[1:]))]
        for k, j in enumerate(param_idx):
            labels[j] = f'a{k}'
    fstr = generator.node_to_string(0, tree, labels)
    print(labels)
    res = single_function(
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
            return_params=return_params
    )
    
    if return_params:
        return res[0], res[1], labels, res[2]
    
    return res[0], res[1], labels
    
    
def tree_to_aifeyn(labels, basis_functions, verbose=True):
    """
    Takes a list of labels defining a function and returns the AIFeyn term of
    complexity and the complexity of the function
    
    Args:
        :labels (list): list of strings giving node labels of tree
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are
            binary operators
        :verbose (bool, default=True): Whether to print results (True) or not (False)
    
    Returns:
        :aifeyn (float): the contribution to description length from describing tree
        :complexity (int): the number of nodes in the function
    """

    # Convert the string to a sympy function
    s = generator.labels_to_shape(labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    fstr = generator.node_to_string(0, tree, labels)
    max_param = simplifier.get_max_param([fstr], verbose=verbose)
    
    # Get the functional complexity
    param_list = ['a%i'%j for j in range(max_param)]
    aifeyn = generator.aifeyn_complexity(labels, param_list)
    if verbose:
        print('Function:', aifeyn)

    return aifeyn, len(labels)
    
    
def string_to_aifeyn(fun, basis_functions, maxvar=20, verbose=True,
    replace_floats=False):
    """
    Takes a string defining a function and returns the AIFeyn term of
    complexity and the complexity of the function
    
    Args:
        :fun (str): String representation of the function to be fitted
        :basis_functions (list): list of lists basis functions. basis_functions[0] are
            nullary, basis_functions[1] are unary and basis_functions[2] are
            binary operators
        :maxvar (int, default=20): The maximum number of variables which could appear
            in the function
        :verbose (bool, default=True): Whether to print results (True) or not (False)
        :replace_floats (bool, default=False): whether to replace any numbers found in
            the function with variables to optimise
    
    Returns:
        :aifeyn (float): the contribution to description length from describing tree
        :complexity (int): the number of nodes in the function
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
        elif lab == 'Div':
            new_labels[j] = '/'
            labels[j] = '/'
        elif lab == 'Sub':
            new_labels[j] = '-'
            labels[j] = '-'
        else:
            new_labels[j] = lab.lower()
            labels[j] = lab.lower()
    param_idx = [j for j, lab in enumerate(new_labels) if generator.is_float(lab) or (lab.startswith('a') and generator.is_float(lab[1:]))]
    assert len(param_idx) <= maxvar
    for k, j in enumerate(param_idx):
        new_labels[j] = f'a{k}'
        
    # Get parent operators
    s = generator.labels_to_shape(new_labels, basis_functions)
    success, _, tree = generator.check_tree(s)
    parents = [None] + [labels[p.parent] for p in tree[1:]]
    
    # Replace floats with symbols (except exponents)
    if replace_floats:
        param_idx = [j for j, lab in enumerate(labels) if (generator.is_float(lab) and not (parents[j].lower() =='pow')) or (lab.startswith('a') and generator.is_float(lab[1:]))]
        for k, j in enumerate(param_idx):
            labels[j] = f'a{k}'
    fstr = generator.node_to_string(0, tree, labels)
    print(labels)

    return tree_to_aifeyn(labels, basis_functions, verbose=verbose)
    

