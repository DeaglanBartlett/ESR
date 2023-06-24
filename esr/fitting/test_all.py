import numpy as np
import sympy
import warnings
import os
import sys
from mpi4py import MPI
from scipy.optimize import minimize
import itertools

from esr.fitting.sympy_symbols import *
import esr.generation.simplifier as simplifier

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def chi2_fcn(x, likelihood, eq_numpy, integrated, signs):
    """Compute chi2 for a function
    
    Args:
        :x (list): parameters to use for function
        :likelihood (fitting.likelihood object): object containing data and likelihood function
        :eq_numpy (numpy function): function to pass to likelihood object to make prediction of y(x)
        :integrated (bool): whether eq_numpy has already been integrated
        :signs (list): each entry specifies whether than parameter should be optimised logarithmically. If None, then do nothing, if '+' then optimise 10**x[i] and if '-' then optimise -10**x[i]
        
    Returns:
        :negloglike (float): - log(likelihood) for this function and parameters
    
    """
    if signs is None:
        p = x
    else:
        p = [None] * len(signs)
        for i in range(len(signs)):
            if signs[i] == None:
                p[i] = x[i]
            elif signs[i] == '+':
                p[i] = 10 ** x[i]
            elif signs[i] == '-':
                p[i] = - 10 ** x[i]
            else:
                raise ValueError
    return likelihood.negloglike(p,eq_numpy, integrated=integrated)
    
    
def get_functions(comp, likelihood, unique=True):
    """Load all functions for a given complexity to use and distribute among ranks
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, functions to convert SR expressions to variable of data and file path
        :unique (bool, default=True): whether to load just the unique functions (True) or all functions (False)
        
    Returns:
        :fcn_list (list): list of strings representing functions to be used by given rank
        :data_start (int): first index of function used by rank
        :data_end (int): last index of function used by rank
        
    """

    if unique:
        unifn_file = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt"%(comp,comp)
    else:
        unifn_file = likelihood.fn_dir + "/compl_%i/all_equations_%i.txt"%(comp,comp)
    
    if comp>=8:
        sys.setrecursionlimit(2000 + 500 * (comp - 8))

    if rank == 0:
        for dirname in [likelihood.base_out_dir, likelihood.out_dir, likelihood.temp_dir]:
            if not os.path.isdir(dirname):
                print('Making dir:', dirname)
                os.mkdir(dirname)
    comm.Barrier()

    if rank==0:
        print("Number of cores:", size, flush=True)

    with open(unifn_file, "r") as f:
        fcn_list = f.readlines()

    nLs = int(np.ceil(len(fcn_list) / float(size)))       # Number of lines per file for given thread

    while nLs*(size-1) > len(fcn_list):
        if rank==0:
            print("Correcting for many cores.", flush=True)
        nLs -= 1

    if rank==0:
        print("Total number of functions: ", len(fcn_list), flush=True)
        print("Number of test points per proc: ", nLs, flush=True)

    data_start = rank*nLs
    data_end = (rank+1)*nLs

    if rank==size-1:
        data_end = len(fcn_list)
    
    return fcn_list[data_start:data_end], data_start, data_end
    
    
def optimise_fun(fcn_i, likelihood, tmax, pmin, pmax, try_integration=False, log_opt=False, max_param=4):
    """Optimise the parameters of a function to fit data
    
    Args:
        :fcn_i (str): string representing function we wish to fit to data
        :likelihood (fitting.likelihood object): object containing data and likelihood function
        :tmax (float): maximum time in seconds to run any one part of simplification procedure for a given function
        :pmin (float): minimum value for each parameter to consider when generating initial guess
        :pmax (float): maximum value for each parameter to consider when generating initial guess
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        :log_opt (bool, default=False): whether to optimise 1 and 2 parameter cases in log space
        :max_param (int, default=4): The maximum number of parameters considered. This sets the shapes of arrays used.
        
    Returns:
        :chi2_i (float): the minimum value of -log(likelihood) (corresponding to the maximum likelihood)
        :params (list): the maximum likelihood values of the parameters
    
    """

    xvar, yvar = likelihood.xvar, likelihood.yvar

    Niter = 30
    Nconv = 5
    
    nparam = simplifier.get_max_param([fcn_i], verbose=False)
    params = np.zeros(max_param)
    
    try:
        fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
            
        if ("a0" in fcn_i)==False:
            Nparams = 0
            eq_numpy = sympy.lambdify(x, eq, modules=["numpy"])
            chi2_i = likelihood.negloglike([], eq_numpy, integrated=integrated)
            return chi2_i, params

        flag_three = False

        mult_arr = np.ones(max_param)
        count_lowest = 0
        inf_count = 0
        exception_count = 0
        
        if nparam > 1:
            all_a = ' '.join([f'a{i}' for i in range(nparam)])
            all_a = list(sympy.symbols(all_a, real=True))
            eq_numpy = sympy.lambdify([x] + all_a, eq, modules=["numpy"])
        else:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
    
        bad_fun = True
        for p in itertools.product([1, -1], repeat=nparam):
            if not (np.sum(np.isnan(eq_numpy(xvar,*p)))>0):
                bad_fun = False
                break
        
        if bad_fun:
            # Don't bother trying to optimise bc this fcn is clearly really bad
            chi2_i = np.inf
            return chi2_i, params
            
        # Reset chi2
        chi2_min = np.inf
            
        if nparam > 2:
            flag_three = True
        
        for j in range(Niter):
        
            if nparam > 2:
                inpt = [np.random.uniform(pmin,pmax) for _ in range(nparam)]
                res = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, None), method="BFGS", options={'maxiter': 7000})    # Default=3000
            elif nparam == 2:
                if log_opt:
                    inpt = [np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax)]           # These are now in log-space, so this is 1e-1 -- 1e1; was -10:10
                    res_pp = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, ['+','+']), method="BFGS")
                    res_mp = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, ['-','+']), method="BFGS")
                    res_pm = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, ['+','-']), method="BFGS")
                    res_mm = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, ['-','-']), method="BFGS")
                
                    choose = np.argmin([res_pp['fun'], res_mp['fun'], res_pm['fun'], res_mm['fun']])
                    mult_arr = np.one(max_param)
                    if choose==0:
                        res = res_pp
                    elif choose==1:
                        res = res_mp
                        mult_arr[0] = -1
                    elif choose==2:
                        res = res_pm
                        mult_arr[1] = -1
                    elif choose==3:
                        res = res_mm
                        mult_arr[0] = -1
                        mult_arr[1] = -1
                    else:
                        print("Some ambiguity in choose", eq, flush=True)
                        res = res_pp
                else:
                    flag_three = True
                    inpt = [np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax)]
                    res = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, None), method="BFGS", options={'maxiter': 5000})    # Default=3000
                    
            else:
                if log_opt:
                    inpt = np.random.uniform(pmin,pmax)
                    res_p = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, ['+']), method="BFGS")
                    res_m = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, ['-']), method="BFGS")

                    mult_arr = np.one(max_param)
                    if res_p['fun']<res_m['fun']:
                        res = res_p
                    elif res_p['fun']>res_m['fun']:
                        res = res_m
                        mult_arr[0] = -1
                    else:
                        # Both give same. Choose positive (arbitrarily)
                        res = res_p
                else:
                    flag_three = True
                    inpt = np.random.uniform(pmin,pmax)
                    res = minimize(chi2_fcn, inpt, args=(likelihood, eq_numpy, integrated, None), method="BFGS", options={'maxiter': 5000})

            if np.isinf(res['fun']):
                inf_count += 1

            # Failure if first 50 all give inf
            if inf_count==50 and np.isinf(chi2_min):
                break

            # Reset count if log-like improves by 2
            if res['fun']-chi2_min < -2.:
                count_lowest=0

            # If within 0.5 of lowest, say converged to that value
            if abs(res['fun']-chi2_min) < 0.5:
                count_lowest += 1

            if res['fun'] < chi2_min:
                best = res
                mult_arr_best = mult_arr
                chi2_min = res['fun']

            # Converged the required number of times, so a success
            if count_lowest==Nconv:
                break

        if chi2_min < 1.e100:
            # Optimisation happened. Print something
            if flag_three:
                params = np.pad(np.array(best.x), (0, max_param-len(best.x)))
            else:
                # Params put in linear space and sign added back in
                params = np.pad(10.**np.array(best.x), (0, max_param-len(best.x))) * mult_arr_best
                 
        # This is after all the iterations, so it's the best we have; reduced chi2
        chi2_i = chi2_min

    except NameError:
        print(NameError)
        # Occurs if function produced not implemented in numpy
        raise NameError

    except simplifier.TimeoutException:
        print('TIMED OUT:', fcn_i, flush=True)
        try:
            if chi2_min < 1.e100:
                if flag_three:
                    params = np.pad(np.array(best.x), (0, max_param-len(best.x)))
                else:
                    params = np.pad(10.**np.array(best.x), (0, max_param-len(best.x))) * mult_arr_best
                chi2_i = chi2_min
            else:
                chi2_i = np.nan
                params[:] = 0.
        except:
            chi2_i = np.nan
            params[:] = 0.

    except Exception as e:
        print(e)
        return np.nan, params

    return chi2_i, params
    
    
def main(comp, likelihood, tmax=5, pmin=0, pmax=3, try_integration=False, log_opt=False):
    """Optimise all functions for a given complexity and save results to file.
    
    This can optimise in log-space, with separate +ve and -ve branch (except when there are >=3 params in which case it does it in linear)
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :tmax (float, default=5.): maximum time in seconds to run any one part of simplification procedure for a given function
        :pmin (float, default=0.): minimum value for each parameter to considered when generating initial guess
        :pmax (float, default=3.): maximum value for each parameter to considered when generating initial guess
        :try_integration (bool, default=False): when likelihood requires integral, whether to try to analytically integrate (True) or just numerically integrate (False)
        
    Returns:
        None
    
    """

    fcn_list_proc, _, _ = get_functions(comp, likelihood)

    chi2 = np.zeros(len(fcn_list_proc))     # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), 4])
    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0:
            print(rank, i, len(fcn_list_proc), flush=True)
        try:
            with simplifier.time_limit(100):
                try:
                    chi2[i], params[i,:] = optimise_fun(fcn_list_proc[i], 
                                                    likelihood, 
                                                    tmax, 
                                                    pmin, 
                                                    pmax, 
                                                    try_integration=try_integration,
                                                    log_opt=log_opt)
                except NameError:
                    if try_integration:
                        chi2[i], params[i,:] = optimise_fun(fcn_list_proc[i], 
                                                    likelihood, 
                                                    tmax, 
                                                    pmin, 
                                                    pmax, 
                                                    try_integration=False,
                                                    log_opt=log_opt)
                    else:
                        raise NameError
        except:
            chi2[i] = np.nan
            params[i,:] = 0.

    out_arr = np.transpose(np.vstack([chi2, params[:,0], params[:,1], params[:,2], params[:,3]]))

    # Save the data for this proc in Partial
    np.savetxt(likelihood.temp_dir + '/chi2_comp'+str(comp)+'weights_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    
    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "chi2_comp'+str(comp)+'weights_*.dat" | sort -V` > ' + likelihood.out_dir + '/negloglike_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/chi2_comp'+str(comp)+'weights_*.dat'
        os.system(string)

    return

