# This optimises in log-space, with separate +ve and -ve branch (except when there are 3 params in which case it does it in linear)
# Uses the weights properly of the function under investigation, not the RAR IF
# Also uses -log(L) as the accuracy metric, not reduced chi2, to prevent the gradient blowing up to formally increase the errors masses

import numpy as np
import sympy
import warnings
import os
import sys
import signal
from mpi4py import MPI
from scipy.optimize import minimize

from filenames import *

sys.path.insert(0, like_dir)
likelihood = __import__(like_file)

name = __name__
import importlib
globals().update(importlib.import_module(sym_file).__dict__)
__name__ = name

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def handler(signum, frame):
    print("A function timed out: ", i, fcn_i, flush=True)
    raise Exception("end of time")

def chi2_fcn_4args(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    return likelihood.negloglike(x,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

def chi2_fcn_3args(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    return likelihood.negloglike(x,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)
    
def chi2_fcn_2args_pp(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    p = [10.**x[0], 10.**x[1]]
    return likelihood.negloglike(p,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

def chi2_fcn_2args_pm(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    p = [10.**x[0], -10.**x[1]]
    return likelihood.negloglike(p,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

def chi2_fcn_2args_mp(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    p = [-10.**x[0], 10.**x[1]]
    return likelihood.negloglike(p,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)
    
def chi2_fcn_2args_mm(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    p = [-10.**x[0], -10.**x[1]]
    return likelihood.negloglike(p,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)
    
def chi2_fcn_1arg_p(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    p = [10.**x[0]]
    return likelihood.negloglike(p,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

def chi2_fcn_1arg_m(x, xvar, yvar, inv_cov, eq_numpy, integrated):
    p = [-10.**x[0]]
    return likelihood.negloglike(p,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)
    
def get_functions(comp, unique=True):

    if unique:
        unifn_file = fn_dir + "/compl_%i/unique_equations_%i.txt"%(comp,comp)
    else:
        unifn_file = fn_dir + "/compl_%i/all_equations_%i.txt"%(comp,comp)
    
    if comp==8:
        sys.setrecursionlimit(2000)
    elif comp==9:
        sys.setrecursionlimit(2500)
    elif comp==10:
        sys.setrecursionlimit(3000)

    if rank == 0:
        for dirname in [out_dir, temp_dir]:
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
    
def load_data():
    if rank == 0:
        print("\nLoading data", flush=True)
        xvar, yvar, inv_cov, _ = likelihood.load_data()
        xvar += 1  # now x = 1 + z
        print("Loaded", flush=True)
    else:
        xvar, yvar, inv_cov = [None] * 3
    xvar = comm.bcast(xvar, root=0)
    yvar = comm.bcast(yvar, root=0)
    inv_cov = comm.bcast(inv_cov, root=0)
    return xvar, yvar, inv_cov
    
    
def optimise_fun(fcn_i, xvar, yvar, inv_cov, tmax, pmin, pmax, try_integration=True):

    Niter_1, Niter_2, Niter_3, Niter_4 = 10, 10, 10, 10
    Nconv_1, Nconv_2, Nconv_3, Nconv_4 = 3, 3, 3, 3

    signal.signal(signal.SIGALRM, handler)
    #signal.alarm(2700)          # 45 mins
    signal.alarm(600)           # 10 mins
    
    params = np.zeros(4)
    
    try:
        fcn_i, eq, integrated = run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
            
        if ("a0" in fcn_i)==False:
            Nparams = 0
            eq_numpy = sympy.lambdify(x, eq, modules=["numpy","sympy"])
            chi2_i = likelihood.negloglike([], eq_numpy, xvar, yvar, inv_cov, integrated=integrated)
            signal.alarm(0)
            return chi2_i, params

        flag_three = False
        if ("a3" in fcn_i) and ("a2" in fcn_i) and ("a1" in fcn_i) and ("a0" in fcn_i):
            Niter = Niter_4
            Nconv = Nconv_4
            flag_three = True           # In this case we don't have any mult_param

            eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, modules=["numpy","sympy"])
            if np.sum(np.isnan(eq_numpy(xvar,1,1,1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,1,1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,-1,1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,1,-1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,-1,1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,1,-1,1)))>0  and np.sum(np.isnan(eq_numpy(xvar,1,-1,-1,1)))>0  and np.sum(np.isnan(eq_numpy(xvar,-1,-1,-1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,1,1,-1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,1,1,-1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,-1,1,-1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,1,-1,-1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,-1,1,-1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,1,-1,-1)))>0  and np.sum(np.isnan(eq_numpy(xvar,1,-1,-1,-1)))>0  and np.sum(np.isnan(eq_numpy(xvar,-1,-1,-1,-1)))>0:
                chi2_i = np.inf            # Don't bother trying to optimise bc this fcn is clearly really bad
                signal.alarm(0)
                return chi2_i, params
                
        elif ("a2" in fcn_i) and ("a1" in fcn_i) and ("a0" in fcn_i):
            Niter = Niter_3
            Nconv = Nconv_3
            flag_three = True           # In this case we don't have any mult_param
            eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, modules=["numpy","sympy"])
            if np.sum(np.isnan(eq_numpy(xvar,1,1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,-1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,1,-1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,-1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,1,-1)))>0  and np.sum(np.isnan(eq_numpy(xvar,1,-1,-1)))>0  and np.sum(np.isnan(eq_numpy(xvar,-1,-1,-1)))>0:
                chi2_i = np.inf            # Don't bother trying to optimise bc this fcn is clearly really bad
                signal.alarm(0)
                return chi2_i, params
                
        elif ("a1" in fcn_i) and ("a0" in fcn_i):
            Niter = Niter_2
            Nconv = Nconv_2
            eq_numpy = sympy.lambdify([x, a0, a1], eq, modules=["numpy","sympy"])
            if np.sum(np.isnan(eq_numpy(xvar,1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,1,-1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1,-1)))>0:
                chi2_i = np.inf
                signal.alarm(0)
                return chi2_i, params

        else:
            Niter = Niter_1
            Nconv = Nconv_1
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy","sympy"])
            if np.sum(np.isnan(eq_numpy(xvar,1)))>0 and np.sum(np.isnan(eq_numpy(xvar,-1)))>0:
                chi2_i = np.inf
                signal.alarm(0)
                return chi2_i, params

        chi2_min = np.inf            # Reset chi2 for this fcn to make sure that every attempt at minimisation improves on this

        mult_arr = np.array([1,1,1,1])
        count_lowest = 0
        inf_count = 0
        exception_count = 0
        
        for j in range(Niter):
            if rank == 0:
                print('\t', j, Niter, flush=True)
            if ("a3" in fcn_i) and ("a2" in fcn_i) and ("a1" in fcn_i) and ("a0" in fcn_i):
                inpt = [np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax)]
                res = minimize(chi2_fcn_4args, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS", options={'maxiter': 7000})    # Default=3000
            elif ("a2" in fcn_i) and ("a1" in fcn_i) and ("a0" in fcn_i):
                inpt = [np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax)]      # Larger range bc linear search here
                res = minimize(chi2_fcn_3args, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS", options={'maxiter': 5000})    # Default=3000
            elif ("a1" in fcn_i) and ("a0" in fcn_i):
                inpt = [np.random.uniform(pmin,pmax), np.random.uniform(pmin,pmax)]           # These are now in log-space, so this is 1e-1 -- 1e1; was -10:10
                res_pp = minimize(chi2_fcn_2args_pp, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS")
                res_mp = minimize(chi2_fcn_2args_mp, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS")
                res_pm = minimize(chi2_fcn_2args_pm, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS")
                res_mm = minimize(chi2_fcn_2args_mm, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS")
                
                choose = np.argmin([res_pp['fun'], res_mp['fun'], res_pm['fun'], res_mm['fun']])
                if choose==0:
                    res = res_pp
                    mult_arr = np.array([1,1,1,1])
                elif choose==1:
                    res = res_mp
                    mult_arr = np.array([-1,1,1,1])
                elif choose==2:
                    res = res_pm
                    mult_arr = np.array([1,-1,1,1])
                elif choose==3:
                    res = res_mm
                    mult_arr = np.array([-1,-1,1,1])
                else:
                    print("Some ambiguity in choose", eq, flush=True)
                    res = res_pp
                    
            else:

                inpt = np.random.uniform(pmin,pmax)
                res_p = minimize(chi2_fcn_1arg_p, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS")
                res_m = minimize(chi2_fcn_1arg_m, inpt, args=(xvar, yvar, inv_cov, eq_numpy, integrated), method="BFGS")

                if res_p['fun']<res_m['fun']:
                    res = res_p
                    mult_arr = np.array([1,1,1,1])
                elif res_p['fun']>res_m['fun']:
                    res = res_m
                    mult_arr = np.array([-1,1,1,1])
                else:
                    res = res_p     # Arbitrarily decide to take the +ve in cases where they're the same, but don't update mult_arr

            if np.isinf(res['fun']):
                inf_count += 1

            if inf_count==50 and np.isinf(chi2_min):        # If you get inf all of the first 50 times, consider this a failure
                signal.alarm(0)
                break

            if res['fun']-chi2_min < -2.:     # negloglike has got better by 2, so reset the count to 0
                count_lowest=0

            if abs(res['fun']-chi2_min) < 0.5:  # If we've ended up within 0.5 of the best value we say this run has converged to that
                count_lowest += 1

            if res['fun'] < chi2_min:
                best = res
                mult_arr_best = mult_arr        # If you're looking at 3 params this is just [1,1,1]
                chi2_min = res['fun']

            # We've converged the required number of times, so we call this a success
            if count_lowest==Nconv:
                signal.alarm(0)
                break
                
            
        if chi2_min < 1.e100:            # This means that some optimisation has happened, so we have something to print for params
            if flag_three:
                params = np.pad(np.array(best.x), (0, 4-len(best.x)))       # Params go in directly
            else:
                params = np.pad(10.**np.array(best.x), (0, 4-len(best.x))) * mult_arr_best      # Params put in linear space and sign added back in
                        
        chi2_i = chi2_min       # This is after all the iterations, so it's the best we have; reduced chi2
    
    except Exception:
        signal.alarm(0)
        return np.nan, params

    signal.alarm(0)

    return chi2_i, params
    
    
def main(comp, tmax=5, pmin=0, pmax=3, data=None):
    
    fcn_list_proc, _, _ = get_functions(comp)

    if data is None:
        xvar, yvar, inv_cov = load_data()
    else:
        xvar, yvar, inv_cov = data

    chi2 = np.zeros(len(fcn_list_proc))     # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), 4])
    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0:
            print(i, len(fcn_list_proc), flush=True)
        chi2[i], params[i,:] = optimise_fun(fcn_list_proc[i], xvar, yvar, inv_cov, tmax, pmin, pmax)

    out_arr = np.transpose(np.vstack([chi2, params[:,0], params[:,1], params[:,2], params[:,3]]))

    # Save the data for this proc in Partial
    np.savetxt(temp_dir + '/chi2_comp'+str(comp)+'weights_'+str(rank)+'.dat', out_arr, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + temp_dir + '/ -name "chi2_comp'+str(comp)+'weights_*.dat" | sort -V` > ' + out_dir + '/negloglike_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + temp_dir + '/chi2_comp'+str(comp)+'weights_*.dat'
        os.system(string)

    return
        
if __name__ == "__main__":
    comp = int(sys.argv[1])
    main(comp)
