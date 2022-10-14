import numpy as np
import math
from scipy.optimize import minimize
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import itertools
import numdifftools as nd

import test_all
from filenames import *

sys.path.insert(0, like_dir)
likelihood = __import__(like_file)

name = __name__
import importlib
globals().update(importlib.import_module(sym_file).__dict__)
__name__ = name

warnings.filterwarnings("ignore")

use_relative_dx = True              # CHANGE

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
    
def load_loglike(comp, data_start, data_end, split=True):
    negloglike, param1, param2, param3, param4 = np.genfromtxt(out_dir + "/negloglike_comp"+str(comp)+".dat", unpack=True)
    negloglike = np.atleast_1d(negloglike)
    param1 = np.atleast_1d(param1)
    param2 = np.atleast_1d(param2)
    param3 = np.atleast_1d(param3)
    param4 = np.atleast_1d(param4)
    if split:
        negloglike = negloglike[data_start:data_end]               # Assuming same order of fcn and chi2 files
        param1 = param1[data_start:data_end]
        param2 = param2[data_start:data_end]
        param3 = param3[data_start:data_end]
        param4 = param4[data_start:data_end]
    return negloglike, param1, param2, param3, param4


def convert_params(fcn_i, eq, integrated, theta_ML, xvar, yvar, inv_cov, negloglike):

    def f4(x):
        return likelihood.negloglike(x,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

    def f3(x):
        return likelihood.negloglike(x,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

    def f2(x):
        return likelihood.negloglike(x,eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

    def f1(x):
        return likelihood.negloglike([x],eq_numpy, xvar, yvar, inv_cov, integrated=integrated)

    params = np.zeros(4)
    deriv = np.zeros(10)
    
    d_list = [1.e-5, 10.**(-5.5), 10.**(-4.5), 1.e-6, 1.e-4, 10.**(-6.5), 10.**(-3.5), 1.e-7, 1.e-3, 10.**(-7.5), 10.**(-2.5), 1.e-8, 1.e-2, 1.e-9, 1.e-10, 1.e-11]
    method_list = ["central", "forward", "backward"]

    if ("a3" in fcn_i) and ("a2" in fcn_i) and ("a1" in fcn_i) and ("a0" in fcn_i):
        k=4
    
        eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, modules=["numpy","sympy"])
    
        Hfun = nd.Hessian(f4)
        Hmat = Hfun(theta_ML)
    
        Fisher_diag = np.array([Hmat[0,0], Hmat[1,1], Hmat[2,2], Hmat[3,3]])                # 2nd derivatives of -log(L) wrt params
    
        deriv[:] = np.array([Hmat[0,0], Hmat[0,1], Hmat[0,2], Hmat[0,3], Hmat[1,1], Hmat[1,2], Hmat[1,3], Hmat[2,2], Hmat[2,3], Hmat[3,3]])
    
        Delta = np.sqrt(12./Fisher_diag)
        Nsteps = abs(np.array(theta_ML))/Delta
    
        if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0) or (np.sum(np.isinf(Fisher_diag)) > 0) or (np.sum(Nsteps<1) > 0):
            for d2, meth in itertools.product(d_list, method_list):
                if use_relative_dx:
                    Hfun = nd.Hessian(f4, step = np.abs(d2*theta_ML)+1.e-15, method=meth)          # Step is array, one for each param
                else:
                    Hfun = nd.Hessian(f4, step = d2, method=meth)
                Hmat = Hfun(theta_ML)
                Fisher_diag_tmp = np.array([Hmat[0,0], Hmat[1,1], Hmat[2,2], Hmat[3,3]])
                Delta_tmp = np.sqrt(12./Fisher_diag_tmp)
                Nsteps_tmp = abs(np.array(theta_ML))/Delta_tmp
                if (np.sum(Fisher_diag_tmp <= 0.) <= 0) and (np.sum(np.isnan(Fisher_diag_tmp)) <= 0) and (np.sum(np.isinf(Fisher_diag)) <= 0) and (np.sum(Nsteps_tmp<1)<=0):
                    if negloglike<0.:
                        print("Succeeded at rectifying Fisher:", fcn_i, d2, meth, Fisher_diag_tmp, theta_ML, Delta_tmp, Nsteps_tmp, flush=True)
                    Fisher_diag = Fisher_diag_tmp
                    Delta, Nsteps = Delta_tmp, Nsteps_tmp
                    deriv[:] = np.array([Hmat[0,0], Hmat[0,1], Hmat[0,2], Hmat[0,3], Hmat[1,1], Hmat[1,2], Hmat[1,3], Hmat[2,2], Hmat[2,3], Hmat[3,3]])
                    break

        negloglike = f4(params[:k])
    
    elif ("a2" in fcn_i) and ("a1" in fcn_i) and ("a0" in fcn_i):
        k=3
    
        theta_ML = theta_ML[:-1]            # Only keep as many params as we have for this fcn
    
        eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, modules=["numpy","sympy"])
    
        Hfun = nd.Hessian(f3)
        Hmat = Hfun(theta_ML)
    
        Fisher_diag = np.array([Hmat[0,0], Hmat[1,1], Hmat[2,2]])
    
        deriv[:] = np.array([Hmat[0,0], Hmat[0,1], Hmat[0,2], np.nan, Hmat[1,1], Hmat[1,2], np.nan, Hmat[2,2], np.nan, np.nan])
    
        Delta = np.sqrt(12./Fisher_diag)
        Nsteps = abs(np.array(theta_ML))/Delta
    
        if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0) or (np.sum(np.isinf(Fisher_diag)) > 0) or (np.sum(Nsteps<1) > 0):
            for d2, meth in itertools.product(d_list, method_list):
                if use_relative_dx:
                    Hfun = nd.Hessian(f3, step = np.abs(d2*theta_ML)+1.e-15, method=meth)
                else:
                    Hfun = nd.Hessian(f3, step = d2, method=meth)
                Hmat = Hfun(theta_ML)
                Fisher_diag_tmp = np.array([Hmat[0,0], Hmat[1,1], Hmat[2,2]])
                Delta_tmp = np.sqrt(12./Fisher_diag_tmp)
                Nsteps_tmp = abs(np.array(theta_ML))/Delta_tmp
                if (np.sum(Fisher_diag_tmp <= 0.) <= 0) and (np.sum(np.isnan(Fisher_diag_tmp)) <= 0) and (np.sum(np.isinf(Fisher_diag)) <= 0) and (np.sum(Nsteps_tmp<1)<=0):
                    if negloglike<0.:
                        print("Succeeded at rectifying Fisher:", fcn_i, d2, meth, Fisher_diag_tmp, theta_ML, Delta_tmp, Nsteps_tmp, flush=True)
                    Fisher_diag = Fisher_diag_tmp
                    Delta, Nsteps = Delta_tmp, Nsteps_tmp
                    deriv[:] = np.array([Hmat[0,0], Hmat[0,1], Hmat[0,2], np.nan, Hmat[1,1], Hmat[1,2], np.nan, Hmat[2,2], np.nan, np.nan])
                    break

        negloglike = f3(params[:k])
    
    elif ("a1" in fcn_i) and ("a0" in fcn_i):
        k=2
    
        theta_ML = theta_ML[:-2]
    
        eq_numpy = sympy.lambdify([x, a0, a1], eq, modules=["numpy","sympy"])
        
        Hfun = nd.Hessian(f2)                       # Calculate the fcn in general
        Hmat = Hfun(theta_ML)       # Evaluate it at the ML point
    
        Fisher_diag = np.array([Hmat[0,0], Hmat[1,1]])
    
        deriv[:] = np.array([Hmat[0,0], Hmat[0,1], np.nan, np.nan, Hmat[1,1], np.nan, np.nan, np.nan, np.nan, np.nan])
        
        Delta = np.sqrt(12./Fisher_diag)
        Nsteps = abs(np.array(theta_ML))/Delta

        if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0) or (np.sum(np.isinf(Fisher_diag)) > 0) or (np.sum(Nsteps<1) > 0):
            for d2, meth in itertools.product(d_list, method_list):
                if use_relative_dx:
                    Hfun = nd.Hessian(f2, step = np.abs(d2*theta_ML)+1.e-15, method=meth)
                else:
                    Hfun = nd.Hessian(f2, step = d2, method=meth)
                Hmat = Hfun(theta_ML)
                Fisher_diag_tmp = np.array([Hmat[0,0], Hmat[1,1]])
                Delta_tmp = np.sqrt(12./Fisher_diag_tmp)
                Nsteps_tmp = abs(np.array(theta_ML))/Delta_tmp
                if (np.sum(Fisher_diag_tmp <= 0.) <= 0) and (np.sum(np.isnan(Fisher_diag_tmp)) <= 0) and (np.sum(np.isinf(Fisher_diag)) <= 0) and (np.sum(Nsteps_tmp<1)<=0):
                    if negloglike<0.:
                        print("Succeeded at rectifying Fisher:", fcn_i, d2, meth, Fisher_diag_tmp, theta_ML, Delta_tmp, Nsteps_tmp, flush=True)
                    Fisher_diag = Fisher_diag_tmp
                    Delta, Nsteps = Delta_tmp, Nsteps_tmp
                    deriv[:] = np.array([Hmat[0,0], Hmat[0,1], np.nan, np.nan, Hmat[1,1], np.nan, np.nan, np.nan, np.nan, np.nan])
                    break

        negloglike = f2(params[:k])

    elif ("a0" in fcn_i):
        k=1
        
        theta_ML = theta_ML[:-3]            # Only keep as many params as we have for this fcn
        
        try:
            eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy","sympy"])
        except Exception:
            print("BAD:", fcn_i)
            Fisher_diag = np.nan
            deriv[:] = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            return params, negloglike, deriv, codelen
        
        Hfun = nd.Hessian(f1)
        Hmat = Hfun(theta_ML)
    
        Fisher_diag = Hmat[0,0]
    
        deriv[:] = np.array([Hmat[0,0], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    
        Delta = np.sqrt(12./Fisher_diag)
        Nsteps = abs(np.array(theta_ML))/Delta
    
        if (np.sum(Fisher_diag <= 0.) > 0) or (np.sum(np.isnan(Fisher_diag)) > 0) or (np.sum(np.isinf(Fisher_diag)) > 0) or (np.sum(Nsteps<1) > 0):
            for d2, meth in itertools.product(d_list, method_list):
                if use_relative_dx:
                    Hfun = nd.Hessian(f1, step = np.abs(d2*theta_ML)+1.e-15, method=meth)
                else:
                    Hfun = nd.Hessian(f1, step = d2, method=meth)
                Hmat = Hfun(theta_ML)
                Fisher_diag_tmp = Hmat[0,0]
                Delta_tmp = np.sqrt(12./Fisher_diag_tmp)
                Nsteps_tmp = abs(np.array(theta_ML))/Delta_tmp
                if (np.sum(Fisher_diag_tmp <= 0.) <= 0) and (np.sum(np.isnan(Fisher_diag_tmp)) <= 0) and (np.sum(np.isinf(Fisher_diag)) <= 0) and (np.sum(Nsteps_tmp<1)<=0):
                    if negloglike<0.:
                        print("Succeeded at rectifying Fisher:", fcn_i, d2, meth, Fisher_diag_tmp, theta_ML, Delta_tmp, Nsteps_tmp, flush=True)
                    Fisher_diag = Fisher_diag_tmp
                    Delta, Nsteps = Delta_tmp, Nsteps_tmp
                    deriv[:] = np.array([Hmat[0,0], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
                    break

        negloglike = f1(params[:k])
    
    else:                   # No params
        codelen = 0
        return params, negloglike, deriv, codelen

    # Must indicate a bad fcn, so just need to make sure it doesn't have a good -log(L)
    if (np.sum(Fisher_diag <= 0.) > 0.) or (np.sum(np.isnan(Fisher_diag)) > 0):
        if negloglike<0.:
            print("ATTENTION, a relatively good function has Fisher element negative, zero or nan: ", fcn_i, negloglike, Fisher_diag, flush=True)
        codelen = np.nan
        return params, negloglike, deriv, codelen

    if np.sum(Nsteps<1)>0:         # Possibly should reevaluate -log(L) with the param(s) set to 0, but doesn't matter unless the fcn is a very good one
        if negloglike<-1500.:
            print("ATTENTION, a relatively good function is having a parameter set to 0: ", fcn_i, negloglike, theta_ML, Delta, Nsteps, Fisher_diag, flush=True)
        
        theta_ML[Nsteps<1] = 0.         # Set any parameter to 0 that doesn't have at least one precision step, and recompute -log(L).
        
        negloglike_orig = np.copy(negloglike)
        
        if k==1:
            negloglike = f1(theta_ML)                    # Modified here but if this doesn't happen they stay the same
        elif k==2:
            negloglike = f2(theta_ML)       # All params still here, just some of them might be 0
        elif k==3:
            negloglike = f3(theta_ML)
        elif k==4:
            negloglike = f4(theta_ML)
            
        if negloglike_orig<-1500.:          # Should be exactly the same condition as the above
            print("negloglikes:", negloglike_orig, negloglike, flush=True)

        k -= np.sum(Nsteps<1)       # For the codelen, we effectively don't have the parameter that had Nsteps<1
            
        if k<0:
            print("This shouldn't have happened", flush=True)
            quit()
        elif k==0:                  # If we have no parameters left then the parameter codelength is 0 so we can move on
            codelen = 0
            return params, negloglike, deriv, codelen
        
        Fisher_diag = Fisher_diag[Nsteps>=1]     # Only consider these parameters in the codelen
        theta_ML = theta_ML[Nsteps>=1]
    
    codelen = -k/2.*math.log(3.) + np.sum( 0.5*np.log(Fisher_diag) + np.log(abs(np.array(theta_ML))) )

    params[:] = np.pad(theta_ML, (0, 4-len(theta_ML)))            # New params after the setting to 0, padded to length 4 as always
    
    if (np.isinf(codelen) or codelen>200.) and negloglike<-1500.:
        print("ATTENTION, a relative good function has a very big codelength:", fcn_i, k, Fisher_diag, theta_ML, Delta, Nsteps, negloglike, codelen, flush=True)

    return params, negloglike, deriv, codelen

    
def main(comp, tmax=5, data=None):

    if comp==8:
        sys.setrecursionlimit(2000)
    elif comp==9:
        sys.setrecursionlimit(2500)
    elif comp==10:
        sys.setrecursionlimit(3000)

    if data is None:
        xvar, yvar, inv_cov = test_all.load_data()
    else:
        xvar, yvar, inv_cov = data

    fcn_list_proc, data_start, data_end = test_all.get_functions(comp)
    negloglike, param1_proc, param2_proc, param3_proc, param4_proc = load_loglike(comp, data_start, data_end)

    codelen = np.zeros(len(fcn_list_proc))          # This is now only for this proc
    params = np.zeros([len(fcn_list_proc), 4])
    deriv = np.zeros([len(fcn_list_proc), 10])

    for i in range(len(fcn_list_proc)):           # Consider all possible complexities
        if rank == 0:
            print(i, len(fcn_list_proc), flush=True)

        if np.isnan(negloglike[i]) or np.isinf(negloglike[i]):
            codelen[i]=np.nan
            continue
            
        fcn_i = fcn_list_proc[i].replace('\n', '')
        fcn_i = fcn_list_proc[i].replace('\'', '')
        
        
        theta_ML = np.array([param1_proc[i], param2_proc[i], param3_proc[i], param4_proc[i]])

        try:
           fcn_i, eq, integrated = run_sympify(fcn_i, tmax=tmax)
        except Exception as ex:
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            print("Failed sympification", fcn_i, flush=True)
            print(template.format(type(ex).__name__, ex.args), flush=True)
            chi2[i] = np.inf
            continue


        try:
            params[i,:], negloglike[i], deriv[i,:], codelen[i] = convert_params(fcn_i, eq, integrated, theta_ML, xvar, yvar, inv_cov, negloglike[i])
        except:
            params[i,:] = 0.
            deriv[i,:] = 0.
            codelen[i] = 0
        
    out_arr = np.transpose(np.vstack([codelen, negloglike, params[:,0], params[:,1], params[:,2], params[:,3]]))

    out_arr_deriv = np.transpose(np.vstack([deriv[:,0], deriv[:,1], deriv[:,2], deriv[:,3], deriv[:,4], deriv[:,5], deriv[:,6], deriv[:,7], deriv[:,8], deriv[:,9]]))

    np.savetxt(temp_dir + '/codelen_deriv_'+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')
    np.savetxt(temp_dir + '/derivs_'+str(comp)+'_'+str(rank)+'.dat', out_arr_deriv, fmt='%.7e')

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + temp_dir + '/ -name "codelen_deriv_'+str(comp)+'_*.dat" | sort -V` > ' + out_dir + '/codelen_comp'+str(comp)+'_deriv.dat'
        os.system(string)
        string = 'rm ' + temp_dir + '/codelen_deriv_'+str(comp)+'_*.dat'
        os.system(string)

        string = 'cat `find ' + temp_dir + '/ -name "derivs_'+str(comp)+'_*.dat" | sort -V` > ' + out_dir + '/derivs_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + temp_dir + '/derivs_'+str(comp)+'_*.dat'
        os.system(string)

if __name__ == "__main__":
    comp = int(sys.argv[1])
    main(comp)
