import numpy as np
import math
import sympy
from mpi4py import MPI
import warnings
import os
import sys
import numdifftools as nd

import test_all
import test_all_Fisher
from sympy_symbols import *

sys.path.insert(0, '../')
import simplifier

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(comp, likelihood, tmax=5, try_integration=False):

    def f4(x):
        return likelihood.negloglike(x,eq_numpy, integrated=integrated)

    def f3(x):
        return likelihood.negloglike(x,eq_numpy, integrated=integrated)

    def f2(x):
        return likelihood.negloglike(x,eq_numpy, integrated=integrated)

    def f1(x):
        return likelihood.negloglike([x],eq_numpy, integrated=integrated)
        
    invsubs_file = likelihood.fn_dir + "/compl_%i/inv_subs_%i.txt"%(comp,comp)
    match_file = likelihood.fn_dir + "/compl_%i/matches_%i.txt"%(comp,comp)
    
    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood, unique=False)
    negloglike, param1, param2, param3, param4 = test_all_Fisher.load_loglike(comp, likelihood, data_start, data_end, split=False)

    max_param = 4

    all_inv_subs_proc = simplifier.load_subs(invsubs_file, max_param)[data_start:data_end]
    matches_proc = np.atleast_1d(np.loadtxt(match_file).astype(int))[data_start:data_end]

    all_fish = np.loadtxt(likelihood.out_dir + '/derivs_comp'+str(comp)+'.dat')   # 2D array of shape (# unique fcns, 10)
    all_fish = np.atleast_2d(all_fish)

    codelen = np.zeros(len(fcn_list_proc))              # Both of these are also just for this proc
    negloglike_all = np.zeros(len(fcn_list_proc))
    index_arr = np.zeros(len(fcn_list_proc))
    params = np.zeros([len(fcn_list_proc), 4])

    for i in range(len(fcn_list_proc)):                 # The part of all eqs analysed by this proc
        if i%1000==0 and rank==0:
            print(i)

        fcn_i = fcn_list_proc[i].replace('\'', '')
        
        nparams = simplifier.count_params([fcn_i], max_param)[0]

        index = matches_proc[i]          # Index in total unique eqs file, common to all procs

        index_arr[i] = index

        negloglike_all[i] = negloglike[index]           # Assign the likelihood of this variant to the that of the unique eq

        if np.isnan(negloglike[index]) or np.isinf(negloglike[index]):          # Element of the unique eqs file, common to all procs
            codelen[i] = np.nan
            continue

        if nparams==0:
            continue
        elif nparams==1:
            k=1
            measured = [param1[index]]
        elif nparams==2:
            k=2
            measured = [param1[index], param2[index]]
        elif nparams==3:
            k=3
            measured = [param1[index], param2[index], param3[index]]
        elif nparams==4:
            k=4
            measured = [param1[index], param2[index], param3[index], param4[index]]
        else:
            print("Bad number of params")
            quit()
        
        fish_measured = all_fish[index,:]               # Access from the unique eqs all_fish array, common to all procs

        try:
            p, fish = simplifier.convert_params(measured, fish_measured, all_inv_subs_proc[i])
        except Exception as ex:
            codelen[i] = np.inf
            continue
        
        if np.sum(fish<=0)>0:
            codelen[i] = np.inf
            continue
        
        Delta = np.zeros(fish.shape)
        m = (fish != 0)
        Delta[m] = np.atleast_1d(np.sqrt(12./fish[m]))
        Delta[~m] = np.inf
        Nsteps = np.atleast_1d(np.abs(np.array(p)))
        m = (Delta != 0)
        Nsteps[m] /= Delta[m]
        Nsteps[~m] = np.nan
        
        if np.sum(Nsteps<1)>0:         # should reevaluate -log(L) with the param(s) set to 0, but doesn't matter unless the fcn is a very good one
            
            try:
                p[Nsteps<1] = 0.         # Set any parameter to 0 that doesn't have at least one precision step, and recompute -log(L).
            except (IndexError, TypeError):
                p=0.
            
            try:            # It's possible that after putting params to 0 the likelihood is botched, in which case give it nan
                fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=try_integration)
                if k==1:
                    eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
                    negloglike_all[i] = f1(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
                elif k==2:
                    eq_numpy = sympy.lambdify([x, a0, a1], eq, modules=["numpy"])
                    negloglike_all[i] = f2(p)       # All params still here, just some of them might be 0
                elif k==3:
                    eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, modules=["numpy"])
                    negloglike_all[i] = f3(p)
                elif k==4:
                    eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, modules=["numpy"])
                    negloglike_all[i] = f4(p)

            except NameError:
                if try_integration:
                    fcn_i, eq, integrated = likelihood.run_sympify(fcn_i, tmax=tmax, try_integration=False)
                    if k==1:
                        eq_numpy = sympy.lambdify([x, a0], eq, modules=["numpy"])
                        negloglike_all[i] = f1(p)               # Modified here for this variant, but if this doesn't happen it stays the same as the unique eq
                    elif k==2:
                        eq_numpy = sympy.lambdify([x, a0, a1], eq, modules=["numpy"])
                        negloglike_all[i] = f2(p)       # All params still here, just some of them might be 0
                    elif k==3:
                        eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, modules=["numpy"])
                        negloglike_all[i] = f3(p)
                    elif k==4:
                        eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, modules=["numpy"])
                        negloglike_all[i] = f4(p)
                else:
                    negloglike_all[i] = np.nan

            except:
                negloglike_all[i] = np.nan

            k -= np.sum(Nsteps<1)       # For the codelen, we effectively don't have the parameter that had Nsteps<1
                
            if k<0:
                print("This shouldn't have happened", flush=True)
                quit()
            elif k==0:                  # If we have no parameters left then the parameter codelength is 0 so we can move on
                continue
            
            fish = fish[Nsteps>=1]     # Only consider these parameters in the codelen
            p = p[Nsteps>=1]
        
        try:        # If p was an array, we can make a list out of it
            list_p = list(p)
            params[i,:] = np.pad(p, (0, 4-len(p)))
        except:     # p is either a number or nothing
            if p:   # p is a number
                params[i,:] = np.array([p, 0, 0, 0])
            else:
                params[i,:] = np.zeros(4)
        
        assert len(params[i,:])==4
        
        try:
            codelen[i] = -k/2.*math.log(3.) + np.sum( 0.5*np.log(fish) + np.log(abs(np.array(p))) )
        except:
            codelen[i] = np.nan

    out_arr = np.transpose(np.vstack([negloglike_all, codelen, index_arr, params[:,0], params[:,1], params[:,2], params[:,3]]))

    np.savetxt(likelihood.temp_dir + '/codelen_matches_'+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.7e')        # Save the data for this proc in Partial

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "codelen_matches_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/codelen_matches_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/codelen_matches_'+str(comp)+'_*.dat'
        os.system(string)
        
    return

