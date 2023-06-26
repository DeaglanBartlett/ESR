import math
import numpy as np
from mpi4py import MPI
import os, sys
from prettytable import PrettyTable
import csv

import esr.fitting.test_all as test_all

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(comp, likelihood):
    """Combine the description lengths of all functions of a given complexity, sort by this and save to file.
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
    
    Returns:
        None
    
    """

    unifn_file = likelihood.fn_dir + "/compl_%i/unique_equations_%i.txt"%(comp,comp)
    allfn_file = likelihood.fn_dir + "/compl_%i/all_equations_%i.txt"%(comp,comp)
    aifeyn_file = likelihood.fn_dir + "/compl_%i/%s%i.txt"%(comp,likelihood.fnprior_prefix,comp)

    use_deriv = False

    with open(unifn_file, "r") as f:         # All
        fcn_list = f.read().splitlines()

    with open(allfn_file, "r") as f:         # All
        fcn_list_all = f.read().splitlines()

#    negloglike, codelen, index, param1, param2, param3, param4 = np.genfromtxt(likelihood.out_dir + "/codelen_matches_comp"+str(comp)+".dat", unpack=True)        # All
    data = np.genfromtxt(likelihood.out_dir + "/codelen_matches_comp"+str(comp)+".dat") # All
    negloglike = data[:,0]
    codelen = data[:,1]
    index = data[:,2]
    params = data[:,3:]
    aifeyn = np.genfromtxt(aifeyn_file) # All
    codelen = np.atleast_1d(codelen)
    index = np.atleast_1d(index)
    aifeyn = np.atleast_1d(aifeyn)

    fcn_list_proc, data_start, data_end = test_all.get_functions(comp, likelihood)

    DL_min = np.zeros(len(fcn_list_proc))
    params_min = np.zeros((len(fcn_list_proc), params.shape[1]))  # These are all now specific to the proc

    fcn_min = [None] * len(fcn_list_proc)
    negloglike_min = np.zeros(len(fcn_list_proc))
    codelen_min = np.zeros(len(fcn_list_proc))
    aifeyn_min = np.zeros(len(fcn_list_proc))

    xarr = np.linspace(0, len(fcn_list)-1, len(fcn_list)).astype(int)           # Indices of all the unique fcns, which are what we're looping over
    xarr_proc = xarr[data_start:data_end]        # Which unique function indices this proc will look at

    for i in range(len(fcn_list_proc)):          # Loop over all unique fcns to find variant with min codelength
        if rank==0 and i%1000==0:
            print(i)
            
        negloglike_i, codelen_i, aifeyn_i = negloglike[index==xarr_proc[i]], codelen[index==xarr_proc[i]], aifeyn[index==xarr_proc[i]]           # Arrays of all variants for this unique fcn
        
        m = (index==xarr_proc[i])
        fcn_list_all_i = [fcn_list_all[j] for j in range(len(m)) if m[j]]
        params_i = params[index==xarr_proc[i], :]
        DL = negloglike_i + codelen_i + aifeyn_i
        
        if np.sum(~np.isnan(DL))==0:
            DL_min[i] = np.nan
            continue

        DL_min[i] = np.nanmin(DL)
        params_min[i,:] = params_i[np.nanargmin(DL),:]
        fcn_min[i] = fcn_list_all_i[np.nanargmin(DL)]
        
        negloglike_min[i] = negloglike_i[np.nanargmin(DL)]
        codelen_min[i] = codelen_i[np.nanargmin(DL)]
        aifeyn_min[i] = aifeyn_i[np.nanargmin(DL)]

    out_arr = np.transpose(np.vstack([DL_min] + [params_min[:,i] for i in range(params_min.shape[1])] + [negloglike_min, codelen_min, aifeyn_min]))
    
    prefix = likelihood.combineDL_prefix

    np.savetxt(likelihood.temp_dir + '/'+prefix+str(comp)+'_'+str(rank)+'.dat', out_arr, fmt='%.16e')        # Save the data for this proc in Partial
    np.savetxt(likelihood.temp_dir + '/'+prefix+'fcn_'+str(comp)+'_'+str(rank)+'.dat', fcn_min, fmt="%s")
    # One per unique eqn, but I save the form in "all" that gives the lowest DL

    comm.Barrier()

    if rank == 0:
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "'+prefix+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/'+prefix+'comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/'+prefix+str(comp)+'_*.dat'
        os.system(string)
        
        string = 'cat `find ' + likelihood.temp_dir + '/ -name "'+prefix+'fcn_'+str(comp)+'_*.dat" | sort -V` > ' + likelihood.out_dir + '/'+prefix+'fcn_comp'+str(comp)+'.dat'
        os.system(string)
        string = 'rm ' + likelihood.temp_dir + '/'+prefix+'fcn_'+str(comp)+'_*.dat'
        os.system(string)
        
    if rank==0:         # The rest is done by just one proc
        data = np.genfromtxt(likelihood.out_dir + '/'+prefix+'comp'+str(comp)+'.dat')            # This is the combined results from all procs, and the rest should be as before
        DL_min = data[:,0]
        params_min = data[:,1:1+params.shape[1]]
        print(data.shape)
        negloglike_min = data[:,-3]
        codelen_min = data[:,-2]
        aifeyn_min = data[:,-1]
        
        DL_min = np.atleast_1d(DL_min)
        params_min = np.atleast_2d(params_min)
        negloglike_min = np.atleast_1d(negloglike_min)
        codelen_min = np.atleast_1d(codelen_min)
        aifeyn_min = np.atleast_1d(aifeyn_min)

        with open(likelihood.out_dir + '/'+prefix+'fcn_comp'+str(comp)+'.dat', "r") as f:         # All
            fcn_min = f.read().splitlines()

        mask = ~np.isnan(DL_min)

        xarr = np.linspace(0, len(fcn_list)-1, len(fcn_list)).astype(int)           # fcn_list should be as it was read in at the top

        DL_min = DL_min[mask]
        xarr = xarr[mask]

        arr_sort = np.transpose( sorted( np.transpose(np.vstack([DL_min, xarr])), key = lambda x: x[0] ) )     # Sort by DL but keep track of array indices
        DL_sort = arr_sort[0,:]
        indices_sort = arr_sort[1,:].astype(int)

        params_sort = params_min[indices_sort,:]
        fcn_min_sort = [fcn_min[i] for i in indices_sort]

        negloglike_sort = negloglike_min[indices_sort]
        codelen_sort = codelen_min[indices_sort]
        aifeyn_sort = aifeyn_min[indices_sort]

        if os.path.exists(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat'):           # Start this file from scratch here
            os.remove(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat')

        Nfuncs = 10

        Prel_DL = np.zeros(len(negloglike_sort))+np.inf
        negloglike_previous = np.nan
        for i in range(len(negloglike_sort)):
            if negloglike_sort[i] == negloglike_previous:       # Never happens for 0th fcn bc negloglike would have to be nan
                continue                                        # Prel_DL stays at inf for this duplicate function, so Prel -> 0
            Prel_DL[i] = DL_sort[i] - DL_sort[0]                # Always gives 0 for the 0th function, so this gets the highest Prel
            negloglike_previous = negloglike_sort[i]            # This will then be used for the next fcn

        Prel = np.exp(-Prel_DL)             # Don't want to use every fcn here bc they could be inf or nan, but the best 1000 should be fine
        Prel /= np.sum(Prel)                # Relative probability of fcn, normalised over the top 1000 functions just of this complexity

        ptab = PrettyTable()
        ptab.field_names = ["Rank", "Function", "L(D)", "Prel", "-logL", "Codelen", "AIFeyn"] + [f"a{i}" for i in range(params.shape[1])]

        negloglike_previous = np.nan

        for i in range(len(DL_sort)):
            
            # Only happens for non-duplicates; all Prels should be non-zero
            if i < Nfuncs:
                ptab.add_row([i+1, fcn_min_sort[i], '%.2f'%DL_sort[i], '%.2e'%Prel[i], '%.2f'%negloglike_sort[i], '%.2f'%codelen_sort[i], '%.2e'%aifeyn_sort[i]] + [ '%.2e'%params_sort[i,j] for j in range(params.shape[1])])

            
            with open(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat', 'a') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([i, fcn_min_sort[i], DL_sort[i], Prel[i], negloglike_sort[i], codelen_sort[i], aifeyn_sort[i]] + [params_sort[i,j] for j in range(params.shape[1])])

            negloglike_previous = negloglike_sort[i]
        print(ptab)
    
    comm.Barrier()
        
    return
    
