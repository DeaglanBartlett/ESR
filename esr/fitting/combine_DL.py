import numpy as np
from mpi4py import MPI
import os
from prettytable import PrettyTable
import csv
from collections import defaultdict

import esr.fitting.test_all as test_all

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(comp, likelihood, print_frequency=1000):
    """Combine the description lengths of all functions of a given complexity, sort by this and save to file.
    
    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :print_frequency (int, default=1000): the status of the fits will be printed every ``print_frequency`` number of iterations
    
    Returns:
        None
    
    """
    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')
    
    if rank == 0:
        print('\nComputing description lengths', flush=True)

    allfn_file = likelihood.fn_dir + "/compl_%i/all_equations_%i.txt"%(comp,comp)
    aifeyn_file = likelihood.fn_dir + "/compl_%i/%s%i.txt"%(comp,likelihood.fnprior_prefix,comp)

    _, data_start, data_end = test_all.get_functions(comp, likelihood)

    needed_indices = set(np.arange(data_start, data_end)) # faster lookup for indices
    results = defaultdict(list)
    results_fcn = {}

    # Stream read through codelen_matches_comp*.dat file
    if rank == 0:
        with open(likelihood.out_dir + "/codelen_matches_comp" + str(comp) + ".dat", 'r') as f:
            num_lines = sum(1 for _ in f)  # Count total lines in the file

    with open(likelihood.out_dir + "/codelen_matches_comp" + str(comp) + ".dat", 'r') as f, \
         open(aifeyn_file, 'r') as aifeyn_f, \
         open(allfn_file, 'r') as allfn_f:
        
        for i, (line, line_ai, line_fcn) in enumerate(zip(f, aifeyn_f, allfn_f)):

            if rank==0 and i%print_frequency==0:
                print(f'{i+1} of {num_lines}', flush=True)

            if line.strip() == '':
                continue  # Skip empty lines
            parts = line.strip().split()
            idx = int(float(parts[2]))  # Index is in column 3

            if idx in needed_indices:
                
                negloglike_i = float(parts[0])
                codelen_i = float(parts[1])
                aifeyn_i = float(line_ai.strip())  # Read corresponding AIFeyn value
                DL = negloglike_i + codelen_i + aifeyn_i

                if not np.isfinite(DL) or np.isnan(DL):
                    continue

                if (len(results[idx]) == 0) or (DL < results[idx][0]):  # This is the first time we see this index
                    results[idx] = [DL] + [float(x) for x in parts[3:]] + [negloglike_i, codelen_i, aifeyn_i]
                    results_fcn[idx] = line_fcn.strip()  # Store the function string

    num_cols = len(results[next(iter(results))]) if results else 0

    prefix = likelihood.combineDL_prefix

    output_file = likelihood.temp_dir + '/' + prefix + str(comp) + '_' + str(rank) + '.dat'
    output_file_fcn = likelihood.temp_dir + '/'+prefix+'fcn_'+str(comp)+'_'+str(rank)+'.dat'
    with open(output_file, 'w') as fout, \
         open(output_file_fcn, 'w') as fout_fcn:
        for idx in range(data_start, data_end):
            if idx in results:
                line_data = results[idx]
                fcn = results_fcn[idx]
            else:
                line_data = [np.nan] + [0.0] * (num_cols-1)
                fcn = "None"
            
            fout.write(" ".join(f"{x:.16e}" for x in line_data) + "\n")
            fout_fcn.write(f"{fcn}\n")           

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
        data_entries = []
        num_params = 0
        with open(likelihood.out_dir + '/'+prefix+'comp'+str(comp)+'.dat', 'r') as f, \
             open(likelihood.out_dir + '/'+prefix+'fcn_comp'+str(comp)+'.dat', "r") as fcn_f:
            for i, (line, fcn_line) in enumerate(zip(f, fcn_f)):
                parts = line.strip().split()
                if not parts:
                    continue
                DL = float(parts[0])
                if (not np.isnan(DL)) and (not np.isinf(DL)):
                    data_entries.append( (DL, parts[1:], fcn_line.strip()) )  # Store DL, index, and other info
                if num_params == 0:
                    num_params = len(parts) - 4
        print(f"Number of parameters: {num_params}", flush=True)
        print(f'Original file length: {i+1}', flush=True)
        data_entries.sort(key=lambda x: x[0])
        print(f"Sorted {len(data_entries)} entries by DL for complexity {comp}", flush=True)

        # Get relative probabilities
        Prel_DL = np.array([entry[0] for entry in data_entries])
        log_L = np.array([entry[1][-3] for entry in data_entries])
        Prel_DL -= Prel_DL[0]  # Shift so the best function has DL=0
        Prel = np.exp(-Prel_DL)
        duplicates = log_L[1:] == log_L[:-1]
        Prel[1:][duplicates] = 0.0
        Prel[~np.isfinite(Prel) | np.isnan(Prel)] = 0.0
        Prel /= np.sum(Prel)
        
        ptab = PrettyTable()
        ptab.field_names = ["Rank", "Function", "L(D)", "Prel", "-logL", "Codelen", "AIFeyn"] + [f"a{i}" for i in range(num_params)]

        Nfuncs = 10

        # Start this file from scratch here
        if os.path.exists(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat'):
            os.remove(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat')

        for i, d in enumerate(data_entries):  # Only print the top 10 functions
            if i < Nfuncs:
                fcn = d[-1]
                DL = d[0]
                params = [float(pp) for pp in d[1][:-3]]
                negloglike = float(d[1][-3])
                codelen = float(d[1][-2])
                aifeyn = float(d[1][-1])
                ptab.add_row([i+1, fcn, '%.2f'%DL, '%.2e'%Prel[i], '%.2f'%negloglike, '%.2f'%codelen, '%.2e'%aifeyn] + [ '%.2e'%p for p in params])
    
            with open(likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat', 'a') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow([i, 
                                 d[-1], # fcn
                                 d[0],  # DL
                                 Prel[i], 
                                 d[1][-3], # negloglike
                                 d[1][-2], # codelen
                                 d[1][-1]] + d[1][:-3]) # aifeyn, params

        if len(data_entries) == 0:
            os.system("touch " + likelihood.out_dir + '/'+likelihood.final_prefix+str(comp)+'.dat')

        print(ptab)

        with open(likelihood.out_dir + '/results_pretty_'+str(comp)+'.txt', 'w') as f:
            print(ptab, file=f)

    comm.Barrier()
        
    return
    
