###################################################################
# Script to convert a list of equations and a given data file into
# pairs of complexity and MSE on the data set
###################################################################
import numpy as np
import pandas as pd
import itertools
import sympy
import os
import sys
from mpi4py import MPI

#os.chdir('../ESR/generation')
#sys.path.insert(1, '.')
#import generator
import esr.generation.generator as generator

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dirname = '/Users/deaglanbartlett/Desktop/OneDrive - Nexus365/IAP/Work/Exhaustive_Symbolic_Regression/benchmark/'

basis_functions = [["x", "a"],  # type0
                ["square", "exp", "inv", "sqrt_abs", "log_abs"],  # type1
                ["+", "*", "-", "/", "pow"]]  # type2
data_fname = dirname + 'feynman_I_6_2a.tsv'

fn_fname = dirname + 'DM_feyn_fcns_5hr_2.dat'
output_full_fname = dirname + 'DM_5hr_converted.txt'
output_pareto_fname = dirname + 'DM_5hr_pareto.txt'
output_pareto_fun_fname = dirname + 'DM_5hr_pareto_fun.txt'

fn_fname = dirname + 'DM_feyn_fcns_10hr_2.dat'
output_full_fname = dirname + 'DM_10hr_converted.txt'
output_pareto_fname = dirname + 'DM_10hr_pareto.txt'
output_pareto_fun_fname = dirname + 'DM_10hr_pareto_fun.txt'

fn_fname = dirname + 'qlattice_BIC.dat'
output_full_fname = dirname + 'qlattice_BIC_converted.txt'
output_pareto_fname = dirname + 'qlattice_BIC_pareto.txt'
output_pareto_fun_fname = dirname + 'qlattice_BIC_pareto_fun.txt'


fn_fname = dirname + 'qlattice_MSE.dat'
output_full_fname = dirname + 'qlattice_MSE_converted.txt'
output_pareto_fname = dirname + 'qlattice_MSE_pareto.txt'
output_pareto_fun_fname = dirname + 'qlattice_MSE_pareto_fun.txt'

with open(fn_fname, 'r') as f:
    all_fun = f.read().splitlines()

df = np.array(pd.read_table(data_fname, sep="\t"))
xtrue = df[:,0]
ytrue = df[:,1]
    
# SHUFFLE AND SPLIT
shuffler = np.random.permutation(len(all_fun)) # returns indices to shuffle the list
if rank == 0:
    undo_shuffler = np.argsort(shuffler) # returns indices to undo the shuffle
    shuffler = np.array_split(shuffler, size)
shuffler = comm.scatter(shuffler, root=0)
shuffled_fun = [all_fun[j] for j in shuffler]

# GET COMPLEXITY AND MSE
shuffled_c = np.zeros(len(shuffled_fun), dtype=int)
shuffled_mse = np.zeros(len(shuffled_fun))
for i, s in enumerate(shuffled_fun):
    if rank == 0:
        print(f"{i+1} of {len(shuffled_c)}", flush=True)
    expr, nodes, c = generator.string_to_node(s, basis_functions)
    shuffled_c[i] = c
    eq = sympy.lambdify(generator.x, expr)
    ypred = eq(xtrue)
    shuffled_mse[i] = np.mean((ypred - ytrue) ** 2)
#    l = nodes.to_list(basis_functions)
#    print(c, len(l))


# RECOMBINE
shuffled_c = comm.gather(shuffled_c, root=0)
shuffled_mse = comm.gather(shuffled_mse, root=0)
if rank == 0:
    shuffled_c = list(itertools.chain(*shuffled_c))
    shuffled_mse = list(itertools.chain(*shuffled_mse))
    compl = [shuffled_c[j] for j in undo_shuffler]
    mse = [shuffled_mse[j] for j in undo_shuffler]
else:
    compl = None
    mse = None
compl = comm.bcast(compl, root=0)
mse = comm.bcast(mse, root=0)

compl = np.array(compl)
mse = np.array(mse)

if rank == 0:
    np.savetxt(output_full_fname, np.transpose(np.vstack([compl, mse])))
    
    # Make Pareto front
    unique_compl = np.unique(compl)
    pareto_mse = np.empty(len(unique_compl))
    pareto_fun = [None] * len(unique_compl)
    mse = np.array(mse)
    
    for i, c in enumerate(unique_compl):
        m = compl == c
        if m.sum() == 1:
            pareto_mse[i] = np.squeeze(mse[m])
            j = np.squeeze(np.where(m == 1))
        else:
            pareto_mse[i] = np.amin(mse[m])
            j = np.squeeze(np.where(mse == pareto_mse[i]))
            if isinstance(j, np.ndarray):
                j = j[compl[j]==c][0]
        pareto_fun[i] = all_fun[j]
        
    np.savetxt(output_pareto_fname, np.transpose(np.vstack([unique_compl, pareto_mse])))
    
    with open(output_pareto_fun_fname, 'w') as f:
        for i in range(len(unique_compl)):
            print(unique_compl[i], pareto_mse[i], '(' + pareto_fun[i] + ')', file=f)
    
    i = np.argmin(mse)
    print(compl[i], mse[i], all_fun[i])
        
    
    
quit()

s = 'x * x'
s = '3 * x + 2'
s = '1 / x'
s = 'sqrt(a * x)'
expr, nodes, c = generator.string_to_node(s, basis_functions)

print(expr)
print(nodes)
print(c)

print(nodes.children)

l = nodes.to_list(basis_functions)
print(l)
