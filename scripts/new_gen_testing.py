import time
import numpy as np
import matplotlib.pyplot as plt

import esr.generation.make_trees as make_trees
from esr.fitting.sympy_symbols import *

def main():

#    basis_functions = [["x", "a"],  # nullary
#                        ["inv", "exp", "log"],  # unary
#                        ["+", "*", "-", "/", "pow"]]  # binary
    basis_functions = [["x", "a"],  # nullary
                        ["inv"],  # unary
                        ["+", "*", "-", "/", "pow"]]  # binary
                        
    commutative_dict = {"+":True, "*":True, "-":False, "/":False, "pow":False}
    unary_inv = {"inv":"inv", "exp":"log", "log":"exp"}
    locs = {"inv": inv,
                "square": square,
                "cube": cube,
                "sqrt": sqrt,
                "log": log,
                "pow": pow,
                "x": x
                }
                
    max_comp = 6 #10
    
    start = time.time()
    all_fun = make_trees.make_fun(max_comp, basis_functions, commutative_dict, unary_inv)
    end = time.time()
    print('\n*******************************************************')
    print(f'Total time to make up to comp {max_comp}: {end-start} s')
    print('*******************************************************\n')
    nfun = [len(f) for f in all_fun]
    print(nfun)
    
    # Find number of unique fun ESR makes
    esr_uniq = [0] * len(nfun)
    for comp in range(1, len(nfun)):
        with open(f"../../../esr_test/ESR/esr/function_library/core_maths/compl_{comp}/unique_equations_{comp}.txt", "r") as f:
            esr_uniq[comp] = sum(1 for _ in f)
    print(esr_uniq)
            
    # Find total number of fun ESR makes
    esr_total = [0] * len(nfun)
    for comp in range(1, len(nfun)):
        with open(f"../../../esr_test/ESR/esr/function_library/core_maths/compl_{comp}/all_equations_{comp}.txt", "r") as f:
            esr_total[comp] = sum(1 for _ in f)
    print(esr_total)
            
    nfun = np.array(nfun)[1:]
    esr_uniq = np.array(esr_uniq)[1:]
    esr_total = np.array(esr_total)[1:]
    all_comp = np.arange(max_comp) + 1
    
    ratio_total = nfun / esr_total
    print("\nAll equations")
    for c, r in zip(all_comp, ratio_total):
        print("%i: %.2f"%(c,r))
    full_ratio = nfun.sum() / esr_total.sum()
    print("Summed: %.2f"%full_ratio)
    
    ratio_uniq = nfun / esr_uniq
    print("\nUnique equations")
    for c, r in zip(all_comp, ratio_uniq):
        print("%i: %.2f"%(c,r))
    full_ratio = nfun.sum() / esr_uniq.sum()
    print("Summed: %.2f"%full_ratio)
    
    
    fig, axs = plt.subplots(2, 1, figsize=(7,6), sharex=True)
    axs[0].plot(all_comp, esr_uniq, marker='.', label='Unique ESR')
    axs[0].plot(all_comp, esr_total, marker='.', label='Total ESR')
    axs[0].plot(all_comp, nfun, marker='.', label='New method')
    axs[1].plot(all_comp, ratio_uniq, marker='.', label='Unique ESR')
    axs[1].plot(all_comp, ratio_total, marker='.', label='Total ESR')
    axs[1].axhline(y=1, color='k')
    axs[0].legend()
    axs[1].legend()
    axs[1].set_xlabel('Complexity')
    axs[0].set_ylabel('Number of Functions')
    axs[1].set_ylabel('New method / Old method')
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    fig.align_ylabels()
    fig.tight_layout()
#    plt.show()
    fig.clf()
    plt.close(fig)
    
    print(locs)
    
    # Find which ones we don't need
    for comp in range(1, 5):
        print('\n', comp)
        with open(f"../../../esr_test/ESR/esr/function_library/core_maths/compl_{comp}/unique_equations_{comp}.txt", "r") as f:
            esr_eq = [ff.strip() for ff in f]
        new_eq = [f.to_string(locs) for f in all_fun[comp]]
        print(len(set(new_eq)), len(new_eq), len(esr_eq))
        print(esr_eq)
        print(new_eq)
        shared, orig_idx, new_idx = np.intersect1d(esr_eq, new_eq, return_indices=True)
#        diff = list(set(new_eq) - set(esr_eq))
#        print(diff)
#        diff = list(set(esr_eq) - set(new_eq))
#        print(diff)
        extra = [f for i,f in enumerate(all_fun[comp]) if i not in new_idx]
        print(len(extra))
        print(extra)
        
#    comm.Barrier()
#    if rank == 0:
#        print('starting', max_comp)
#    comm.Barrier()
#    start = time.time()
#    st, end = split_list(len(all_fun[max_comp]))
#    s = [f.to_sympy(locs).__str__() for f in all_fun[max_comp][st:end]]
#    s = comm.gather(s, root=0)
#    end = time.time()
#    if rank == 0:
#        print('done')
#        print(end - start)

#    start = time.time()
#    s = [f.to_sympy(locs).__str__() for f in all_fun[max_comp]]
#    end = time.time()
#    print(end-start)
    
    return


if __name__ == "__main__":
    main()

