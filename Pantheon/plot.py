import sys
import matplotlib.pyplot as plt
import csv
from mpi4py import MPI
import warnings
import numpy as np
import sympy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import os

import likelihood
from filenames import *
from sympy_symbols import *

sys.path.insert(0, esr_dir)
import simplifier

warnings.filterwarnings("ignore")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def main(comp, tmax=5):

    if rank != 0:
        return

    vmin = 1e-50
    vmax = 1
    tmax = 5

    if comp==8:
        sys.setrecursionlimit(2000)
    elif comp==9:
        sys.setrecursionlimit(2500)
    elif comp==10:
        sys.setrecursionlimit(3000)

    if not os.path.isdir(fig_dir):
        print('Making:', fig_dir)
        os.mkdir(fig_dir)

    max_param = 4

    with open(out_dir + '/final_'+str(comp)+'.dat', "r") as f:
        reader = csv.reader(f, delimiter=';')
        data = [row for row in reader]
        
    fcn_list = [d[1] for d in data]
    params = np.array([d[-4:] for d in data], dtype=float)
    DL = np.array([d[2] for d in data], dtype=float)
    DL_min = np.amin(DL[np.isfinite(DL)])
    print('MIN', DL_min)
    alpha = DL_min - DL
    alpha = np.exp(alpha)
    m = (alpha > vmin)
    fcn_list = [d for i, d in enumerate(fcn_list) if m[i]]
    params = params[m,:]
    alpha = alpha[m]

    print('Loading data')
    xvar, yvar, inv_cov, yerr = likelihood.load_data()
    xvar += 1  # now x = 1 + z
    print('Loaded')
    
    fig  = plt.figure(figsize=(7,5))
    ax1  = fig.add_axes([0.10,0.10,0.70,0.85])
    cmap = cm.hot_r
    norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)

    for i in range(len(fcn_list)):                 # The part of all eqs analysed by this proc

        fcn_i = fcn_list[i].replace('\'', '')
        
        k = simplifier.count_params([fcn_i], max_param)[0]
        measured = params[i,:k]
        
        try:
            fcn_i, eq, integrated = run_sympify(fcn_i, tmax=tmax)
            if k == 0:
                eq_numpy = sympy.lambdify([x], eq, "numpy")
            elif k==1:
                eq_numpy = sympy.lambdify([x, a0], eq, "numpy")
            elif k==2:
                eq_numpy = sympy.lambdify([x, a0, a1], eq, "numpy")
            elif k==3:
                eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, "numpy")
            elif k==4:
                eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, "numpy")
            ypred = likelihood.get_mu(xvar, measured, eq_numpy, integrated=integrated)
        except:
            fcn_i, eq, integrated = run_sympify(fcn_i, tmax=tmax, try_integration=False)
            if k == 0:
                eq_numpy = sympy.lambdify([x], eq, "numpy")
            elif k==1:
                eq_numpy = sympy.lambdify([x, a0], eq, "numpy")
            elif k==2:
                eq_numpy = sympy.lambdify([x, a0, a1], eq, "numpy")
            elif k==3:
                eq_numpy = sympy.lambdify([x, a0, a1, a2], eq, "numpy")
            elif k==4:
                eq_numpy = sympy.lambdify([x, a0, a1, a2, a3], eq, "numpy")
            ypred = likelihood.get_mu(xvar, measured, eq_numpy, integrated=integrated)

        ax1.plot(xvar-1, ypred, color=cmap(norm(alpha[i])), zorder=len(fcn_list)-i)
        
    ax1.errorbar(xvar-1, yvar, yerr=yerr, fmt='.', markersize=5, zorder=len(fcn_list)+1, capsize=1, elinewidth=1, color='k', alpha=1)
    ax1.set_xlabel(r'$z$')
    ax1.set_ylabel(r'$\mu \left( z \right)$')
    ax1.set_xlim(0, None)

    ax2  = fig.add_axes([0.85,0.10,0.05,0.85])
    cb1  = mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,orientation='vertical')
    cb1.set_label(r'$\exp \left( MDL - DL \right)$')
    fig.tight_layout()
    fig.savefig(fig_dir + '/plot_%i.png'%comp)

    return
        
if __name__ == "__main__":
    comp = int(sys.argv[1])
    main(comp)
