import os
import matplotlib.pyplot as plt
import csv
import numpy as np

def pareto_plot(dirname, savename, do_DL=True, do_logL=True):
    """
    Plot the pareto front using the files in a given directory
    
    Args:
        :dirname (str): The directory name to consider.
        :savename (str): File name to save file (within dirname)
        :do_DL (bool, default=True): Whether to plot the description length in the pareto front
        :do_logL (bool, default=True): Whether to plot the log-likelihood in the pareto front
    """
    
    if (not do_DL) and (not do_logL):
        return

    all_f = os.listdir(dirname)
    all_f = [f for f in all_f if f.startswith('final_')]
    all_comp = [int(f[len('final_'):-len('.dat')]) for f in all_f]
    all_logL = np.empty(len(all_comp))
    all_DL = np.empty(len(all_comp))
    
    for i, fname in enumerate(all_f):
        print(i, fname)
        
        with open(dirname + '/' + fname, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            data = [row for row in reader]
            data = np.array([d[2:7] for d in data], dtype=float)
            
        # Get min DL
        all_DL[i] = np.amin(data[:,0])
        all_logL[i] = np.amin(data[:,2])
    
    all_DL -= np.amin(all_DL)
    all_logL -= np.amin(all_logL)

    fig, ax1 = plt.subplots(1, 1, figsize=(5,3.5), sharex=True)
    cm = plt.get_cmap('Set1')
    
    if do_DL and do_logL:
        ax2 = ax1.twinx()
        ax1.plot(all_comp, all_DL, marker='.', color=cm(0), markersize=5)
        ax1.plot(all_comp, all_logL, marker='.', color=cm(1), markersize=5)
        
        ax1.set_ylabel(r'$\Delta L \left( D \right)$')
        ax2.set_ylabel(r'$ \left| \Delta \log\mathcal{L} \right|$')
        ax1.yaxis.label.set_color(cm(0))
        ax1.tick_params(axis='y', colors=cm(0))
        ax2.spines['left'].set_color(cm(0))

        ax2.yaxis.label.set_color(cm(1))
        ax2.tick_params(axis='y', colors=cm(1))
        ax2.spines['right'].set_color(cm(1))

        ax2.set_ylim(0, None)

    else:
        if do_DL:
            y = all_DL
            c = cm(0)
        else:
            y = all_logL
            c = cm(1)
            
        ax1.plot(all_comp, y, marker='.', color=c, markersize=5)
        
        if do_DL:
            ax1.set_ylabel(r'$\Delta L \left( D \right)$')
        else:
            ax1.set_ylabel(r'$ \left| \Delta \log\mathcal{L} \right|$')
        ax1.yaxis.label.set_color(c)
        ax1.tick_params(axis='y', colors=c)
    
    ax1.set_ylim(0, None)
    ax1.set_xticks(all_comp)
    ax1.set_xticklabels(all_comp)
    ax1.set_xlabel(r'Complexity')
    
    fig.tight_layout()
    fig.savefig(dirname + '/' + savename, bbox_inches='tight')
    
    fig.clf()
    plt.close(fig)
        
    return
