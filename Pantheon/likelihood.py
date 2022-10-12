import astropy.constants
import astropy.units as apu
import numpy as np
import pandas as pd
import scipy.integrate
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from filenames import *

def load_data():
    
    data = pd.read_csv(data_file, delim_whitespace=True)
    origlen = len(data)
    ww = (data['zHD']>0.01)
    zCMB = data['zHD'][ww]
    mu_obs = data['MU_SH0ES'][ww]
    mu_err = data['MU_SH0ES_ERR_DIAG'][ww]  # for plotting
    
    with open(cov_file, 'r') as f:
        line = f.readline()
        n = int(len(zCMB))
        C = np.zeros((n,n))
        ii = -1
        jj = -1
        mine = 999
        maxe = -999
        for i in range(origlen):
            jj = -1
            if ww[i]:
                ii += 1
            for j in range(origlen):
                if ww[j]:
                    jj += 1
                val = float(f.readline())
                if ww[i]:
                    if ww[j]:
                        C[ii,jj] = val
                        
    inv_cov = np.linalg.inv(C)

    return zCMB.to_numpy(), mu_obs.to_numpy(), inv_cov, mu_err.to_numpy()

Hfid = 67.4 * apu.km / apu.s / apu.Mpc
mu_const =  astropy.constants.c / Hfid / (10 * apu.pc)
mu_const = 5 * np.log10(mu_const.to(''))

delta_z = 0.002
def get_mu(zp1, a, eq_numpy, integrated=False):
    if integrated:
        dL = eq_numpy(zp1, *a) - eq_numpy(1, *a)
    else:
        if len(a) == 0:
            lam = lambda x: 1 / np.sqrt(eq_numpy(x))
        else:
            lam = lambda x: 1 / np.sqrt(eq_numpy(x, *a))
        dL = np.ones(len(zp1)) * np.nan
        for i in range(len(dL)):
#            dL[i], _ = scipy.integrate.quad(lam, 1, zp1[i])
            x = np.linspace(1, zp1[i], int((zp1[i]-1)/delta_z))
            dL[i] = scipy.integrate.simpson(lam(x), x)
    dL *= zp1
    mu = 5 * np.log10(dL) + mu_const
    return mu
    
    
def negloglike(a, eq_numpy, xvar, yvar, inv_cov, integrated=False):
    mu_pred = get_mu(xvar, np.atleast_1d(a), eq_numpy, integrated=integrated)
    nll = 0.5 * np.dot((mu_pred - yvar), np.dot(inv_cov,(mu_pred - yvar)))
    if np.isnan(nll):
        return np.inf
    return nll
