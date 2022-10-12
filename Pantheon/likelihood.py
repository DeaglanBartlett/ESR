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

delta_z = 0.02
min_nz = 10
data_x = None
data_mask = None

def get_mu(zp1, a, eq_numpy, integrated=False):
    global data_x, data_mask

    if integrated:
        dL = eq_numpy(zp1, *a) - eq_numpy(1, *a)
    else:
        if data_x is None or data_mask is None:
            nx = int(np.ceil((zp1.max() - zp1.min()) / delta_z))
            data_x = np.concatenate(
                        (np.linspace(1, zp1.min(), min_nz),
                        np.linspace(zp1.min() + delta_z, zp1.max() + delta_z, nx),
                        zp1))
            data_x = np.sort(np.unique(data_x))
            data_mask = np.squeeze(np.array([np.where(data_x==d)[0] for d in zp1]))
        
        if len(a) == 0:
            dL = 1 / np.sqrt(eq_numpy(data_x))
        else:
            dL = 1 / np.sqrt(eq_numpy(data_x, *a))

        dL = scipy.integrate.cumtrapz(dL, x=data_x, initial=0)
        dL = dL[data_mask]
    
    dL *= zp1
    mu = 5 * np.log10(dL) + mu_const
    return mu

    
def negloglike(a, eq_numpy, xvar, yvar, inv_cov, integrated=False):
    mu_pred = get_mu(xvar, np.atleast_1d(a), eq_numpy, integrated=integrated)
    nll = 0.5 * np.dot((mu_pred - yvar), np.dot(inv_cov,(mu_pred - yvar)))
    if np.isnan(nll):
        return np.inf
    return nll
