import numpy as np
from filenames import *


def load_data():
    xvar, yvar, e_yvar = np.genfromtxt(data_file, unpack=True)
    xvar -= 1
    inv_cov = 1 / e_yvar ** 2
    return xvar, yvar, inv_cov, e_yvar


def get_pred(zp1, a, eq_numpy, **kwargs):
    return np.sqrt(eq_numpy(zp1, *a))


def negloglike(a, eq_numpy, xvar, yvar, inv_cov, **kwargs):
    H = get_pred(xvar, np.atleast_1d(a), eq_numpy)
    if not np.all(np.isreal(H)):
        return np.inf
    nll = np.sum(0.5 * (H - yvar) ** 2 * inv_cov)  # inv_cov diagonal, so is vector here
    if np.isnan(nll):
        return np.inf
    return nll


