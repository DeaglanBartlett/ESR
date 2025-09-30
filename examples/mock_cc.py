import numpy as np
from esr.fitting.likelihood import CCLikelihood


seed = 1234
np.random.seed(seed)

cc_like = CCLikelihood()
print(cc_like.xvar.shape)
frac_err = cc_like.yerr / cc_like.yvar
print(np.amin(frac_err), np.amax(frac_err),
      np.median(frac_err), np.mean(frac_err))
zmin = cc_like.xvar.min() - 1
zmax = cc_like.xvar.max() - 1
print(zmin, zmax)

# Generate mock tracers
zmin = 0.07
zmax = 2
nz = 100 * cc_like.xvar.shape[0]
# nz = 10 * cc_like.xvar.shape[0]
# nz = 1000
# nz = 20 * cc_like.xvar.shape[0]
# nz = 800

all_z = 10 ** np.random.uniform(np.log10(1+zmin), np.log10(1+zmax), nz) - 1
all_z = np.sort(all_z)
print(all_z.min(), all_z.max())

# Get truth
H0 = 67.4
Om0 = 0.315
Htrue = H0 * np.sqrt(Om0 * (1 + all_z) ** 3 + (1 - Om0))

# Scatter by yfracerr
# yfracerr = 0.2
# yfracerr = 0.1
# yfracerr = 0.05
yfracerr = 0.01
Hmeas = np.random.normal(Htrue, Htrue*yfracerr)
m = (Hmeas < 0)
while m.sum() > 0:
    Hmeas[m] = np.random.normal(Htrue[m], Htrue[m]*yfracerr)
    m = (Hmeas < 0)

fname = '../data/mock/CC_Hubble_%i_' % nz + str(yfracerr) + '.dat'
outarr = np.transpose(np.vstack([all_z, Hmeas, Htrue*yfracerr]))
np.savetxt(fname, outarr)
