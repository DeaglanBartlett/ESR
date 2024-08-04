import astropy.constants
import astropy.units as apu
import numpy as np
import pandas as pd
import scipy.integrate
import sympy
import os
import warnings

from esr.fitting.sympy_symbols import (
    square, cube, sqrt, log, pow, x, a0, a1, a2, inv
)

from esr.generation.simplifier import time_limit
import esr.generation.simplifier

class Likelihood:
    """Likelihood class used to fit a function directly
    
    Args:
        :data_file (str): Name of the file containing the data to use
        :cov_file (str): Name of the file containing the errors/covariance on the data
        :run_name (str): The name to be associated with this likelihood, e.g. 'my_esr_run'
        :data_dir (str, default=None): The path containing the data and cov files
        :fn_set (str, default='core_maths'): The name of the function set to use with the likelihood. Must match one of those defined in ``generation.duplicate_checker``
    
    """

    def __init__(self, data_file, cov_file, run_name, data_dir=None, fn_set='core_maths'):

        esr_dir = os.path.abspath(os.path.join(os.path.dirname(esr.generation.simplifier.__file__), '..', '')) + '/'
        if data_dir is None:
            self.data_dir = esr_dir + '/data/'
        else:
            self.data_dir = data_dir
        self.data_file = self.data_dir + '/' + data_file
        self.cov_file = self.data_dir + '/' + cov_file
        self.fn_dir = esr_dir + "function_library/" + fn_set + "/"
        if data_dir is None:
            self.like_dir = esr_dir + "/fitting/"
        else:
            self.like_dir = data_dir + "/fitting/"
        if not os.path.isdir(self.like_dir):
            os.mkdir(self.like_dir)
        self.fnprior_prefix = "aifeyn_"
        self.combineDL_prefix = "combine_DL_"
        self.final_prefix = "final_"
        
        self.base_out_dir = self.like_dir + "/output/"
        self.temp_dir = self.base_out_dir + "/partial_" + run_name
        self.out_dir = self.base_out_dir + "/output_" + run_name
        self.fig_dir = self.base_out_dir + "/figs_" + run_name
        
        # Warning to not use MSE for DL
        self.is_mse = False

    def get_pred(self, x, a, eq_numpy, **kwargs):
        """Return the predicted y(x)
        
        Args:
            :x (float or np.array): x value being used
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :y (float or np.array): the predicted y value at x supplied
        
        """
        try:
            return eq_numpy(x, *a)
        except Exception:
            return np.inf

    def clear_data(self):
        """Clear data used for numerical integration (not required in most cases)"""
        pass
    

    def run_sympify(self, fcn_i, **kwargs):
        """Sympify a function

        Args:
            :fcn_i (str): string representing function we wish to fit to data

        Returns:
            :fcn_i (str): string representing function we wish to fit to data (with superfluous characters removed)
            :eq (sympy object): sympy object representing function we wish to fit to data
            :integrated (bool, always False): whether we analytically integrated the function (True) or not (False)

        """

        fcn_i = fcn_i.replace('\n', '')
        fcn_i = fcn_i.replace('\'', '')

        eq = sympy.sympify(fcn_i,
                    locals={"inv": inv,
                            "square": square,
                            "cube": cube,
                            "sqrt": sqrt,
                            "log": log,
                            "pow": pow,
                            "x": x,
                            "a0": a0,
                            "a1": a1,
                            "a2": a2})
        return fcn_i, eq, False



class CCLikelihood(Likelihood):
    """Likelihood class used to fit cosmic chronometer data.
    Should be used as a template for other likelihoods as all functions in this class are required in fitting functions.
    
    """
        
    def __init__(self):

        super().__init__('CC_Hubble.dat', 'CC_Hubble.dat', 'cc_dimful')
        
        self.Hfid = 1.
        self.ylabel = r'$H \left( z \right) \ / \ H_{\rm fid}$'  # for plotting
        self.xvar, self.yvar, self.yerr = np.genfromtxt(self.data_file, unpack=True)
        self.xvar += 1
        self.yvar /= self.Hfid
        self.yerr /= self.Hfid
        self.inv_cov = 1 / self.yerr ** 2


    def get_pred(self, zp1, a, eq_numpy, **kwargs):
        """Return the predicted H(z), which is the square root of the functions we are using.
        
        Args:
            :zp1 (float or np.array): 1 + z for redshift z
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :H (float or np.array): the predicted Hubble parameter at redshifts supplied
        
        """
        return np.sqrt(eq_numpy(zp1, *a))


    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function
        
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
        
        """
        
        H = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(H)):
            return np.inf
        nll = np.sum(0.5 * (H - self.yvar) ** 2 * self.inv_cov)  # inv_cov diagonal, so is vector here
        if np.isnan(nll):
            return np.inf
        return nll


class PanthLikelihood(Likelihood):
    """Likelihood class used to fit Pantheon data"""
    
    def __init__(self):

        super().__init__(
            '/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',
            '/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov',
            'panth_dimful',
        )

        self.Hfid = 1.0 * apu.km / apu.s / apu.Mpc
        data = pd.read_csv(self.data_file, delim_whitespace=True)
        origlen = len(data)
        ww = (data['zHD']>0.01)
        zCMB = data['zHD'][ww]
        mu_obs = data['MU_SH0ES'][ww]
        mu_err = data['MU_SH0ES_ERR_DIAG'][ww]  # for plotting
        
        with open(self.cov_file, 'r') as f:
            _ = f.readline()
            n = int(len(zCMB))
            C = np.zeros((n,n))
            ii = -1
            jj = -1
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
            
        self.ylabel = r'$\mu \left( z \right)$'
        self.xvar = zCMB.to_numpy() + 1
        self.yvar = mu_obs.to_numpy()
        self.inv_cov = np.linalg.inv(C)
        self.yerr = mu_err.to_numpy()

        self.mu_const =  astropy.constants.c / self.Hfid / (10 * apu.pc)
        self.mu_const = 5 * np.log10(self.mu_const.to(''))

        self.delta_z = 0.02
        self.min_nz = 10
        self.data_x = None
        self.data_mask = None

    def clear_data(self):
        """Clear data used for numerical integration"""
        self.data_x = None
        self.data_mask = None


    def get_pred(self, zp1, a, eq_numpy, integrated=False):
        """Return the predicted distance modulus from the H^2 function supplied.
        
        Args:
            :zp1 (float or np.array): 1 + z for redshift z
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            :integrated (bool, default=False): whether we previously analytically integrated the function (True) or not (False)
            
        Returns:
            :mu (float or np.array): the predicted distance modulus at redshifts supplied
        
        """

        if integrated:
            dL = eq_numpy(zp1, *a) - eq_numpy(1, *a)
        else:
            if self.data_x is None or self.data_mask is None:
                nx = int(np.ceil((zp1.max() - zp1.min()) / self.delta_z))
                self.data_x = np.concatenate(
                        (np.linspace(1, zp1.min(), self.min_nz),
                        np.linspace(zp1.min() + self.delta_z, zp1.max() + self.delta_z, nx),
                        zp1))
                self.data_x = np.sort(np.unique(self.data_x))
                self.data_mask = np.squeeze(np.array([np.where(self.data_x==d)[0] for d in zp1]))

            if len(a) == 0:
                dL = 1 / np.sqrt(eq_numpy(self.data_x))
            else:
                dL = 1 / np.sqrt(eq_numpy(self.data_x, *a))

            # If prediction is a float, need to make it an array of length = len(self.data_x)
            if np.isscalar(dL):
                dL = np.full(len(self.data_x), dL)

            dL = scipy.integrate.cumulative_trapezoid(dL, x=self.data_x, initial=0)
            dL = dL[self.data_mask]

        dL *= zp1
        mu = 5 * np.log10(dL) + self.mu_const

        return mu


    def negloglike(self, a, eq_numpy, integrated=False):
        """Negative log-likelihood for a given function
        
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
        
        """
        mu_pred = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy, integrated=integrated)
        if not np.all(np.isreal(mu_pred)):
            return np.inf
        nll = 0.5 * np.dot((mu_pred - self.yvar), np.dot(self.inv_cov,(mu_pred - self.yvar)))
        if np.isnan(nll):
            return np.inf
        return nll


    def run_sympify(self, fcn_i, tmax=5, try_integration=True):
        """Sympify a function
        
        Args:
            :fcn_i (str): string representing function we wish to fit to data
            :tmax (float): maximum time in seconds to attempt analytic integration
            :try_integration (bool, default=True): as the likelihood requires an integral, whether to try to analytically integrate (True) or not (False)
            
        Returns:
            :fcn_i (str): string representing function we wish to fit to data (with superfluous characters removed)
            :eq (sympy object): sympy object representing function we wish to fit to data
            :integrated (bool): whether we we able to analytically integrate the function (True) or not (False)
        
        """
        
        fcn_i = fcn_i.replace('\n', '')
        fcn_i = fcn_i.replace('\'', '')

        eq = sympy.sympify(fcn_i,
                    locals={"inv": inv,
                            "square": square,
                            "cube": cube,
                            "sqrt": sqrt,
                            "log": log,
                            "pow": pow,
                            "x": x,
                            "a0": a0,
                            "a1": a1,
                            "a2": a2})

        if try_integration:
            try:
                with time_limit(tmax):
                    eq2 = sympy.integrate(1 / sqrt(eq), x)
                    if eq2.has(sympy.Integral):
                        raise ValueError
                    eq = eq2
                    integrated = True
            except Exception:
                integrated = False
        else:
            integrated = False

        return fcn_i, eq, integrated




class MockLikelihood(Likelihood):
    """Likelihood class used to fit mock cosmic chronometer data
    
    Args:
        :nz (int): number of mock redshifts to use
        :yfracerr (float): the fractional uncertainty on the cosmic chronometer mock we are using
        :data_dir (str, default=None): The path containing the data and cov files
    
    """
        
    def __init__(self, nz, yfracerr, data_dir=None):
        super().__init__(
            '/mock/CC_Hubble_%i_'%nz + str(yfracerr) + '.dat',
            '/mock/CC_Hubble_%i_'%nz + str(yfracerr) + '.dat',
            'mock_%i_'%nz + str(yfracerr),
            data_dir=data_dir
        )

        self.Hfid = 1.
        self.ylabel = r'$H \left( z \right) \ / \ H_{\rm fid}$'  # for plotting
        self.xvar, self.yvar, self.yerr = np.genfromtxt(self.data_file, unpack=True)
        self.xvar += 1
        self.yvar /= self.Hfid
        self.yerr /= self.Hfid
        self.inv_cov = 1 / self.yerr ** 2


    def get_pred(self, zp1, a, eq_numpy, **kwargs):
        """Return the predicted H(z), which is the square root of the functions we are using.
        
        Args:
            :zp1 (float or np.array): 1 + z for redshift z
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :H (float or np.array): the predicted Hubble parameter at redshifts supplied
        
        """
        return np.sqrt(eq_numpy(zp1, *a))


    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function
        
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives H^2
            
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
        
        """
        H = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(H)):
            return np.inf
        nll = np.sum(0.5 * (H - self.yvar) ** 2 * self.inv_cov)  # inv_cov diagonal, so is vector here
        if np.isnan(nll):
            return np.inf
        return nll


class MSE(Likelihood):
    """Likelihood class used to fit a function directly using a MSE
    
    IMPORTANT - MSE is NOT a likelihood in the probabilistic sense.
    It should not be used for MDL calculations as the answer will
    be nonesense since an uncertainty is required for MDL to have meaning.
    
    Args:
        :data_file (str): Name of the file containing the data to use
        :run_name (str): The name to be associated with this likelihood, e.g. 'my_esr_run'
        :data_dir (str, default=None): The path containing the data and cov files
        :fn_set (str, default='core_maths'): The name of the function set to use with the likelihood. Must match one of those defined in ``generation.duplicate_checker``
    
    """

    def __init__(self, data_file, run_name, data_dir=None, fn_set='core_maths'):
        
        super().__init__(data_file, data_file, run_name, data_dir=data_dir, fn_set=fn_set)
        self.ylabel = r'$y$'    # for plotting
        self.xvar, self.yvar, self.yerr = np.loadtxt(self.data_file, unpack=True)
        self.yerr = 0.
        
        warnings.warn("You are using the MSE class. MSE is NOT a likelihood in the probabilistic sense. It should not be used for MDL calculations as the answer will be nonesense since an uncertainty is required for MDL to have meaning.")
        self.is_mse = True # Warning to not use MSE for DL


    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function. Here it is (y-ypred)^2
        Note that this is technically not a log-likelihood, but the function
        name is required to be accessed by other functions.
        
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y
            
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
        
        """

        ypred = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(ypred)):
            return np.inf
        nll = np.mean((ypred - self.yvar) ** 2)
        if np.isnan(nll):
            return np.inf
        return nll


class GaussLikelihood(Likelihood):
    """Likelihood class used to fit a function directly using a Gaussian likelihood
    
    Args:
        :data_file (str): Name of the file containing the data to use
        :run_name (str): The name to be associated with this likelihood, e.g. 'my_esr_run'
        :data_dir (str, default=None): The path containing the data and cov files
        :fn_set (str, default='core_maths'): The name of the function set to use with the likelihood. Must match one of those defined in ``generation.duplicate_checker``
    
    """

    def __init__(self, data_file, run_name, data_dir=None, fn_set='core_maths'):
        
        super().__init__(data_file, data_file, run_name, data_dir=data_dir, fn_set=fn_set)
        self.ylabel = r'$y$'    # for plotting
        self.xvar, self.yvar, self.yerr = np.loadtxt(self.data_file, unpack=True)


    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function.
        
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y
            
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
        
        
        """

        ypred = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if not np.all(np.isreal(ypred)):
            return np.inf
        nll = np.sum(0.5 * (ypred - self.yvar) ** 2 / self.yerr ** 2 + 0.5 * np.log(2 * np.pi) + np.log(self.yerr))
        if np.isnan(nll):
            return np.inf
        return nll
        

class PoissonLikelihood(Likelihood):
    """Likelihood class used to fit a function directly using a Poisson likelihood
    
    Args:
        :data_file (str): Name of the file containing the data to use
        :run_name (str): The name to be associated with this likelihood, e.g. 'my_esr_run'
        :data_dir (str, default=None): The path containing the data and cov files
        :fn_set (str, default='core_maths'): The name of the function set to use with the likelihood. Must match one of those defined in ``generation.duplicate_checker``
    
    """
        
    def __init__(self, data_file, run_name, data_dir=None, fn_set='core_maths'):
        
        super().__init__(data_file, data_file, run_name, data_dir=data_dir, fn_set=fn_set)
        self.ylabel = r'$y$'    # for plotting
        self.xvar, self.yvar = np.loadtxt(self.data_file, unpack=True)
        self.yerr = np.sqrt(self.yvar)
        
    def negloglike(self, a, eq_numpy, **kwargs):
        """Negative log-likelihood for a given function. Here it is a Poisson
        
        Args:
            :a (list): parameters to subsitute into equation considered
            :eq_numpy (numpy function): function to use which gives y
            
        Returns:
            :nll (float): - log(likelihood) for this function and parameters
        
        """

        ypred = self.get_pred(self.xvar, np.atleast_1d(a), eq_numpy)
        if (not np.all(np.isreal(ypred))) or (not np.all(ypred > 0)):
            return np.inf
        nll = np.sum(ypred - self.yvar * np.log(ypred))
        if np.isnan(nll):
            return np.inf
        return nll
    
