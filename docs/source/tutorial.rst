.. default-role:: math

Tutorial
========

Function Generation
-------------------

To generate all functions at a given complexity (here complexity 5), one simply needs to run the following.

.. code-block:: python

	import esr.generation.duplicate_checker

	runname = 'core_maths'
	comp = 5
	esr.generation.duplicate_checker.main(runname, comp)

Numerical duplicate diagnostic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic simplification is used to generate the fitted catalogue. You may
optionally generate a diagnostic list of expressions that agree on a fixed
set of numerical evaluation points:

.. code-block:: python

	esr.generation.duplicate_checker.main(
		runname, comp, diagnose_numerical_duplicates=True)

This writes ``numerical_duplicate_candidates_<comp>.txt`` inside the
corresponding function-library directory. The report is exploratory only:
it does not remove equations or change the ``matches`` mapping.

Matching numerical fingerprints are useful for locating possible missed
identities, but must not be treated as proof of duplicate models. In
particular, expressions may differ at boundaries or singularities, for
different allowed signs or domains of their parameters, or in their
description-length interpretation. Numerical candidates should be merged
only after an application-specific exact equivalence check.


In ``esr.generation.duplicate_checker`` we have  predefined a few sets of functions which we believe would be useful. However, one simply needs to add another option to the start of that script to define a new run:

.. code-block:: python

	if runname == 'keep_duplicates':
        	basis_functions = [["x", "a"],  # type0
                	["square", "exp", "inv", "sqrt_abs", "log_abs"],  # type1
                	["+", "*", "-", "/", "pow"]]  # type2
    	elif runname == 'core_maths':
        	basis_functions = [["x", "a"],  # type0
                	["inv"],  # type1
                	["+", "*", "-", "/", "pow"]]  # type2
    	elif runname == 'ext_maths':
        	basis_functions = [["x", "a"],  # type0
                	["inv", "sqrt_abs", "square", "exp"],  # type1
                	["+", "*", "-", "/", "pow"]]  # type2
    	elif runname == 'osc_maths':
        	basis_functions = [["x", "a"],  # type0
                	["inv", "sin"],  # type1
                	["+", "*", "-", "/", "pow"]]  # type2

where type 0, 1 and 2 functions are nullary, unary, and binary, respectively.

Fitting to a dataset
--------------------

Suppose we have already generated the equations required for the ``CCLikelihood`` class.
In the following we show the steps that are required to fit the complexity 5 functions to these data.
The various ``fitting`` functions rely on the output of the previous script, so the order cannot change.


.. code-block:: python

	import esr.fitting.test_all
	import esr.fitting.test_all_Fisher
	import esr.fitting.match
	import esr.fitting.combine_DL
	import esr.fitting.plot
	from esr.fitting.likelihood import CCLikelihood

	comp = 5
	likelihood = CCLikelihood()

	esr.fitting.test_all.main(comp, likelihood)
	esr.fitting.test_all_Fisher.main(comp, likelihood)
	esr.fitting.match.main(comp, likelihood)
	esr.fitting.combine_DL.main(comp, likelihood)
	esr.fitting.plot.main(comp, likelihood)

For MPI runs, ``test_all.main`` uses dynamic rank-0 work dispatch by default
when there are at least two worker ranks and more functions than ranks. This
avoids long idle tails when different expressions take very different amounts
of time to optimise. To reproduce the original static rank partitioning, call
``test_all.main(comp, likelihood, dynamic=False)``. Dynamic runs write the
same final ``negloglike_comp*.dat`` file as static runs, and also write a
temporary ``*.checkpoint.dat`` file during long jobs; downstream stages should
use the final file after the run completes.

The Fisher stage defaults to determinant scoring with eigenbasis snapping:
``use_det_I=True, snap_choice=1``. It writes these choices to the output
directory, and ``match.main`` reads them back so that matching cannot silently
use different settings from the Fisher calculation. To compare against the
published diagonal parameter-codelength formula, run
``test_all_Fisher.main(comp, likelihood, use_det_I=False, snap_choice=0)``
and then ``match.main(comp, likelihood)``. This comparison uses the diagonal
formula, but still includes current fixes in expression handling and Fisher
validation, so it is not a byte-for-byte reproduction of an older ESR run.

When ``likelihood.run_sympify`` removes or relabels parameters, for example
through model normalisation, ESR builds a likelihood-aware catalogue. Raw
equations are grouped by their exact symbolic transformed expression after
canonical parameter relabelling; one raw representative is fitted for each
transformed model family, while ``combine_DL`` still uses the raw expression's
tree complexity for the final description length. If this catalogue changes,
the code will fail loudly on stale row counts rather than reusing incompatible
``test_all`` or Fisher outputs; rerun ``test_all.main`` and
``test_all_Fisher.main`` with the current likelihood/settings.

Do not use the likelihood-aware catalogue for likelihoods that evaluate ESR
expressions directly and whose fitted rows already correspond to the raw
``unique_equations`` catalogue. Such likelihoods can opt out by defining
``use_likelihood_catalogue = False`` on the likelihood class or instance. This
is appropriate when ``run_sympify`` parses the expression but does not change
which model family was fitted or how its parameters are laid out.

Once you have run this for many complexities, you can plot the pareto front and save it to file using the following function.

.. code-block:: python

	import esr.plotting.plot

	esr.plotting.plot.pareto_plot(likelihood.out_dir, 'pareto.png')



Fitting a single function
-------------------------

One may wish to fit a single function to your data, instead of the full library produced during the function generation step.
For example, suppose we wish to fit a `\Lambda` CDM expansion history to the cosmic chronometer data.
This function is `y(x) = \theta_0 + \theta_1 x^3` which can be represented as the tree
`[+, \theta_0, \times, \theta_1, {\rm pow}, x, 3]` although we will rewrite `\theta_0` as ``a0``, `\theta_1` as ``a1`` and `\times` as ``*``.
The following script initially loads the cosmic chronometer data then fits the function to this data, returning the negative log-likelihood and the description length.

.. code-block:: python

	from esr.fitting.fit_single import single_function
	from esr.fitting.likelihood import CCLikelihood

	cc_like = CCLikelihood()

	labels = ["+", "a0", "*", "a1", "pow", "x", "3"]
	basis_functions = [["x", "a"],  # type0
			["inv"],  # type1
			["+", "*", "-", "/", "pow"]]  # type2

	logl_lcdm_cc, dl_lcdm_cc = single_function(labels, 
							basis_functions, 
							cc_like, 
							verbose=True)


One can also fit the function directly from writing it as a string. This will convert
the string to a list of labels, which are also returned. Note that this conversion
if not guaranteed to produce the tree representation with the shortest description
length, but does provide an upper limit on the DL of a function.

.. code-block:: python

        from esr.fitting.fit_single import fit_from_string
        from esr.fitting.likelihood import CCLikelihood

        cc_like = CCLikelihood()

        basis_functions = [["x", "a"],  # type0
                        ["inv"],  # type1
                        ["+", "*", "-", "/", "pow"]]  # type2

        logl_lcdm_cc, dl_lcdm_cc, labels = fit_from_string("a0 + a1 * x ** 3",
                                                        basis_functions,
                                                        cc_like,
                                                        verbose=True)


Custom Likelihoods
------------------

To fit a function to your own data, one must create an alternative likelihood using the parent class ``esr.fitting.likelihood.Likelihood``. In the ``__init__()`` for this likelihood, you must define ``xvar``, ``yvar`` and ``yerr`` (the x, y and error on y variables) and a function ``negloglike(self, a, eq_numpy, **kwargs)`` which returns the negative log-likelihood.

For example, a Gaussian likelihood can be defined as

.. code-block:: python

	from esr.fitting.likelihood import Likelihood
	import numpy as np
	import os

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

although note that this is already included as ``esr.fitting.likelihood.GaussLikelihood``.
If you want to use a different set of functions than ``core_maths`` then this can be passed using
the ``fn_set`` argument.

We can then combine the above code with the below to fit a mock dataset

.. code-block:: python

	import esr.fitting.test_all
        import esr.fitting.test_all_Fisher
        import esr.fitting.match
        import esr.fitting.combine_DL
        import esr.fitting.plot

	np.random.seed(123)
	x = np.random.uniform(0.1, 5, 100)
	y = 0.5 * x ** 2
	yerr = np.full(x.shape, 1.0)
	y = y + yerr * np.random.normal(size=len(x))
	np.savetxt('data.txt', np.array([x, y, yerr]).T)
	likelihood = GaussLikelihood('data.txt', 'gauss_example', data_dir=os.getcwd())

	comp = 5

	esr.fitting.test_all.main(comp, likelihood)
	esr.fitting.test_all_Fisher.main(comp, likelihood)
	esr.fitting.match.main(comp, likelihood)
	esr.fitting.combine_DL.main(comp, likelihood)
	esr.fitting.plot.main(comp, likelihood)


We also have a Poisson likelihood already implemented, which can be run as

.. code-block:: python
	
	from esr.fitting.likelihood import PoissonLikelihood
	import numpy as np
	import os

	import esr.fitting.test_all
	import esr.fitting.test_all_Fisher
	import esr.fitting.match
	import esr.fitting.combine_DL
	import esr.fitting.plot

	np.random.seed(123)
	x = np.random.uniform(0.1, 5, 100)
	y = 0.5 * x ** 2
	y = np.random.poisson(y)
	np.savetxt('data.txt', np.array([x, y]).T)
	likelihood = PoissonLikelihood('data.txt', 'poisson_example', data_dir=os.getcwd())

	comp = 5

        esr.fitting.test_all.main(comp, likelihood)
        esr.fitting.test_all_Fisher.main(comp, likelihood)
        esr.fitting.match.main(comp, likelihood)
        esr.fitting.combine_DL.main(comp, likelihood)
        esr.fitting.plot.main(comp, likelihood)
