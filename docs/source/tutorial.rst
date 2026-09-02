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
it does not remove equations or change the ``matches`` mapping. ESR evaluates
at 60 fixed pseudo-random points (``RandomState(42)``), with :math:`x` drawn
from :math:`[0.2, 5]` and each parameter drawn from :math:`[0.5, 3]`. A
candidate collision has the same 60-entry fingerprint after finite values are
formatted as ``%.10e`` (and has non-finite values at the same entries).
Expressions with more than 30% non-finite evaluations are skipped.

Matching numerical fingerprints are useful for locating possible missed
identities, but must not be treated as proof of duplicate models. In
particular, expressions may differ at boundaries or singularities, for
different allowed signs or domains of their parameters, or in their
description-length interpretation. Numerical candidates should be merged
only after an application-specific exact equivalence check.


Choosing a function set
~~~~~~~~~~~~~~~~~~~~~~~

``esr.generation.duplicate_checker`` provides several predefined function
sets. To define another run, add an option near the start of that script:

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


Basic fitting pipeline
~~~~~~~~~~~~~~~~~~~~~~

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


Once you have run this for many complexities, you can plot the pareto front and save it to file using the following function.

.. code-block:: python

	import esr.plotting.plot

	esr.plotting.plot.pareto_plot(likelihood.out_dir, 'pareto.png')

MPI scheduling behaviour
~~~~~~~~~~~~~~~~~~~~~~~~

For MPI runs, ``test_all.main`` uses dynamic rank-0 work dispatch by default
when there are at least two worker ranks and more functions than ranks. This
avoids long idle tails when different expressions take very different amounts
of time to optimise. To reproduce the original static rank partitioning, call
``test_all.main(comp, likelihood, dynamic=False)``. Dynamic runs write the
same final ``negloglike_comp*.dat`` file as static runs, and also write a
temporary ``*.checkpoint.dat`` file during long jobs; downstream stages should
use the final file after the run completes.

Default Fisher scoring and snapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Fisher stage defaults to determinant scoring with eigenbasis snapping:
``use_det_I=True, snap_choice=1``. It writes these choices to the output
directory, and ``match.main`` reads them back so that matching cannot silently
use different settings from the Fisher calculation. This setting diagonalises
the full Hessian, identifies directions with fewer than one precision step,
and maps each such direction to the original parameter with the largest
projection; the description length itself remains in the original parameter
basis. With ``snap_choice=2`` (projected eigenbasis), ESR instead zeros the weak
projected coordinate itself, transforms the retained vector back, and
re-evaluates the likelihood at that point *in the original parameterisation*;
only the snap decision and the description length are expressed in the Hessian
eigenbasis, so the snap and the codelength share a basis. This mode requires
``use_det_I=True``. With
``snap_choice=0``, snapping is assessed independently from each
Hessian diagonal element.

Determinant with diagonal snapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``use_det_I=True`` with ``snap_choice=0`` is allowed, and is the setting to use
if you want to attribute a change to the determinant alone while holding the
published snapping rule fixed. ESR warns when it is used
(``DiagonalSnapDeterminantWarning``): the diagonal test examines one parameter
axis at a time, so it cannot remove an unconstrained direction lying between the
axes, and that direction then stays in `\det H` where the smaller its eigenvalue
the shorter the codelength. Fitting ``x**(a0*a1)``, which is just ``x**a0``
written with a spare parameter, gives a description length of 17.08 under this
pairing against 19.43 for ``x**a0`` itself. Use it for comparison runs, not to
rank a catalogue.

Published diagonal comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare against the published diagonal
parameter-codelength formula, run
``test_all_Fisher.main(comp, likelihood, use_det_I=False, snap_choice=0)``
and then ``match.main(comp, likelihood)``. This comparison uses the diagonal
formula within ESR's current shared fitting pipeline. It is therefore not a
byte-for-byte reproduction of an older ESR run.

Normalised Hessian criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whether a direction counts as unconstrained is judged on the Hessian normalised
by its own diagonal, `D^{-1/2} H D^{-1/2}` for `D = {\rm diag}(H)`, rather than
on the raw eigenvalues. Rescaling a parameter -- a change of units, or writing
`2 a_0` in place of `a_0` -- rescales a row and column of `H` and can make
`\lambda_{\rm min}/\lambda_{\rm max}` arbitrarily small without making the fit
any less determined, so a threshold on the raw spectrum cannot separate a badly
scaled model from a genuinely redundant one. The normalised matrix is unchanged
by that rescaling. A direction whose normalised eigenvalue falls below
``EIGENVALUE_REL_THRESHOLD`` is treated as unconstrained whatever sign it
carries, since finite differencing gives a mathematically flat direction a small
positive or a small negative eigenvalue at random; the same threshold is used to
decide that resolved negative curvature means a saddle, so a fit is never
rejected for curvature it cannot resolve.

Why re-optimisation after snapping is required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Removing an unconstrained direction is mandatory, because the unsnapped
determinant still contains its eigenvalue and the smaller that eigenvalue comes
out the shorter the code it produces. Under ``snap_choice=1`` and ``2`` the
remaining parameters are then re-optimised with the snapped ones held at zero,
since they were fitted alongside the parameter being removed: zeroing an
intercept while leaving the slope where it was would otherwise collapse the
likelihood and make a necessary snap look like a bad one. ``snap_choice=0``
keeps the published behaviour of scoring at the zeroed vector itself.

Likelihood-aware fitted catalogue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``likelihood.run_sympify`` removes or relabels parameters, for example
by applying a likelihood-specific symbolic transformation, ESR can build a
fitted-function catalogue (named ``likelihood_catalogue`` in code identifiers
and output files). The catalogue starts from the simplifier's *unique*
equations and groups them by their exact symbolic transformed expression after
canonical parameter relabelling; one representative is fitted for each
transformed model family, while ``combine_DL`` still uses each generated
expression's own tree complexity for the final description length.

It builds on the simplifier's grouping rather than redoing it from
``all_equations``, because a likelihood transformation cannot split one of those
families: their members differ only by a parameter redefinition, which the
transformation carries through with them. It can only merge families further,
which is the whole point. Starting from every generated tree would instead
re-admit the redundant parameterisations the simplifier removed --
``pow(x,(a0*a1))`` alongside ``pow(x,a0)`` -- and fit them as separate models,
where a near-degenerate Hessian earns such a form a shorter parametric
codelength than the family it is a redundant copy of. If this catalogue changes,
the code will fail loudly on stale row counts rather than reusing incompatible
``test_all`` or Fisher outputs; rerun ``test_all.main`` and
``test_all_Fisher.main`` with the current likelihood/settings.

Building this catalogue is **opt-in**: ``use_likelihood_catalogue`` defaults to
``False`` on ``Likelihood``, so the built-in likelihoods (which do not change the
parameter layout) skip the build entirely, avoiding an expensive all-equations
transformation pass at high complexity. A likelihood whose ``run_sympify``
genuinely removes or relabels parameters should set
``use_likelihood_catalogue = True`` on its class or instance. Such a likelihood
should also set a ``catalogue_transform_version`` string (bumped whenever
``run_sympify`` changes): the catalogue is cached across runs only when a version
is present, because ESR otherwise cannot prove from its best-effort probe
fingerprint that the transformation is unchanged, and so rebuilds every time to
stay correct. A rebuild is also forced when the equation set changes, the
transformation version or fingerprint changes, ``tmax`` or the integration
setting changes, or the cached build had transform failures. (The catalogue does
not depend on the Fisher scoring options ``use_det_I``/``snap_choice``, so
changing those does not rebuild it.)

Worked notebook: likelihood-aware fitted catalogue
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
	:maxdepth: 1

	notebooks/likelihood_catalogue

The above worked example defines a
"shape-only" likelihood whose ``run_sympify`` divides out `f(1)`, so the data
constrain only the shape of the function -- the same situation as a dark-energy
density known only up to its value today, where the model is `f_{\rm DE} =
g/g(1)`. Any overall amplitude is then invisible to the likelihood, so
``a0*x**2``, ``2*x**2`` and ``x**2`` are one model rather than three. The script
runs the complexity-5 catalogue both ways on the same data and takes about a
minute on one core.

Deduplicating after the transformation removes 37 of the 131 fits, and the run
comes out faster rather than slower despite the extra transformation pass over
the unique equations. The equations the transformation has made equivalent stop
being reported separately: in the notebook output ``2*x**2`` and ``x**2`` have
identical description lengths without the catalogue, because they *are* the same
model here, and only one of them survives with it on.

The catalogue changes how much work is done, not what the answer is. The best
description length is 16.7905 either way, and of the equations ranked in both
runs the overwhelming majority agree to within 0.01 nats; where one does move it
is because fitting a model once rather than several times gave the optimiser a
better shot at its maximum likelihood.


Comparing the Fisher scoring options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Worked notebook: Fisher scoring options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
	:maxdepth: 1

	notebooks/fisher_scoring_options

The above example runs the complexity 5 ``core_maths``
catalogue on mock data once per setting and prints what changes between them. It
takes about half a minute on one core, and generates the catalogue first if it
is not already present.

The reason for scoring with the determinant is that a description length should
not depend on how an equation happens to be written. ESR's complexity 5
catalogue keeps three algebraic forms of the same two-parameter linear family
--- ``a0 + a1*x``, ``a0*(a1 + x)`` and ``a0*(a1 - x)`` --- which fit any dataset
identically and have the same functional codelength. On data generated from
`y = 3 + 1.7 x` the notebook gives the three forms parametric codelengths of
7.69704, 8.11305 and 8.11305 under the published diagonal formula, against
6.68189 for all three under the determinant.

The diagonal formula separates the three by 0.416 nats, and so picks a winner
among them on grounds that have nothing to do with the data; the determinant
gives all three the same parametric codelength. The gap between the two is not
arbitrary, since for a Fisher matrix `I` with correlation matrix `C`

.. math::

	\frac{1}{2} \sum_j \ln I_{jj} - \frac{1}{2} \ln \det I =
	-\frac{1}{2} \ln \det C .

This is non-negative by Hadamard's inequality and is fixed by the choice of
parameters, so it carries no information about either the model or the data. The
example evaluates both sides for ``a0 + a1*x``, whose Fisher matrix is analytic,
and gets 1.015148 nats either way.

The same run shows why ``snap_choice=1`` rather than ``2`` is the default: mode
2 keeps a different set of forms of the family, and still spreads their
parametric codelengths over 0.29 nats, because the coordinate it zeros lives in
the Hessian eigenbasis rather than in the parameters themselves.

The example ends by fitting ``a0 + a1*x`` to data generated with and without an
intercept, to show snapping on its own. With a zero intercept, ``a0`` is
consistent with zero to within its own encoding precision, so it is snapped away
and the description length falls from 151.66 to 147.79. All three snapping modes
agree there, as they should: an unwanted intercept lies along a parameter axis,
which is the one case the diagonal test of ``snap_choice=0`` handles as well as
the eigendecomposition. The modes part company when the poorly constrained
direction lies between the parameter axes instead.


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
