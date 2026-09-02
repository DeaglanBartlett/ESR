ESR
----

:ESR: Exhaustive Symbolic Regression
:Authors: Deaglan J. Bartlett and Harry Desmond
:Homepage: https://github.com/DeaglanBartlett/ESR 
:Documentation: https://esr.readthedocs.io
:Pre-computed function sets: https://doi.org/10.5281/zenodo.7339113

.. image:: https://readthedocs.org/projects/esr/badge/?version=latest
  :target: https://esr.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/astro.CO-arXiv%32211.11461-B31B1B.svg
  :target: https://arxiv.org/abs/2211.11461

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7339113.svg
   :target: https://doi.org/10.5281/zenodo.7339113

\

.. image:: https://github.com/DeaglanBartlett/ESR/actions/workflows/build.yml/badge.svg
  :target: https://github.com/DeaglanBartlett/ESR/actions/workflows/build.yml
  :alt: Build Status

.. image:: https://img.shields.io/codecov/c/github/DeaglanBartlett/ESR
  :target: https://app.codecov.io/gh/DeaglanBartlett/ESR
  :alt: Coverage

About
=====

ESR (Exhaustive Symbolic Regression) is a symbolic regression algorithm which efficiently 
and systematically finds all possible equations at fixed complexity 
(defined to be the number of nodes in its tree representation)
given a set of basis functions.
This is achieved by identifying the unique equations, so that one
minimises the number of equations which one would have to fit to data.
These unique equations are fitted to data and the results applied to
the full list of equations, so we know the maximum likelihood parameters
of every equation generated.
We then select the best function using the minimum description length principle.
By considering all equations, this procedure is guaranteed 
to find the true optimum for a
given complexity and basis function set.

We provide all required code and plotting routines to reproduce the 
results of the original ESR paper, which applies this method
to a catalogue of cosmic chronometers and the Pantheon+ sample of 
supernovae to learn the Hubble rate as a function of redshift.
The functions generated for this paper and an additional function
set containing exp, sqrt and square are 
`publicly available <https://doi.org/10.5281/zenodo.7339113>`_.

Installation
=============

To install ESR and its dependencies in a new virtual environment, run

.. code:: bash

	python3 -m venv esr_env
	source esr_env/bin/activate
	git clone git@github.com:DeaglanBartlett/ESR.git
	cd ESR
	pip install -e .

If you are unable to clone the repo with the above, try the https version instead

.. code:: bash

	git clone https://github.com/DeaglanBartlett/ESR.git

Fisher scoring and parameter snapping
=====================================

The fitting pipeline computes Fisher matrices after the maximum-likelihood
fit and uses them in the parametric part of the minimum-description-length
score. The default Fisher scoring uses the positive-definite full Hessian
determinant with eigenbasis snapping:

.. code:: python

	esr.fitting.test_all_Fisher.main(comp, likelihood,
	                                 use_det_I=True, snap_choice=1)

This is the recommended setting for new runs because it accounts for
parameter correlations and rejects non-positive-definite Hessians.
``examples/fisher_scoring_options.py`` runs a small catalogue under each
setting and shows what changes; see "Comparing the Fisher scoring options"
in the tutorial for its output.

This full-Hessian determinant encoding (in place of the diagonal Fisher
approximation of the original ESR paper) and the eigenbasis treatment of
weakly constrained directions are motivated by the rotated-Fisher
description-length approach of Kronberger, Olivetti de França, Bartlett,
Desmond & Ferreira (2026),
`"Guiding Multi-Objective Genetic Programming with Description Length Improves
Symbolic Regression Solutions" <https://arxiv.org/abs/2605.22374>`_. ESR's
explicit zero-and-re-evaluate snapping is not a byte-for-byte reproduction of
that paper's derivation, but uses the same rotated-Fisher idea; we refer to it
for the underlying motivation.

For the :math:`k` retained parameters, the determinant score is

.. math::

  L_{\mathrm{par}}=-\frac{k}{2}\ln 3+\frac{1}{2}\ln\det H+
  \sum_i\max\left[\ln|\theta_i|,\frac{1}{2}\ln\left(\frac{12}{H_{ii}}\right)\right],

where :math:`H` is the Hessian of the negative log-likelihood at the fitted
point. The determinant is used only when the active Hessian is positive
definite.

With ``snap_choice=1``, ESR diagonalises the full Hessian, finds directions
with fewer than one precision step, and maps each such direction back to the
original parameter with the largest projection. The description length is
still evaluated in the original parameter basis. With ``snap_choice=0``, the
corresponding decision uses each diagonal element :math:`H_{ii}` independently.
Snapping is retained only if it improves the description length, except when a
degenerate Hessian direction makes snapping mandatory.

With ``snap_choice=2`` (projected eigenbasis), ESR instead zeros the weak
*projected* coordinate :math:`b_j=(V^{\top}\theta)_j` itself and transforms the
retained vector back with :math:`\theta'=Vb`. The likelihood is then re-evaluated
at :math:`\theta'` **in the original parameterisation** (the back-transform is
what makes the snapped point comparable to the unsnapped fit); only the *snap
decision* and the *codelength* are expressed in the eigenbasis of :math:`H`.
There the Hessian is diagonal (its eigenvalues :math:`\lambda_j`), so the volume
term becomes :math:`\frac{1}{2}\sum_j\ln\lambda_j` and the precision floor uses
:math:`\frac{1}{2}\ln\left(12/\lambda_j\right)` against :math:`|b_j|`. The snap
and the codelength are therefore expressed in the same basis, in contrast to
``snap_choice=1``, which keeps the floor term in the original :math:`H_{ii}`
coordinates. This mode requires ``use_det_I=True``. Because the eigenvectors of
a repeated (or nearly repeated) eigenvalue are not unique, both the retained
directions and the ``snap_choice=2`` codelength become basis-sensitive when
:math:`H` has clustered eigenvalues; ESR emits a warning in that case. This is
not unique to mode 2 -- the per-coordinate precision floor makes every snap mode
coordinate-dependent, so ``snap_choice=1`` is not strictly
re-parameterisation-invariant either, but it is always evaluated in the fixed
original basis and so is the more predictable choice for clustered spectra.

``use_det_I=True`` may also be paired with ``snap_choice=0``, which holds the
published snapping rule fixed and so isolates the effect of the determinant on
its own. ESR warns when it is used (``DiagonalSnapDeterminantWarning``), because
it is not safe for ranking a catalogue: diagonal snapping tests one parameter
axis at a time and so cannot remove an unconstrained direction lying between the
axes. Such a direction stays in :math:`\det H`, where the smaller its eigenvalue
the shorter the codelength, so a redundant parameterisation can score better than
the model it is a redundant copy of. Fitting ``x**(a0*a1)`` -- which is just
``x**a0`` written with a spare parameter -- gives a description length of 17.08
under this pairing against 19.43 for ``x**a0`` itself, while the published
diagonal formula correctly charges it 28.12 and the default settings drop it.

The published diagonal
Fisher approximation remains available for comparison:

.. code:: python

	esr.fitting.test_all_Fisher.main(comp, likelihood,
	                                 use_det_I=False, snap_choice=0)

The comparison uses

.. math::

  L_{\mathrm{par}}=-\frac{k}{2}\ln 3+
  \sum_i\left[\frac{1}{2}\ln H_{ii}+\ln|\theta_i|\right].

It still runs through ESR's current shared fitting pipeline, so it is not a
byte-for-byte reproduction of an older ESR run.

Likelihood-aware fitted catalogue
---------------------------------

If a likelihood's ``run_sympify`` method removes or relabels parameters
or otherwise changes the symbolic expression supplied to the likelihood,
ESR can build a fitted-function catalogue. The simplifier's unique equations
are deduplicated again after that likelihood-specific symbolic transformation;
ESR fits one representative of each transformed symbolic model family, then maps
the result back to all generated expressions so their original tree complexities
can still enter the final description length. It refines the simplifier's own
grouping rather than redoing it from every generated tree: a transformation
cannot split one of those families (their members differ only by a parameter
redefinition, which it carries through with them), it can only merge them
further, and starting from every tree would re-admit the redundant
parameterisations the simplifier had already removed.

This catalogue is opt-in. The built-in likelihoods evaluate the generated
expressions directly (or, for Pantheon, integrate them) without changing the
parameter layout, so they set ``use_likelihood_catalogue = False`` and skip
building it -- which also avoids a one-time transformation pass over the unique
equations that becomes expensive at high complexity. A custom likelihood whose
``run_sympify`` genuinely removes or relabels parameters must set
``use_likelihood_catalogue = True`` (on the class or instance) to enable the
catalogue; otherwise ESR fits the raw expressions with the wrong parameter
count. (The code identifiers and the ``likelihood_catalogue_comp*`` output
files retain the older ``likelihood_catalogue`` name for this feature.)

``examples/likelihood_catalogue.py`` is a worked example, and
``examples/fisher_scoring_options.py`` covers the Fisher scoring options; see
the tutorial for their output.

The built catalogue is cached, keyed on the equation set and a best-effort
fingerprint of the transform evaluated on a few fixed probe expressions. That
fingerprint can miss a transform change that only affects expressions unlike the
probes, so for **guaranteed** cache invalidation set a
``catalogue_transform_version`` attribute on the likelihood and change it
whenever you change ``run_sympify``. A transforming likelihood that supplies no
``catalogue_transform_version`` is not cached at all -- it is rebuilt on every
call (and ESR warns) rather than risk reusing a stale mapping.

Numerical duplicate diagnostic
------------------------------

Numerical duplicate checks are available only as an opt-in diagnostic:

.. code:: python

	esr.generation.duplicate_checker.main(
	    runname, comp, diagnose_numerical_duplicates=True)

The diagnostic evaluates each expression at 60 fixed pseudo-random points
(``RandomState(42)``): :math:`x` is sampled from :math:`[0.2, 5]` and each
parameter from :math:`[0.5, 3]`. It hashes the 60 results after formatting
each finite value as ``%.10e``; a candidate collision requires exactly the
same formatted fingerprint, including the positions of non-finite values.
Expressions with more than 30% non-finite evaluations are not fingerprinted.

The diagnostic writes candidate fingerprint collisions but does not remove
or remap equations. Matching numerical fingerprints should be treated as
possible missed identities, not as proof of model equivalence.

Licence and Citation
====================

Users are required to cite the Exhaustive Symbolic Regression `Paper <https://arxiv.org/abs/2211.11461>`_
for which the following bibtex can be used

.. code:: bibtex

	@ARTICLE{Bartlett_2022,
  		author={Bartlett, Deaglan J. and Desmond, Harry and Ferreira, Pedro G.},
  		journal={IEEE Transactions on Evolutionary Computation}, 
  		title={Exhaustive Symbolic Regression}, 
  		year={2024},
  		volume={28},
  		number={4},
  		pages={950-964},
  		keywords={Mathematical models;Complexity theory;Optimization;Numerical models;Biological system modeling;Standards;Search problems;Cosmology data analysis;minimum description length;model selection;symbolic regression (SR)},
  		doi={10.1109/TEVC.2023.3280250},
  		archivePrefix = "arXiv",
  		eprint = {2211.11461},
  		primaryClass = "astro-ph.CO",
  		adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv221111461B},
  		adsnote = {Provided by the SAO/NASA Astrophysics Data System}
	}

If the user uses the `pre-computed function sets <https://doi.org/10.5281/zenodo.7339113>`_ 
then they must also cite

.. code:: bibtex

	@dataset{bartlett_deaglan_j_2022_7339113,
  	author       = {Bartlett, Deaglan J. and Desmond, Harry and Ferreira, Pedro G.},
  	title        = {Exhaustive Symbolic Regression Function Sets},
  	month        = nov,
  	year         = 2022,
  	note         = {{DJB is supported by the Simons Collaboration on 
                   ``Learning the Universe'' and was supported by
                   STFC and Oriel College, Oxford. HD is supported by
                   a Royal Society University Research Fellowship
                   (grant no. 211046). PGF acknowledges support from
                   European Research Council Grant No: 693024 and the
                   Beecroft Trust.}},
  	publisher    = {Zenodo},
  	doi          = {10.5281/zenodo.7339113},
  	url          = {https://doi.org/10.5281/zenodo.7339113}
	}

The software is available on the MIT licence:

Copyright 2022 Deaglan J. Bartlett

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Contributors
============
Below is a list of contributors to this repository. 

`Deaglan Bartlett <https://github.com/DeaglanBartlett>`_ (CNRS & Sorbonne Université, Institut d’Astrophysique de Paris and Astrophysics, University of Oxford)

`Harry Desmond <https://github.com/harrydesmond>`_ (Institute of Cosmology & Gravitation, University of Portsmouth)

Examples
========

To run the Pantheon example from Paper 1, one must download the
`Pantheon data <https://github.com/PantheonPlusSH0ES/DataRelease>`_
and place in the 'data' directory.

Documentation
=============

The documentation for this project can be found
`at this link <https://esr.readthedocs.io/>`_

Acknowledgements
================
DJB is supported by the Simons Collaboration on "Learning the Universe" and was supported by STFC and Oriel College, Oxford.
HD is supported by a Royal Society University Research Fellowship (grant no. 211046).
