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
degenerate Hessian direction makes snapping mandatory. The published diagonal
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
ESR builds a fitted-function catalogue automatically. Raw expressions are
deduplicated after that likelihood-specific symbolic transformation. ESR fits
one representative of each transformed symbolic model family, then maps the
result back to all raw expressions so their original tree complexities can
still enter the final description length.

Likelihoods that evaluate generated expressions directly, without changing
their fitted model family or parameter layout, should leave this catalogue
inactive. If a custom likelihood has a ``run_sympify`` method only for parsing
or diagnostics and its stored ``negloglike_comp*.dat`` files correspond to the
raw unique catalogue, set ``use_likelihood_catalogue = False`` on the
likelihood class or instance before running ``test_all_Fisher``/``match``.
(The code identifiers and the ``likelihood_catalogue_comp*`` output files
retain the older ``likelihood_catalogue`` name for this feature.)

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
  	author       = {Bartlett, Deaglan J. and
                  Desmond, Harry and
                  Ferreira, Pedro G.},
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
