ExhaustiveSR
------------

:ExhaustiveSR: Exhaustive Symbolic Regression
:Authors: Deaglan J. Bartlett and Harry Desmond
:Homepage: https://github.com/DeaglanBartlett/ExhaustiveSR 
:Documentation: https://exhaustivesr.readthedocs.io

.. image:: https://readthedocs.org/projects/exhaustivesr/badge/?version=latest
  :target: https://exhaustivesr.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. image:: https://img.shields.io/badge/astro.CO-arXiv%3AXXXX.YYYYY-B31B1B.svg
  :target: https://arxiv.org/abs/XXXX.YYYYY

About
=====

ExhaustiveSR is a symbolic regression algorithm which efficiently 
and systematically finds find all possible equations at fixed complexity 
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
results of the original ExhaustiveSR paper, which applies this method
to a catalogue of cosmic chronometers and the Pantheon+ sample of 
supernovae to learn the Hubble rate as a function of redshift.
The functions generated for this paper are publicly available
AT THIS LINK.

Licence and Citation
====================

Users are required to cite the Exhaustive Symbolic Regression `Paper <https://arxiv.org/abs/XXXX.YYYYY>`_
for which the following bibtex can be used

.. code:: bibtex

  @ARTICLE{2022arXivXXXXYYYYY,
       author = {{Bartlett}, D.~J. and {Desmond}, H. and {Ferreira}, P.~G.},
        title = "{Exhaustive Symbolic Regression}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2022,
        month = nov,
          eid = {arXiv:XXXX.YYYYY},
        pages = {arXiv:XXXX.YYYYY},
  archivePrefix = {arXiv},
       eprint = {XXXX.YYYYY},
  primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXivXXXXYYYYY},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
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
`at this link <https://exhaustivesr.readthedocs.io/>`_

Acknowledgements
================
DJB is supported by the Simons Collaboration on "Learning the Universe" and was supported by STFC and Oriel College, Oxford.
HD is supported by a Royal Society University Research Fellowship (grant no. 211046).

