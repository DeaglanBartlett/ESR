# ExhaustiveSR

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

## Documentation
The documentation for this project can be found AT THIS LINK

## Users are required to cite the Exhaustive Symbolic Regression Papers:

* arXiv:PAPER1 (Paper 1)

## Contributors
Below is a list of contributors to this repository. 
* [Deaglan Bartlett](https://github.com/DeaglanBartlett) (CNRS & Sorbonne Universit\'{e}, Institut d’Astrophysique de Paris)
* [Harry Desmond](https://github.com/harrydesmond) (Institute of Cosmology & Gravitation, University of Portsmouth)

## Examples

To run the Pantheon example from Paper 1, one must download the
[Pantheon data](https://github.com/PantheonPlusSH0ES/DataRelease)
and place in the 'data' directory.

## Acknowledgements
DJB is supported by the Simons Collaboration on ``Learning the Universe'' and was supported by STFC and Oriel College, Oxford.
HD is supported by a Royal Society University Research Fellowship (grant no. 211046).
