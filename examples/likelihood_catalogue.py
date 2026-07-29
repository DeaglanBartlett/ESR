"""
Worked example of the likelihood-aware fitted catalogue.

Some likelihoods do not see the function ESR generated. They see a transformed
version of it, and the transformation is part of the model: the cosmological
case is a dark-energy density known only up to its value today, so the data
constrain the shape f_DE = g/g(1) rather than g itself. The likelihood here does
the same thing to a Gaussian fit -- its ``run_sympify`` divides out f(1), so only
the shape of f matters.

That has a consequence for the equation catalogue. ESR's simplifier deduplicated
the generated equations as they were *written*, but after the transformation many
of those distinct equations describe the same model: any overall amplitude has
become invisible, so ``a0*x**2``, ``2*x**2`` and ``x**2`` are one model, not
three. Fitting them separately is wasted work, and reporting them separately
clutters the ranking with copies of the same answer.

Setting ``use_likelihood_catalogue = True`` makes ESR deduplicate again, after
the transformation, and fit one representative per transformed model. This script
runs the complexity-5 ``core_maths`` catalogue both ways on the same data and
compares them.

The catalogue refines the simplifier's grouping rather than redoing it from
every generated tree. A transformation cannot split one of the simplifier's
families -- their members differ only by a parameter redefinition, which the
transformation carries through with them -- so it can only merge families
further. Starting from every generated tree would instead re-admit the redundant
parameterisations the simplifier had removed.

It takes about a minute on one core, and generates the function catalogue first
if it is not already present. Run it with ``mpirun`` for the fitting to be shared
across ranks.
"""

import os
import shutil
import time
import numpy as np
import sympy
from mpi4py import MPI

import esr.generation.duplicate_checker
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
from esr.fitting.likelihood import GaussLikelihood
from esr.fitting.sympy_symbols import x as xsym

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

comp = 5
run_name = 'likelihood_catalogue'
work_dir = os.path.join(os.getcwd(), run_name)
data_file = 'shape_data.txt'


class ShapeLikelihood(GaussLikelihood):
    """Gaussian likelihood that sees only the shape f(x)/f(1) of a function.

    ``use_likelihood_catalogue`` is opt-in and defaults to False on
    ``Likelihood``, because building the catalogue costs a transformation pass
    over the unique equations and the built-in likelihoods do not need it. A
    likelihood like this one, whose ``run_sympify`` genuinely changes the model,
    should turn it on. ``catalogue_transform_version`` should be set alongside it
    and bumped whenever ``run_sympify`` changes: without a version ESR cannot
    prove a cached catalogue is still valid, so it rebuilds on every call.
    """

    use_likelihood_catalogue = True
    catalogue_transform_version = 1

    def run_sympify(self, fcn_i, tmax=5, try_integration=False):
        fcn_i, eq, integrated = super().run_sympify(
            fcn_i, tmax=tmax, try_integration=try_integration)
        try:
            normalisation = eq.subs(xsym, 1)
            if normalisation.is_number and normalisation == 0:
                return fcn_i, eq, integrated
            eq = sympy.cancel(sympy.simplify(eq / normalisation))
        except Exception:
            #  A function that cannot be normalised is left as it is; the
            #  catalogue build reports any such failures.
            pass
        return fcn_i, eq, integrated


#  (1) Mock data whose shape is x**2
if rank == 0:
    os.makedirs(work_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    xvar = np.linspace(0.5, 3.0, 100)
    yerr = np.full_like(xvar, 0.3)
    yvar = xvar ** 2 + rng.normal(scale=yerr)
    np.savetxt(os.path.join(work_dir, data_file),
               np.array([xvar, yvar, yerr]).T)
comm.Barrier()

#  (2) Generate the catalogue if it is not already on disk
probe = ShapeLikelihood(data_file, run_name, data_dir=work_dir,
                        base_out_dir=work_dir)
if rank == 0:
    if not os.path.isfile(os.path.join(probe.fn_dir, f'compl_{comp}',
                                       f'unique_equations_{comp}.txt')):
        for c in range(1, comp + 1):
            esr.generation.duplicate_checker.main('core_maths', c)
comm.Barrier()

#  (3) Run the pipeline with the catalogue off and on
summary = {}
for use_catalogue in [False, True]:
    tag = 'with' if use_catalogue else 'without'
    likelihood = ShapeLikelihood(data_file, tag, data_dir=work_dir,
                                 base_out_dir=work_dir)
    likelihood.use_likelihood_catalogue = use_catalogue

    start = time.time()
    esr.fitting.test_all.main(comp, likelihood)
    elapsed = time.time() - start

    esr.fitting.test_all_Fisher.main(comp, likelihood,
                                     use_det_I=True, snap_choice=1)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)

    if rank == 0:
        n_fitted = sum(1 for _ in open(
            esr.fitting.test_all.function_catalogue_path(
                comp, likelihood, unique=True)))
        shutil.copy(os.path.join(likelihood.out_dir, f'final_{comp}.dat'),
                    os.path.join(work_dir, f'final_{comp}_{tag}.dat'))
        summary[use_catalogue] = (n_fitted, elapsed)
comm.Barrier()

if rank != 0:
    raise SystemExit


def load_ranking(tag):
    """Description length and its terms, in rank order"""
    rows = []
    with open(os.path.join(work_dir, f'final_{comp}_{tag}.dat'), 'r') as f:
        for line in f:
            parts = line.rstrip('\n').split(';')
            if len(parts) < 7:
                continue
            rows.append({'rank': int(parts[0]), 'eq': parts[1],
                         'DL': float(parts[2]), 'negloglike': float(parts[4]),
                         'codelen': float(parts[5])})
    return rows


rankings = {tag: load_ranking(tag) for tag in ['without', 'with']}

print('\n=== How much is fitted ===')
for use_catalogue in [False, True]:
    n_fitted, elapsed = summary[use_catalogue]
    print(f'    use_likelihood_catalogue={str(use_catalogue):<5}  '
          f'{n_fitted:3d} functions fitted   test_all took {elapsed:5.1f} s')

#  (4) The top of each ranking. Without the catalogue the same model appears
#  several times over, once per equation that the transformation has made
#  equivalent; the description lengths are identical because they *are* the same
#  model.
print('\n=== Top of the ranking ===')
for tag in ['without', 'with']:
    print(f'\n{tag} the catalogue:')
    for row in rankings[tag][:6]:
        print(f'    rank {row["rank"]:2d}  {row["eq"]:<26} '
              f'L = {row["DL"]:8.4f}   -log(L) = {row["negloglike"]:7.4f}')

#  (5) The catalogue changes how much work is done, not what the answer is.
best = {tag: rankings[tag][0] for tag in ['without', 'with']}
shared = ({row['eq']: row for row in rankings['without']},
          {row['eq']: row for row in rankings['with']})
common = sorted(set(shared[0]) & set(shared[1]))
differences = np.array([shared[1][eq]['DL'] - shared[0][eq]['DL']
                        for eq in common])
print('\n=== Is it the same answer? ===')
print(f'    best description length: {best["without"]["DL"]:.4f} without, '
      f'{best["with"]["DL"]:.4f} with')
print(f'    {len(common)} equations ranked in both runs; '
      f'{int(np.sum(np.abs(differences) < 0.01))} agree to within 0.01 nats')
if differences.size and np.abs(differences).max() >= 0.01:
    worst = common[int(np.argmax(np.abs(differences)))]
    print(f'    largest change: {worst} '
          f'{shared[0][worst]["DL"]:.4f} -> {shared[1][worst]["DL"]:.4f}')

print("""
Summary
-------
The likelihood only sees f(x)/f(1), so every equation that differs from another
by an overall factor describes the same model. Without the catalogue ESR fits
each of them separately and reports each of them separately, which is why the
same best model appears more than once at the top of the ranking with an
identical description length each time.

With the catalogue those equations are collapsed onto one transformed model and
fitted once, so the equations the transformation has made equivalent no longer
appear as separate entries. Fewer functions are fitted, and the run comes out
faster rather than slower despite the extra transformation pass. The description
lengths themselves are unchanged: this is a statement about which equations are
the same model, not about how any of them is scored. Where a description length
does move, it is because fitting a model once rather than several times gave the
optimiser a better shot at its maximum likelihood.

Turn it on for any likelihood whose run_sympify changes the model it is handed,
and set catalogue_transform_version so the result can be cached. It is off by
default because the built-in likelihoods do not transform the parameter layout,
and the build is not free.
""")
