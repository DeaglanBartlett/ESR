"""
Worked example of the Fisher scoring options ``use_det_I`` and ``snap_choice``.

The parametric term of the description length asks how precisely each fitted
parameter must be encoded. ESR's published formula uses only the diagonal of the
Fisher matrix (``use_det_I=False``); the default now uses the determinant of the
full Hessian (``use_det_I=True``). Parameters that the data cannot resolve are
"snapped" to zero and not paid for: ``snap_choice=0`` tests one parameter axis
at a time, ``snap_choice=1`` (the default) diagonalises the Hessian and tests
its eigendirections, and ``snap_choice=2`` additionally evaluates the snapped
fit in the rotated basis.

The script has two parts.

1. It runs the complexity-5 ``core_maths`` catalogue on mock data once per
   setting. ESR's catalogue keeps several algebraic forms of the same
   two-parameter linear family; they fit the data identically and cost the same
   to write down, so they should have identical description lengths. Under the
   diagonal formula they do not. Under the determinant they agree exactly, and
   the gap between the two is shown to be a known function of the parameter
   correlations rather than anything to do with the data.

2. It fits a single function to two datasets to show what snapping does: a
   parameter consistent with zero is removed and the description length falls.

It takes about a minute on one core, and generates the function catalogue first
if it is not already present. Run it with ``mpirun`` for the fitting to be
shared across ranks.
"""

import contextlib
import io
import os
import shutil
import numpy as np
from mpi4py import MPI

import esr.generation.duplicate_checker
import esr.fitting.test_all
import esr.fitting.test_all_Fisher
import esr.fitting.match
import esr.fitting.combine_DL
from esr.fitting.fit_single import fit_from_string
from esr.fitting.likelihood import GaussLikelihood

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

comp = 5
run_name = 'fisher_options'
work_dir = os.path.join(os.getcwd(), run_name)
data_file = 'fisher_options_data.txt'

# The scoring settings to compare. The first is the published diagonal
# codelength, the second is the current default, and the third is the projected
# eigenbasis, included to show why it is not the default.
settings = [(False, 0), (True, 1), (True, 2)]


def tag(use_det_I, snap_choice):
    return f'det{int(use_det_I)}_snap{snap_choice}'


#  (1) Mock data: y = 3 + 1.7 x, i.e. the two-parameter linear family. The
#  slope is deliberately not an integer, so no one-parameter equation in the
#  catalogue reproduces it and the true model really is a two-parameter one.
if rank == 0:
    os.makedirs(work_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    xvar = np.linspace(1.0, 5.0, 100)
    yerr = np.full_like(xvar, 0.5)
    yvar = 3.0 + 1.7 * xvar + rng.normal(scale=yerr)
    np.savetxt(os.path.join(work_dir, data_file),
               np.array([xvar, yvar, yerr]).T)
comm.Barrier()

likelihood = GaussLikelihood(data_file, run_name, data_dir=work_dir,
                             base_out_dir=work_dir)

#  (2) Generate the catalogue if it is not already on disk
if rank == 0:
    if not os.path.isfile(os.path.join(likelihood.fn_dir, f'compl_{comp}',
                                       f'unique_equations_{comp}.txt')):
        for c in range(1, comp + 1):
            esr.generation.duplicate_checker.main('core_maths', c)
comm.Barrier()

#  (3) Fit once (the maximum-likelihood fits do not depend on the scoring
#  options), then score, match and rank once per setting
esr.fitting.test_all.main(comp, likelihood)

for use_det_I, snap_choice in settings:
    esr.fitting.test_all_Fisher.main(comp, likelihood, use_det_I=use_det_I,
                                     snap_choice=snap_choice)
    esr.fitting.match.main(comp, likelihood)
    esr.fitting.combine_DL.main(comp, likelihood)
    if rank == 0:
        shutil.copy(
            os.path.join(likelihood.out_dir, f'final_{comp}.dat'),
            os.path.join(work_dir,
                         f'final_{comp}_{tag(use_det_I, snap_choice)}.dat'))
comm.Barrier()

if rank != 0:
    raise SystemExit


#  (4) Read the rankings back
def load_ranking(use_det_I, snap_choice):
    """Rank, description length and its terms, keyed by equation string"""
    result = {}
    fname = os.path.join(work_dir,
                         f'final_{comp}_{tag(use_det_I, snap_choice)}.dat')
    with open(fname, 'r') as f:
        for line in f:
            parts = line.rstrip('\n').split(';')
            if len(parts) < 7:
                continue
            result[parts[1]] = {'rank': int(parts[0]), 'DL': float(parts[2]),
                                'negloglike': float(parts[4]),
                                'codelen': float(parts[5]),
                                'aifeyn': float(parts[6])}
    return result


rankings = {s: load_ranking(*s) for s in settings}

#  (5) The same model family, written several ways. a0*(a1 + x) = a0*a1 + a0*x
#  and a0*(a1 - x) = a0*a1 - a0*x are the same two-parameter family as
#  a0 + a1*x, so all of them describe the data equally well and cost the same to
#  write down as an equation. Members are found by their maximum likelihood
#  rather than from a hard-coded list, so whichever spelling a given setting
#  happens to keep is picked up.
reference = rankings[(False, 0)]['a0 + a1*x']['negloglike']

print('\n=== One model family, several algebraic forms ===')
print(f'    (every form here fits the data with -log(L) = {reference:.5f})')
for use_det_I, snap_choice in settings:
    print(f'\nuse_det_I={use_det_I}, snap_choice={snap_choice}')
    present = {eq: r for eq, r in rankings[(use_det_I, snap_choice)].items()
               if abs(r['negloglike'] - reference) < 1e-4}
    for eq, r in sorted(present.items(), key=lambda kv: kv[1]['rank']):
        print(f'    {eq:<14} rank {r["rank"]:3d}   L = {r["DL"]:9.5f}   '
              f'parametric = {r["codelen"]:8.5f}   '
              f'functional = {r["aifeyn"]:7.5f}')
    if len(present) > 1:
        codelens = [r['codelen'] for r in present.values()]
        print(f'    spread in the parametric term across these '
              f'{len(present)} forms: {max(codelens) - min(codelens):.5f} nats')

#  (6) The diagonal formula's excess is exactly the correlation term. For a
#  linear model the Fisher matrix is known analytically, so the gap between the
#  two parametric codelengths can be predicted rather than merely measured:
#  0.5*sum(log(I_jj)) - 0.5*log(det(I)) = -0.5*log(det(C)), for C the
#  correlation matrix of I. This is non-negative by Hadamard's inequality and
#  depends only on how the model was parameterised, not on how well it fits.
xvar, yvar, yerr = np.loadtxt(os.path.join(work_dir, data_file), unpack=True)
design = np.vstack([np.ones_like(xvar), xvar]).T
fisher = design.T @ design / yerr[0] ** 2
scale = np.sqrt(np.diag(fisher))
_, logdet_corr = np.linalg.slogdet(fisher / np.outer(scale, scale))
measured = (rankings[(False, 0)]['a0 + a1*x']['codelen']
            - rankings[(True, 1)]['a0 + a1*x']['codelen'])
print('\n=== What the determinant drops, for a0 + a1*x ===')
print(f'    predicted -0.5*log(det(C)) = {-0.5 * logdet_corr:.6f} nats')
print(f'    measured parametric difference = {measured:.6f} nats')

#  (7) What snapping does, and where the modes differ. Fit a0 + a1*x to three
#  datasets: one where the intercept is real, one where it is exactly zero, and
#  one taken so far from x = 0 that the intercept cannot be determined at all
#  even though a0 itself comes out large. These fits use one function at a time,
#  so they do not need the catalogue.
basis_functions = [["x", "a"], ["inv"], ["+", "*", "-", "/", "pow"]]
snap_dir = os.path.join(work_dir, 'snapping')
os.makedirs(snap_dir, exist_ok=True)

cases = [
    ('y = 3 + 2x, x in [-2, 2]', 3.0, -2.0, 2.0),
    ('y = 2x,     x in [-2, 2]', 0.0, -2.0, 2.0),
    ('y = 3 + 2x, x near 1e4  ', 3.0, 1.0e4, 1.0e4 + 40.0),
]

print('\n=== Snapping a parameter the data do not need ===')
for label, intercept, xlo, xhi in cases:
    rng = np.random.default_rng(0)
    xs = np.linspace(xlo, xhi, 200)
    es = np.full_like(xs, 0.5)
    ys = intercept + 2.0 * xs + rng.normal(scale=es)
    np.savetxt(os.path.join(snap_dir, 'snap_data.txt'),
               np.array([xs, ys, es]).T)
    snap_likelihood = GaussLikelihood('snap_data.txt', 'snapping',
                                      data_dir=snap_dir, base_out_dir=snap_dir)
    print(f'\ntruth: {label}    fitting a0 + a1*x')
    for use_det_I, snap_choice in [(True, 0), (True, 1), (True, 2)]:
        #  fit_from_string reports its own progress; quieten it so the
        #  comparison is easy to read
        with contextlib.redirect_stdout(io.StringIO()):
            negloglike, DL, labels, params = fit_from_string(
                'a0 + a1*x', basis_functions, snap_likelihood, verbose=False,
                return_params=True, use_det_I=use_det_I,
                snap_choice=snap_choice)
        kept = int(np.sum(np.asarray(params) != 0))
        print(f'    snap_choice={snap_choice}: L = {DL:9.4f}   '
              f'-log(L) = {negloglike:9.4f}   parameters kept = {kept}   '
              f'a0 = {params[0]:12.5f}   a1 = {params[1]:.5f}')

print("""
Summary
-------
The diagonal formula charges equivalent forms of one model differently, so which
form ESR reports as best is settled by an algebraic accident rather than by the
data. The determinant gives them the same parametric codelength, and the charge
it drops is exactly the correlation term above: it is fixed by the choice of
parameters, so it carries no information about the model or the data.

snap_choice=2 is shown for comparison only. It zeroes the coordinate in the
rotated basis, so which member of a family survives depends on the eigenbasis;
above it keeps a different set of forms from the other two settings and does not
give them equal codelengths. Prefer snap_choice=1 unless you specifically want
the projected treatment.

The snapping section shows where the modes part company. A real intercept is
kept by all of them, and an exactly zero intercept is dropped by all of them:
that one lies along a parameter axis, which is the case the diagonal test of
snap_choice=0 handles as well as the eigendecomposition does.

The third dataset is the one that separates them. Taken over a narrow range near
x = 1e4, the data fix the slope but say nothing about where the line crosses
x = 0, so the intercept is not determined -- and it is the *combination* of a0
and a1 that is unconstrained, not a0 on its own. snap_choice=0 looks at each
parameter separately, sees an a0 far larger than its own precision step, and
keeps it -- at a value nowhere near the 3.0 the data were generated with, and
which lands somewhere different every time the script is run. The eigenbasis
modes find the unconstrained direction, drop the intercept and refit the slope,
and report the same numbers every time. Note that snap_choice=0 comes out with
the *shorter* description length there, by charging for a parameter it has not
actually determined.

For the same reason, pairing use_det_I=True with snap_choice=0 warns. It is a
legitimate comparison setting -- it is how you attribute a change to the
determinant alone -- but diagonal snapping cannot remove an unconstrained
direction from det(H), so it is not safe for ranking a catalogue.
""")
