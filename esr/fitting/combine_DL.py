import numpy as np
from mpi4py import MPI
import os
from pathlib import Path
from prettytable import PrettyTable
import csv
import warnings
from collections import defaultdict

import esr.fitting.test_all as test_all
from esr.fitting.utils import combine_temp_files, fitting_paths, raw_catalogue_paths

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main(comp, likelihood, print_frequency=1000):
    """Combine the description lengths of all functions of a given complexity, sort by this and save to file.

    Args:
        :comp (int): complexity of functions to consider
        :likelihood (fitting.likelihood object): object containing data, likelihood functions and file paths
        :print_frequency (int, default=1000): the status of the fits will be printed every ``print_frequency`` number of iterations

    Returns:
        None

    """
    if likelihood.is_mse:
        raise ValueError('Cannot use MSE with description length')

    if rank == 0:
        print('\nComputing description lengths', flush=True)

    raw_paths = raw_catalogue_paths(comp, likelihood)
    fit_paths = fitting_paths(comp, likelihood, rank=rank)
    allfn_file = raw_paths['all']
    aifeyn_file = raw_paths['fnprior']

    _, data_start, data_end = test_all.get_functions(comp, likelihood)

    needed_indices = set(np.arange(data_start, data_end)
                         )  # faster lookup for indices
    results = defaultdict(list)
    results_fcn = {}

    codelen_file = Path(fit_paths['codelen_matches'])
    companion_files = {
        "codelen matches": codelen_file,
        "AIFeyn": Path(aifeyn_file),
        "function catalogue": Path(allfn_file),
    }
    line_counts = {}
    for name, path in companion_files.items():
        with path.open('r') as f:
            line_counts[name] = sum(1 for _ in f)
    if len(set(line_counts.values())) != 1:
        counts = ", ".join(f"{name}={count}" for name, count in line_counts.items())
        raise ValueError(
            "Companion files for description-length combination have unequal "
            f"line counts ({counts}); regenerate the matched results.")
    num_lines = line_counts["codelen matches"]

    malformed_records = defaultdict(int)

    with codelen_file.open('r') as f, \
            open(aifeyn_file, 'r') as aifeyn_f, \
            open(allfn_file, 'r') as allfn_f:

        for i, (line, line_ai, line_fcn) in enumerate(zip(f, aifeyn_f, allfn_f)):

            if rank == 0 and i % print_frequency == 0:
                print(f'{i+1} of {num_lines}', flush=True)

            if line.strip() == '':
                malformed_records["empty rows"] += 1
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                malformed_records["rows with fewer than three fields"] += 1
                continue
            try:
                values = [float(value) for value in parts]
                aifeyn_i = float(line_ai.strip())
                idx = int(values[2])  # Index is in column 3
            except (ValueError, IndexError, OverflowError):
                malformed_records["rows with non-numeric fields"] += 1
                continue

            if idx in needed_indices:
                negloglike_i = values[0]
                codelen_i = values[1]
                DL = negloglike_i + codelen_i + aifeyn_i

                if not np.isfinite(DL) or np.isnan(DL):
                    continue

                # This is the first time we see this index
                if (len(results[idx]) == 0) or (DL < results[idx][0]):
                    results[idx] = [
                        DL] + values[3:] + [negloglike_i, codelen_i, aifeyn_i]
                    # Store the function string
                    results_fcn[idx] = line_fcn.strip()

    if rank == 0 and malformed_records:
        summary = ", ".join(
            f"{count} {reason}" for reason, count in malformed_records.items())
        warnings.warn(
            f"Skipped malformed description-length records: {summary}.",
            RuntimeWarning,
            stacklevel=2,
        )

    num_params = max((len(row) - 4 for row in results.values()), default=0)
    # Ensure all ranks agree on parameter width so the concatenated file has a
    # consistent number of columns even when some ranks have no valid results.
    num_params = comm.allreduce(num_params, op=MPI.MAX)
    num_cols = num_params + 4

    output_file = fit_paths['combined_rank']
    output_file_fcn = fit_paths['combined_functions_rank']
    with open(output_file, 'w') as fout, \
            open(output_file_fcn, 'w') as fout_fcn:
        for idx in range(data_start, data_end):
            if idx in results:
                row = results[idx]
                params = row[1:-3]
                if len(params) > num_params:
                    raise RuntimeError(
                        "Parameter width exceeds the combined-file maximum.")
                line_data = [row[0]] + params + [0.0] * (num_params - len(params)) + row[-3:]
                fcn = results_fcn[idx]
            else:
                line_data = [np.nan] + [0.0] * (num_cols-1)
                fcn = "None"

            fout.write(" ".join(f"{x:.16e}" for x in line_data) + "\n")
            fout_fcn.write(f"{fcn}\n")

    comm.Barrier()

    if rank == 0:
        combine_temp_files(
            likelihood.temp_dir,
            fit_paths['combined_rank_pattern'],
            fit_paths['combined'])
        combine_temp_files(
            likelihood.temp_dir,
            fit_paths['combined_functions_rank_pattern'],
            fit_paths['combined_functions'])
        data_entries = []
        num_params = 0
        with open(fit_paths['combined'], 'r') as f, \
                open(fit_paths['combined_functions'], "r") as fcn_f:
            for i, (line, fcn_line) in enumerate(zip(f, fcn_f)):
                parts = line.strip().split()
                if not parts:
                    continue
                DL = float(parts[0])
                if (not np.isnan(DL)) and (not np.isinf(DL)):
                    # Store DL, index, and other info
                    data_entries.append((DL, parts[1:], fcn_line.strip()))
                num_params = max(num_params, len(parts) - 4)
        print(f"Number of parameters: {num_params}", flush=True)
        n_read = i + 1 if 'i' in dir() else 0
        print(f'Original file length: {n_read}', flush=True)
        data_entries.sort(key=lambda x: x[0])
        print(
            f"Sorted {len(data_entries)} entries by DL for complexity {comp}", flush=True)

        #  Get relative probabilities
        if len(data_entries) == 0:
            print("(no valid functions at this complexity)", flush=True)
            path = Path(fit_paths['final'])
            path.write_text("")
            comm.Barrier()
            return

        Prel_DL = np.array([entry[0] for entry in data_entries])
        log_L = np.array([entry[1][-3] for entry in data_entries])
        Prel_DL -= Prel_DL[0]  # Shift so the best function has DL=0
        Prel = np.exp(-Prel_DL)
        duplicates = log_L[1:] == log_L[:-1]
        Prel[1:][duplicates] = 0.0
        Prel[~np.isfinite(Prel) | np.isnan(Prel)] = 0.0
        Prel /= np.sum(Prel)

        ptab = PrettyTable()
        ptab.field_names = ["Rank", "Function", "L(D)", "Prel", "-logL", "Codelen", "AIFeyn"] + [
            f"a{i}" for i in range(num_params)]

        Nfuncs = 10

        # Start this file from scratch here
        if os.path.exists(fit_paths['final']):
            os.remove(fit_paths['final'])

        for i, d in enumerate(data_entries):  # Only print the top 10 functions
            if i < Nfuncs:
                fcn = d[-1]
                DL = d[0]
                params = [float(pp) for pp in d[1][:-3]]
                # Pad params to num_params (0-param functions have fewer columns)
                while len(params) < num_params:
                    params.append(0.0)
                if len(params) > num_params:
                    raise RuntimeError(
                        "Parameter width exceeds the combined-file maximum.")
                negloglike = float(d[1][-3])
                codelen = float(d[1][-2])
                aifeyn = float(d[1][-1])
                ptab.add_row([i+1, fcn, '%.2f' % DL, '%.2e' % Prel[i], '%.2f' % negloglike,
                             '%.2f' % codelen, '%.2e' % aifeyn] + ['%.2e' % p for p in params])

            with open(fit_paths['final'], 'a') as f:
                writer = csv.writer(f, delimiter=';')
                # Pad params to num_params for consistent column count
                row_params = list(d[1][:-3])
                if len(row_params) > num_params:
                    raise RuntimeError(
                        "Parameter width exceeds the combined-file maximum.")
                row_params += ['0.0'] * (num_params - len(row_params))
                writer.writerow([i,
                                 d[-1],  # fcn
                                 d[0],  # DL
                                 Prel[i],
                                 d[1][-3],  # negloglike
                                 d[1][-2],  # codelen
                                 d[1][-1]] + row_params)  # aifeyn, params

        print(ptab)

        with open(fit_paths['results_pretty'], 'w') as f:
            print(ptab, file=f)

    comm.Barrier()

    return
