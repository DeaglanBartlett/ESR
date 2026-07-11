# Current state

- PR #57 (`feature/det_I`) updates preserve the legacy 1-D Fisher-diagonal return from `convert_params`; `match.py` explicitly requests the full transformed matrix for determinant scoring.
- `combine_DL` now rejects unequal companion-file lengths, warns once on rank 0 for malformed records, preserves the true global maximum parameter width, and clears a stale final file when no rows are valid.
- Inverse substitution index/substitution length mismatches now raise, and the numerical-diagnostic report path uses `os.path.join`.
- README/docs/tutorial and `fit_single` now define Fisher scoring, diagonal/eigenbasis snapping, likelihood-aware catalogues, and numerical fingerprints.
- Focused regressions, Ruff, compileall, docs HTML (apart from the existing missing `_static` warning), and the full suite pass: `140 passed` in 154 s with numerical backends pinned to one local core.
- Glamdring fingerprint baselines completed on one core in `cmb-priority`: c9 job `783642` processed 28,465 equations in 58.510 s (90 MiB peak RSS); c10 job `783643` processed 98,022 equations in 245.646 s (115 MiB). Keep the optional diagnostic serial; MPI parallelisation is not justified by these timings.

# Next steps

1. Push the reviewed PR #57 update and resolve Deaglan's threads after confirming the GitHub replies.
