"""Execute the documentation notebooks to check they still run.

The docs embed these notebooks with ``nb_execution_mode = 'off'``, so their
stored outputs are rendered as committed and are never regenerated at build
time. That keeps the docs build fast and reproducible, but it means the outputs
can go stale silently as the code changes. This script re-runs each notebook and
fails if any cell raises, without writing the outputs back -- so CI checks that
the examples still work while the committed outputs stay as they are.

Run it from a scratch directory: the notebooks write their fitting output into
the working directory.

    python docs/execute_notebooks.py [notebook ...]
"""
import glob
import os
import sys

import nbformat
from nbclient import NotebookClient

TIMEOUT = 1800


def main(paths):
    if not paths:
        here = os.path.dirname(os.path.abspath(__file__))
        paths = sorted(glob.glob(os.path.join(here, 'source', 'notebooks', '*.ipynb')))
    if not paths:
        raise SystemExit('no notebooks found')
    for path in paths:
        print(f'executing {path}', flush=True)
        notebook = nbformat.read(path, as_version=4)
        NotebookClient(notebook, timeout=TIMEOUT, kernel_name='python3').execute()
        print(f'  ok: {path}', flush=True)
    print(f'{len(paths)} notebook(s) executed cleanly', flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
