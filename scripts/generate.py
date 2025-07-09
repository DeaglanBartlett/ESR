import esr.generation.duplicate_checker
import esr
import os

runname = 'koza_maths_const'
max_comp = 9

fn_lib = os.path.join(os.path.dirname(esr.__file__), 'function_library', runname)

for comp in range(max_comp, max_comp+1):
    if not os.path.isfile(os.path.join(fn_lib, f'compl_{comp}', f'unique_equations_{comp}.txt')):
        esr.generation.duplicate_checker.main(runname, comp)
