#which_run = 'pantheon'
which_run = 'cosmic_chronometers'

if which_run == 'pantheon':

    esr_dir = '/mnt/zfsusers/deaglan/symbolic_regression/brute_force/simplify_brute/'
    data_file = esr_dir + '/data/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat'
    cov_file = esr_dir + '/data/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov'
    fn_dir = esr_dir + "core_maths/"
    temp_dir = esr_dir + "/Pantheon/partial_panth"
    out_dir = esr_dir + "/Pantheon/output_panth"
    fig_dir = esr_dir + "/Pantheon/figs_panth"
    like_dir = esr_dir + "/Pantheon/"
    like_file = "likelihood_panth"
    sym_file = "symbols_panth"

elif which_run == 'cosmic_chronometers':

    # COSMIC CHRONOMETERS
    esr_dir = '/mnt/zfsusers/deaglan/symbolic_regression/brute_force/simplify_brute/'
    data_file = esr_dir + '/data/CC_Hubble.dat' 
    fn_dir = esr_dir + "core_maths/"
    temp_dir = esr_dir + "/Pantheon/partial_cc"
    out_dir = esr_dir + "/Pantheon/output_cc"
    fig_dir = esr_dir + "/Pantheon/figs_cc"
    like_dir = esr_dir + "/Pantheon/"
    like_file = "likelihood_cc"
    sym_file = "symbols_cc"

