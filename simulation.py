import numpy as np #

#MNE - fNIRS Analysis
import mne
from mne.io import read_raw_snirf
import mne_nirs
from mne_nirs.experimental_design import make_first_level_design_matrix
from mne_nirs.statistics import run_glm
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

#np.random.seed(1)
#
#sfreq = 3.0
#amp = 4.0
#
#raw = mne_nirs.simulation.simulate_nirs_raw(
#    sfreq=sfreq, sig_dur=60 * 5, amplitude=amp, isi_min=15.0, isi_max=45.0
#)
#raw.plot(duration=300, show_scrollbars=False)
#
#design_matrix = make_first_level_design_matrix(
#    raw, stim_dur=5.0, drift_order=1, drift_model="polynomial"
#)
#fig, ax1 = plt.subplots(figsize=(10, 6), constrained_layout=True)
#fig = plot_design_matrix(design_matrix, ax=ax1)
#
#plt.show()
#
#glm_est = run_glm(raw, design_matrix)
#print(glm_est.theta())
#
#exit()
#
#
#
#
#
#
#
#
#def print_results(glm_est, truth):
#    """Print the results of GLM estimate."""
#    print(
#        "Estimate:",
#        glm_est.theta()[0][0],
#        "  MSE:",
#        glm_est.MSE()[0],
#        "  Error (uM):",
#        1e6 * (glm_est.theta()[0][0] - truth * 1e-6),
#    )
#
#print_results(glm_est, amp)


#load raw snirf
#raw = read_raw_snirf("data/balance.snirf")
#load optical density
#od = read_raw_snirf("data/optical_density.snirf")
#load hemoglobin
hb = read_raw_snirf("data/hemoglobin.snirf")
hb.plot()
plt.show()
#BLOCK DESIGN
#REST : 40 s
#RIGHT : 10 s
#REST : 20 s
#LEFT : 10 s
#REST : 40 s
#RIGHT : 10 s
#REST : 20 s
#LEFT : 10 s
#REST : 40 s
#RIGHT : 10 s
#REST : 20 s
#LEFT : 10 s 
#REST  : 40s