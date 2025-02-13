from datasets.fnirs import read_snirf, find_snirf_in_folder, write_snirfs
from preprocessing.fnirs import preprocess_snirf
from visualization.fnirs import plot_snirf, plot_psd_snirf
from analysis.fnirs import epochs_snirf

paths, snirfs = find_snirf_in_folder("data/OMID-13-12-024")

#preprocessed = [preprocess_snirf(f,) for f in snirfs]
epochs, order = [epochs_snirf(f, tmin=-2, tmax=15) for f in snirfs]

#preprocessed_sr = [preprocess_snirf(f, tddr=True, ) for f in snirfs]

#[plot_snirf(f) for f in preprocessed]
#[plot_psd_snirf(f) for f in preprocessed]

#[plot_snirf(f) for f in preprocessed_sr]
#[plot_psd_snirf(f) for f in preprocessed_sr]

#write_snirfs(paths, preprocessed, append_to_path="_OD_HB_filtered")
#write_snirfs(paths, preprocessed, append_to_path="_OD_HB_filtered_tddr_znom_pca")

#plot_snirf(preprocessed[0], False)
#plot_psd_snirf(preprocessed[0], False)
#
#plot_snirf(preprocessed_sr[0], False)
#plot_psd_snirf(preprocessed_sr[0], False)
#
#import matplotlib.pyplot as plt
#plt.show()


