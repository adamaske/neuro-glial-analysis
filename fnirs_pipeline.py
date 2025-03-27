from datasets.fnirs import read_snirf, find_snirf_in_folder, write_snirfs
from preprocessing.fnirs import preprocess_snirf
from visualization.fnirs import plot_snirf, plot_psd_snirf
from analysis.fnirs import epochs_snirf


# Load subject
sub1 = ["data/subject01/2025-03-24_001_PREPROCESSED.snirf", # Trial 1 - Supination
        "data/subject01/2025-03-24_002_PREPROCESSED.snirf", # Trial 2 - Pronation
        "data/subject01/2025-03-24_003_PREPROCESSED.snirf", # Trial 3 - Supination
        "data/subject01/2025-03-24_004_PREPROCESSED.snirf", # Trial 4 - Pronation
        "data/subject01/2025-03-24_005_PREPROCESSED.snirf", # Trial 5 - Supination
        "data/subject01/2025-03-24_006_PREPROCESSED.snirf"  # Trial 6 - Pronation
        ]

paths, snirfs = find_snirf_in_folder("data/subject01/trials")

preprocessed = [preprocess_snirf(f) for f in snirfs]

from pearson_correlation import pearson_r_channel_by_channel


[pearson_r_channel_by_channel(f) for f in preprocessed]
exit()

paths, snirfs = find_snirf_in_folder("data/OMID-13-12-024")
preprocessed = [preprocess_snirf(f) for f in snirfs]

import numpy as np

import scipy.io
# Example NumPy array

data = np.array(preprocessed[0].get_data())
scipy.io.savemat('omid_channels.mat', {'channels': data})
exit()
print(epochs_snirf(snirfs[0], -5, 15))
exit()
epoch_dict = [epochs_snirf(f, tmin=0, tmax=10) for f in snirfs]
exit()
import matplotlib.pyplot as plt
print(snirfs[0].info["ch_names"])
plt.show()
preprocessed = [preprocess_snirf(f) for f in snirfs]

print(preprocessed[0].info["ch_names"])
plot_snirf(snirfs[0], False)
preprocessed[0].plot()
exit()

preprocessed_sr = [preprocess_snirf(f, spike_removal=True) for f in snirfs]

plot_snirf(preprocessed[0], False)
plot_psd_snirf(preprocessed[0], False)

plot_snirf(preprocessed_sr[0], False)
plot_psd_snirf(preprocessed_sr[0], False)
#[plot_snirf(f) for f in preprocessed]
#[plot_psd_snirf(f) for f in preprocessed]
#[plot_snirf(f) for f in preprocessed_sr]
#[plot_psd_snirf(f) for f in preprocessed_sr]

exit()

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


