import os
from neuropipeline.fnirs import fNIRS
from neuropipeline.eeg import EEG
from functional import composite_correlation, plot_r_matrix
import matplotlib.pyplot as plt

subject_01_fnirs_S = [
"data/Subject01/Trial 1 - Supination/2025-03-24_001.snirf",
"data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf",
"data/Subject01/Trial 5 - Supination/2025-03-24_005.snirf",
]
subject_01_fnirs_P = [
"data/Subject01/Trial 2 - Pronation/2025-03-24_002.snirf",
"data/Subject01/Trial 4 - Pronation/2025-03-24_004.snirf",
"data/Subject01/Trial 6 - Pronation/2025-03-24_006.snirf",
]
subject_02_fnirs_S = [
"data/Subject02/Trial 1/2025-03-27_002.snirf",
"data/Subject02/Trial 3/2025-03-27_004.snirf",
"data/Subject02/Trial 5/2025-03-27_006.snirf",
]
subject_02_fnirs_P = [
"data/Subject02/Trial 2/2025-03-27_003.snirf",
"data/Subject02/Trial 4/2025-03-27_005.snirf",
"data/Subject02/Trial 6/2025-03-27_007.snirf",
]
subject_03_fnirs_S = [
"data/BeforeCVit_Subject02/Lower Body/Trial 5 - Supination/2025-04-10_005.snirf",
"data/BeforeCVit_Subject02/Lower Body/Trial 3 - Supination/2025-04-10_003.snirf",
"data/BeforeCVit_Subject02/Lower Body/Trial 1 - Supination/2025-04-10_001.snirf",
]
subject_03_fnirs_P = [
"data/BeforeCVit_Subject02/Lower Body/Trial 6 - Pronation/2025-04-10_006.snirf",
"data/BeforeCVit_Subject02/Lower Body/Trial 4 - Pronation/2025-04-10_004.snirf",
"data/BeforeCVit_Subject02/Lower Body/Trial 2 - Pronation/2025-04-10_002.snirf",
]

subject_01_eeg_S = [
"data/Subject01/Trial 5 - Supination/heel2025.03.24_14.45.30.hdf5",
"data/Subject01/Trial 3 - Supination/heel2025.03.24_14.36.01.hdf5",
"data/Subject01/Trial 1 - Supination/heel2025.03.24_14.27.28.hdf5",
]
subject_01_eeg_P = [
"data/Subject01/Trial 4 - Pronation/heel2025.03.24_14.40.18.hdf5",
"data/Subject01/Trial 2 - Pronation/heel2025.03.24_14.31.33.hdf5",
"data/Subject01/Trial 6 - Pronation/heel2025.03.24_14.50.12.hdf5",
]
subject_02_eeg_S = [
"data/Subject02/Trial 5/HeelSubject22025.03.27_11.29.06.hdf5",
"data/Subject02/Trial 3/HeelSubject22025.03.27_11.21.29.hdf5",
"data/Subject02/Trial 1/HeelSubject22025.03.27_11.14.27.hdf5",
]
subject_02_eeg_P = [
"data/Subject02/Trial 2/HeelSubject22025.03.27_11.17.47.hdf5",
"data/Subject02/Trial 4/HeelSubject22025.03.27_11.25.31.hdf5",
"data/Subject02/Trial 6/HeelSubject22025.03.27_11.32.54.hdf5",
]
subject_03_eeg_S = [
"data/BeforeCVit_Subject02/Lower Body/Trial 5 - Supination/PreCitVitSub22025.04.10_10.02.36.hdf5",
"data/BeforeCVit_Subject02/Lower Body/Trial 3 - Supination/PreCitVitSub22025.04.10_09.51.47.hdf5",
"data/BeforeCVit_Subject02/Lower Body/Trial 1 - Supination/PreCitVitSub22025.04.10_09.41.53.hdf5",
]
subject_03_eeg_P = [
"data/BeforeCVit_Subject02/Lower Body/Trial 6 - Pronation/PreCitVitSub22025.04.10_10.06.49.hdf5",
"data/BeforeCVit_Subject02/Lower Body/Trial 4 - Pronation/PreCitVitSub22025.04.10_09.56.05.hdf5",
"data/BeforeCVit_Subject02/Lower Body/Trial 2 - Pronation/PreCitVitSub22025.04.10_09.47.03.hdf5",
]

fnirs_S = list(subject_01_fnirs_S + subject_02_fnirs_S + subject_03_fnirs_S)
fnirs_P = list(subject_01_fnirs_P + subject_02_fnirs_P + subject_03_fnirs_P)
eeg_S = list(subject_01_eeg_S + subject_02_eeg_S + subject_03_eeg_S)
eeg_P = list(subject_01_eeg_P + subject_02_eeg_P + subject_03_eeg_P)

print("fNIRS : Supination")
for i, path in enumerate(fnirs_S):
    print(f"{i+1}. ", os.path.basename(path))
    
print("fNIRS : Pronation")
for i, path in enumerate(fnirs_P):
    print(f"{i+1}. ", os.path.basename(path))
    
print("EEG : Supination")
for i, path in enumerate(eeg_S):
    print(f"{i+1}. ", os.path.basename(path))
    
print("EEG : Pronation")
for i, path in enumerate(eeg_P):
    print(f"{i+1}. ", os.path.basename(path))

fnirs_S = [fNIRS(file) for file in fnirs_S]
# HbO S -> Composite Correlation
for i, fnirs in enumerate(fnirs_S):
    
    #fnirs.plot_channels() # See raw data, close window to proceeed

    fnirs.remove_features([2, 5, 6]) # Remove markers 
    fnirs.trim_from_features(cut_from_first_feature=5, cut_from_last_feature=10) # Remove data before and after block design data
    
    fnirs.preprocess(optical_density=True, 
                     hemoglobin_concentration=True,
                     temporal_filtering=True, # Bandpass 0.01 to 0.1 Hz
                     normalization=True, # z-normalization
                     )
    
    #fnirs.plot_channels() # Review the processsed

    hbo_data, hbo_names, hbr_data, hbr_names = fnirs.split()
    
    # Epoching
     
    markers = {3 : "Pronation", 
              4 : "Supination"
              }
    
    
    tmin = -2
    tmax = 20
    for j, desc in enumerate(fnirs.feature_descriptions):
        onset = fnirs.feature_onsets[j]
        
        marker = markers[desc] 
        
        start = int((onset + tmin) * fnirs.sampling_frequency)
        end = int((onset + tmax) * fnirs.sampling_frequency)
        
        hbo_epoch = hbo_data[:, start:end]
        hbr_epoch = hbr_data[:, start:end]
        
        # 
        r = composite_correlation(hbo_epoch, hbo_names, fnirs.sampling_frequency, 16, 0.01, 0.1)
        plot_r_matrix(r, hbo_names, f"HbO Epoch {j + 1}")
        
        plt.show()
        
        epoch_data = fnirs.channel_data[:, start:end]
        
        print("epoch : ", epoch_data.shape)
        print(f"{j}. ", marker)
        
exit()
    
r = composite_correlation(hbo_data, hbo_names, fnirs.sampling_frequency, None, 0.01, 0.1)
plot_r_matrix(r, hbo_names, "Supination Composite Matrix")
    