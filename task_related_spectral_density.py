
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration
from mne.io import read_raw_snirf

# Omid
omid_roi_channels_right = ["S14_D12", "S7_D6", "S14_D6", "S7_D15", "S10_D6", "S14_D15", "S10_S12"]
omid_roi_channels_left  = ["S12_D12", "S10_D2", "S3_D11", "S12_D2", "S10_D12", "S3_D2", "S12_D11"]

omid_folder = "data/OMID-13-12-024"
omid_filepaths = ["2024-12-13_001/2024-12-13_001.snirf", #both feet on ground
                  "2024-12-13_002/2024-12-13_002.snirf", #one feet off ground
                  "2024-12-13_003/2024-12-13_003.snirf", #easy mat
                  "2024-12-13_004/2024-12-13_004.snirf", #hard mat
                ]

omid_snirfs = []
for file in omid_filepaths:
    snirf = read_raw_snirf(os.path.join(omid_folder, file))
    omid_snirfs.append(snirf)

# Danial
daniel_roi_channels_right = ["S10_D12", "S10_D10", "S12_D12", "S14_D14", "S12_D10"]
daniel_roi_channels_left  = ["S8_D6", "S4_D6", "S6_D6", "S8_D8", "S6_D8"]

daniel_folder = "data/DANIEL-28-01-2025"
daniel_filepaths = ["2025-01-28_001/2025-01-28_001.snirf",
                    "2025-01-28_002/2025-01-28_002.snirf"
                    ]

daniel_snirfs = []
for file in daniel_filepaths:
    snirf = read_raw_snirf(os.path.join(daniel_folder, file))
    daniel_snirfs.append(snirf)

omid_1   =  light_intensity_to_hemoglobin_concentration(omid_snirfs[1]) #Feet on ground
omid_2   =  light_intensity_to_hemoglobin_concentration(omid_snirfs[3]) #Difficult mat
daniel_1 =  light_intensity_to_hemoglobin_concentration(daniel_snirfs[0]) #Hands
daniel_2 =  light_intensity_to_hemoglobin_concentration(daniel_snirfs[1]) #Hands


#print(omid_1.info["ch_names"])
#print(daniel_1.info["ch_names"])

def get_indices_from_channels(wanted_channels, ch_names):
    indicies = []
    for idx in range(len(ch_names)):
        ch_name = ch_names[idx].split(" ")[0]
        for ch in wanted_channels:
            if ch == ch_name:
                indicies.append(idx)
    return indicies

o_l = get_indices_from_channels(omid_roi_channels_left, omid_1.info["ch_names"])
o_r = get_indices_from_channels(omid_roi_channels_right, omid_1.info["ch_names"])
print("Omid Left Indices : ", o_l)
print("Omid Right Indices: ", o_r)

d_l = get_indices_from_channels(daniel_roi_channels_left, daniel_1.info["ch_names"])
d_r = get_indices_from_channels(daniel_roi_channels_right, daniel_1.info["ch_names"])
print("Daniel Left Indices : ", d_l)
print("Daniel Right Indices: ", d_r)

d_roi_left = []
d_roi_right = []
d_data = daniel_1.get_data()
for ch in d_l:
    d_roi_left.append(d_data[ch])

for ch in d_r:
    d_roi_right.append(d_data[ch])

daniel_1.plot()
plt.plot(d_roi_left[0])

plt.show()
exit()
#stats
omid_fs = omid_snirfs[0].info["sfreq"]
daniel_fs = daniel_snirfs[0].info["sfreq"]
print("Omid Sampling Frequency", omid_fs)
print("Daniel Sampling Frequency", daniel_fs)

#Extract ROI channels
print()

# identify what channels are left vs right hemisphere


# ROIs :
# - Sensory Cortex -> C3, C4, CP1, CP2, etc
# - Premotor Cortex -> etc..

# Omid Stimulation Frequency

# Daniel Stimulation Frequency



def task_related_spectral_density(time_series, stimulation_frequency, bandwidth, resolution, ):
    lowcut = stimulation_frequency - (bandwidth/2.0)
    highcut = stimulation_frequency + (bandwidth/2.0)

    d_x = (highcut - lowcut) / resolution #delta X
    
    # Generate a sample signal with noise
    fs = 1000  # Sampling frequency in Hz
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second duration
    freq1, freq2 = 50, 120  # Two frequencies in Hz
    signal = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)


daniel_S_f = 0.04


omid_S_f = 0.0333

bandwidths = [0.02, 0.03, 0.04, 0.05]
resolutions = [ 200, 500, 1000]

for w in bandwidths:
    for n in resolutions:

        task_related_spectral_density(S_f, w, n)
