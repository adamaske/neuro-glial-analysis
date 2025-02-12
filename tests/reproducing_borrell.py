import numpy as np
import os
import matplotlib.pyplot as plt
from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration, light_intensity_to_optical_density, optical_density_to_hemoglobin_concentration
from mne.io import read_raw_snirf
from preprocessing.fnirs.filtering import butter_bandpass_filter
from analysis.frequencies import compute_fft, compute_psd, plot_sos_frequency_response
from mne_nirs.io.snirf import write_raw_snirf
from sklearn.decomposition import PCA, FastICA
import mne
from mne.datasets import sample
from mne.decoding import UnsupervisedSpatialFilter
# Articles
# Zhang et al 2005 -> PCA filter 
# Eigenvector Spatialfilter for reduction of physiological interference in diffuse optical imaging
# https://www.spiedigitallibrary.org/journals/journal-of-biomedical-optics/volume-10/issue-01/011014/Eigenvector-based-spatial-filtering-for-reduction-of-physiological-interference-in/10.1117/1.1852552.full

# Borrel et al 2023
# Laterality Index Calculations in a Control Study of Functional Near Infrared Spectroscopy
# https://link.springer.com/article/10.1007/s10548-023-00942-3

# Omid
roi_channels_right = ["S14_D12", "S7_D6", "S14_D6", "S7_D15", "S10_D6", "S14_D15", "S10_D12"]
roi_channels_left  = ["S12_D12", "S10_D2", "S3_D11", "S12_D2", "S10_D12", "S3_D2", "S12_D11"]

omid_folder = "data/OMID-13-12-024"
omid_filepaths = ["2024-12-13_001/2024-12-13_001.snirf", #both feet on ground
                  "2024-12-13_002/2024-12-13_002.snirf", #one feet off ground
                  "2024-12-13_003/2024-12-13_003.snirf", #easy mat
                  "2024-12-13_004/2024-12-13_004.snirf", #hard mat
                ]
    
    
def hanlde_files():
    omid_snirfs = []
    for file in omid_filepaths:
        snirf = read_raw_snirf(os.path.join(omid_folder, file))

        #pca = PCA(n_components=0.8)
        #hb = optical_density_to_hemoglobin_concentration(od)

        # CROP TO ANNOTATIONS
       #cropped = snirf.crop(tmin=start, tmax=end)
        od = light_intensity_to_optical_density(snirf)
        hb = optical_density_to_hemoglobin_concentration(od)
        omid_snirfs.append(hb)

    idx = 0
    for hb in omid_snirfs:
        write_raw_snirf(hb, "data/trsd/omid_"+str(idx)+".snirf")
        idx += 1


def get_indices_from_channels(wanted_channels, ch_names):
    hbo = []
    hbr = []
    for wanted in wanted_channels: # First "S14_D12"
        for ch_name in ch_names:
            name = ch_name.split(" ")[0]
            wl =  ch_name.split(" ")[1]
            if name == wanted:
                assert(wl == "hbo" or wl  == "hbr")
                if wl == "hbo":
                    hbo.append(idx)
                if wl == "hbr":
                    hbr.append(idx)
    return hbo, hbr

def get_channels_by_indices(indices, data):
    channels = []
    for idx in indices:
        channels.append(np.divide(data[idx], np.max(data[idx])))
    return channels


hbs = []
for idx in range(4):
    hb = read_raw_snirf("data/trsd/omid_" + str(idx) +".snirf")
    fs = hb.info["sfreq"]
    data = hb.get_data()
    hbo_left, hbr_left = get_indices_from_channels(roi_channels_left, hb.info["ch_names"])
    hbo_right, hbr_right = get_indices_from_channels(roi_channels_right, hb.info["ch_names"])
    # Only hbo for now
    left_channels = np.array(get_channels_by_indices(hbo_left, data))
    right_channels = np.array(get_channels_by_indices(hbo_right, data))
    
    left_foot_blocks = []  # should be 3 x 7
    right_foot_blocks = [] # Should be 3 x 7

    # Then we need the onset of each right foot
    for ann in hb.annotations:
            desc = ann["description"]
            onset = ann["onset"]

            tmin = -2 #Get 2 seconds before 
            tmax = 15 #Get 15 seconds after

            start = int((onset + tmin) * fs)
            end = int((onset + tmax) * fs)

            block_channels = []
            if desc == "2": # LEFT FOOT
                left_foot_blocks.append(left_channels[:, start:end])

            if desc == "1": # RIGHT FOOT
                right_foot_blocks.append(right_channels[:, start:end])
    
    left_foot = np.array(left_foot_blocks)
    right_foot = np.array(right_foot_blocks)

    fig, axs = plt.subplots(len(left_channels), len(left_foot))
    fig.suptitle(f"LEFT FOOT @ tmin={tmin}, tmax={tmax}")
    for col in range(len(left_foot)):
        axs[0, col].set_title(f"Block {str(col +1 )}", fontsize=12, fontweight='bold')
    for row in range(len(left_channels)):
        axs[row, 0].set_ylabel(roi_channels_left[row], fontsize=12, fontweight='bold', rotation=0, labelpad=15)
    
    for i in range(len(left_foot)):

        for j in range(len(left_channels)):

            x = left_foot[i][j]
            filtered = butter_bandpass_filter(x, 0.01, 0.2, fs, 10)
            freqs, spectra = compute_fft(filtered, fs, 0.2)
            axs[j][i].plot(filtered)
    
    plt.show()
    print("left_foot: ", left_foot.shape)
    print("right_foot: ", right_foot.shape)

    exit()