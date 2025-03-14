import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from datasets.fnirs import read_snirf, find_snirf_in_folder
from preprocessing.fnirs import preprocess_snirf

# Montage Information


def pearson_r_channel_by_channel(snirf):
    channel_data = snirf.get_data().T  # Transpose to get channels as rows
    channel_names = snirf.info["ch_names"]

    hbo_indices = []
    hbr_indices = []
    hbo_names = []
    hbr_names = []


    for i, name in enumerate(channel_names):
        parts = name.split()
        source_detector = parts[0]
        wavelength = parts[1]

        # Consider changing the value being the i-index instead of the actual data to save performance
        if wavelength == "760" or wavelength == "HbR".lower():
            hbr_indices.append(i)
            hbr_names.append(source_detector)
        elif wavelength == "850" or wavelength == "HbO".lower():
            hbo_indices.append(i)
            hbo_names.append(source_detector)
        else:
            raise ValueError(f"Unexpected wavelength: {wavelength}")

    num_hbo = len(hbo_indices)
    num_hbr = len(hbr_indices)

    # Create separate correlation matrices for HbO and HbR
    hbo_correlation_matrix = np.zeros((num_hbo, num_hbo))
    hbr_correlation_matrix = np.zeros((num_hbr, num_hbr))

    hbo_p_value_matrix = np.zeros((num_hbo, num_hbo))
    hbr_p_value_matrix = np.zeros((num_hbr, num_hbr))

    # Calculate HbO-HbO correlations
    for i, idx_x in enumerate(hbo_indices):
        for j, idx_y in enumerate(hbo_indices):
            r_value, p_value = pearsonr(channel_data[idx_x], channel_data[idx_y])
            hbo_correlation_matrix[i, j] = r_value
            hbo_p_value_matrix[i, j] = p_value

    # Calculate HbR-HbR correlations
    for i, idx_x in enumerate(hbr_indices):
        for j, idx_y in enumerate(hbr_indices):
            r_value, p_value = pearsonr(channel_data[idx_x], channel_data[idx_y])
            hbr_correlation_matrix[i, j] = r_value
            hbr_p_value_matrix[i, j] = p_value


    if True:
    # Create subplots

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(num_hbo + num_hbr + 4, max(num_hbo, num_hbr) + 2))

        # Plot HbO correlation matrix
        im1 = ax1.imshow(hbo_correlation_matrix, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
        ax1.set_xticks(np.arange(num_hbo))
        ax1.set_yticks(np.arange(num_hbo))
        ax1.set_xticklabels(hbo_names, rotation=45, ha="right")
        ax1.set_yticklabels(hbo_names)
        ax1.set_title("HbO-HbO Correlation Matrix")
        fig.colorbar(im1, ax=ax1, label="Pearson's r")

        # Plot HbR correlation matrix
        im2 = ax2.imshow(hbr_correlation_matrix, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
        ax2.set_xticks(np.arange(num_hbr))
        ax2.set_yticks(np.arange(num_hbr))
        ax2.set_xticklabels(hbr_names, rotation=45, ha="right")
        ax2.set_yticklabels(hbr_names)
        ax2.set_title("HbR-HbR Correlation Matrix")
        fig.colorbar(im2, ax=ax2, label="Pearson's r")

        plt.tight_layout()
    return hbo_correlation_matrix, hbr_correlation_matrix, hbo_p_value_matrix, hbr_p_value_matrix

def pearson_correlation_roi(snirf, rois):

    # firsly average the channesl
    channel_data = snirf.get_data()
    channel_names = snirf.info["ch_names"]

    roi_hbo_averages = []
    roi_hbr_averages = []
    roi_names = []

    for i, roi in enumerate(rois):
        print(f"ROI : {roi}")
        roi_names.append(roi)

        roi_channel_names = rois[roi]
        print(f"Channels : {roi_channel_names}")

        hbo_channels = []
        hbr_channels = []
        for j, name in enumerate(channel_names):
            parts = name.split()
            source_detector = parts[0]
            wavelength = parts[1]

            if source_detector in roi_channel_names:
                channel = channel_data[j]
                if wavelength == "760" or wavelength == "HbR".lower():
                    hbr_channels.append(channel)
                if wavelength == "850" or wavelength == "HbO".lower():
                    hbo_channels.append(channel)

        # Average the 
        hbo_avg = np.mean(hbo_channels, axis=0)
        hbo_avg = (hbo_avg - np.min(hbo_avg)) / (np.max(hbo_avg) - np.min(hbo_avg))

        hbr_avg = np.mean(hbr_channels, axis=0)
        hbr_avg = (hbr_avg - np.min(hbr_avg)) / (np.max(hbr_avg) - np.min(hbr_avg))

        roi_hbo_averages.append(hbo_avg)
        roi_hbr_averages.append(hbr_avg)

        if False: # Plotting
            plt.subplot(1, 1, 1)

            for k, ch in enumerate(hbo_channels):
                ch = (ch - np.min(ch)) / (np.max(ch) - np.min(ch))
                plt.plot(ch, label=f"{k} HbO", linewidth=2)

            plt.plot(hbo_avg, label="HbO", color="red", linewidth=4)
            plt.plot(hbr_avg, label="HbR", color="blue", linewidth=4)
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (mM)")
            plt.title(f"{roi} : Averaged Channels")
            plt.show()

    num_hbo = len(roi_hbo_averages)
    num_hbr = len(roi_hbr_averages)

    # Create separate correlation matrices for HbO and HbR
    hbo_correlation_matrix = np.zeros((num_hbo, num_hbo))
    hbr_correlation_matrix = np.zeros((num_hbr, num_hbr))

    hbo_p_value_matrix = np.zeros((num_hbo, num_hbo))
    hbr_p_value_matrix = np.zeros((num_hbr, num_hbr))

    for i, hbo in enumerate(roi_hbo_averages):
        for j, hbo2 in enumerate(roi_hbo_averages):
            r_value, p_value = pearsonr(hbo, hbo2)
            hbo_correlation_matrix[i, j] = r_value
            hbo_p_value_matrix[i, j] = p_value

    for i, hbr in enumerate(roi_hbo_averages):
        for j, hbr2 in enumerate(roi_hbo_averages):
            r_value, p_value = pearsonr(hbr, hbr2)
            hbr_correlation_matrix[i, j] = r_value
            hbr_p_value_matrix[i, j] = p_value

    if False:
# Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(num_hbo + num_hbr + 4, max(num_hbo, num_hbr) + 2))

        # Plot HbO correlation matrix
        im1 = ax1.imshow(hbo_correlation_matrix, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
        ax1.set_xticks(np.arange(num_hbo))
        ax1.set_yticks(np.arange(num_hbo))
        ax1.set_xticklabels(roi_names, rotation=45, ha="right")
        ax1.set_yticklabels(roi_names)
        ax1.set_title("HbO-HbO Correlation Matrix")
        fig.colorbar(im1, ax=ax1, label="Pearson's r")

        # Plot HbR correlation matrix
        im2 = ax2.imshow(hbr_correlation_matrix, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
        ax2.set_xticks(np.arange(num_hbr))
        ax2.set_yticks(np.arange(num_hbr))
        ax2.set_xticklabels(roi_names, rotation=45, ha="right")
        ax2.set_yticklabels(roi_names)
        ax2.set_title("HbR-HbR Correlation Matrix")
        fig.colorbar(im2, ax=ax2, label="Pearson's r")

        plt.tight_layout()
        plt.show()

    # How to return these

    return hbo_correlation_matrix, hbr_correlation_matrix, hbo_p_value_matrix, hbr_p_value_matrix

snirf = read_snirf("data/OMID-13-12-024/2024-12-13_001/2024-12-13_001.snirf")
snirf = preprocess_snirf(snirf)

rois = {"S1"    : ['S8_D16', 'S9_D1', 'S9_D5', 'S9_D9'],
        "S2"    : ['S8_D16', 'S9_D1', 'S9_D5', 'S9_D9',],
        "M1"    : ['S12_D12', 'S13_D10', 'S13_D13', 'S14_D6', 'S14_D12', 'S14_D15'],
        "SMA"   : ['S4_D13', 'S5_D5', 'S5_D6', 'S5_D7'],
        "PMA"   : ['S9_D9', 'S10_D2', 'S10_D6', 'S10_D9', 'S10_D12'],
        "BROCA" : ['S2_D3', 'S2_D4', 'S3_D2', 'S3_D3', 'S3_D10', 'S3_D11', 'S4_D3'],
        }

paths, snirfs = find_snirf_in_folder("data/OMID-13-12-024")
preprocessed = [preprocess_snirf(f) for f in snirfs]

hbo_corr_matrices = []
hbr_corr_matrices = []

# Ignore P values for now?
hbo_p_value_matrices = []
hbr_p_value_matrices = []

for snirf in preprocessed:
    # DO THIS FOR EVERY SUBJECT -> Then average all these matrices across the subjects
    hbo_correlation_matrix, hbr_correlation_matrix, hbo_p_value_matrix, hbr_p_value_matrix = pearson_r_channel_by_channel(snirf)
    
    hbo_corr_matrices.append(hbo_correlation_matrix)
    hbr_corr_matrices.append(hbr_correlation_matrix)

    # Ignore P values for now?
    hbo_p_value_matrices.append(hbo_p_value_matrices)
    hbr_p_value_matrices.append(hbr_p_value_matrices)

plt.show()
# Calculate mean
mean_hbo_corr = np.mean(hbo_corr_matrices, axis = 0)
plt.imshow(hbo_correlation_matrix, cmap='RdBu_r', interpolation='nearest', vmin=-1, vmax=1)
plt.show()
#plt.1.set_xticks(np.arange(num_hbo))
#plt.1.set_yticks(np.arange(num_hbo))
#plt.1.set_xticklabels(hbo_names, rotation=45, ha="right")
#plt.1.set_yticklabels(hbo_names)
#plt.1.set_title("HbO-HbO Correlation Matrix")
#plt.g.colorbar(im1, ax=ax1, label="Pearson's r")
exit()
mean_hbr_corr = np.mean(hbr_corr_matrices, axis = 0)

mean_hbo_p = np.mean(hbo_p_value_matrices, axis = 0)
mean_hbr_p = np.mean(hbr_p_value_matrices, axis = 0)

# PLOT THE MEANS