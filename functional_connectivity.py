
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets.fnirs import read_snirf, find_snirf_in_folder
from preprocessing.fnirs import preprocess_snirf
from scipy.signal import correlate
from scipy.stats import pearsonr

path = "data/DanielAfterCVitamin/Lower Body"

paths, snirfs = find_snirf_in_folder(path)

[print(p) for p in paths]

snirfs = [preprocess_snirf(f) for f in snirfs]

def functional_connectivity(snirf, plot=True):
    
    channel_data = snirf.get_data()
    channel_names = snirf.info["ch_names"]
    N = channel_data.shape[0]  # Total number of channels

    hbo_r = np.zeros((N // 2, N // 2))  # HbO correlation matrix
    hbr_r = np.zeros((N // 2, N // 2))  # HbR correlation matrix
    hbo_cross = np.zeros((N // 2, N // 2))  # HbO cross-correlation matrix
    hbr_cross = np.zeros((N // 2, N // 2))  # HbR cross-correlation matrix

    for i in range(N // 2, N):  # Iterate through HbO channels
        for j in range(N // 2, N):  # Iterate through HbO channels
            r, _ = pearsonr(channel_data[i], channel_data[j])
            hbo_r[i - N // 2, j - N // 2] = r
            Cxy = correlate(channel_data[i], channel_data[j], mode='full')
            Cxy_normalized = Cxy / (np.sqrt(np.sum(channel_data[i]**2)) * np.sqrt(np.sum(channel_data[j]**2)))
            hbo_cross[i - N // 2, j - N // 2] = np.max(np.abs(Cxy_normalized))
        
    for i in range(N // 2):  # Iterate through HbR channels
        for j in range(N // 2):  # Iterate through HbR channels
            r, _ = pearsonr(channel_data[i], channel_data[j])
            hbr_r[i, j] = r
            Cxy = correlate(channel_data[i], channel_data[j], mode='full')
            Cxy_normalized = Cxy / (np.sqrt(np.sum(channel_data[i]**2)) * np.sqrt(np.sum(channel_data[j]**2)))
            hbr_cross[i, j] = np.max(np.abs(Cxy_normalized))
          
          
    # Thresholding r-matrices
    hbo_r_mean = np.mean(hbo_r)
    hbr_r_mean = np.mean(hbr_r)

    hbo_r_thresholded = np.where(hbo_r >= hbo_r_mean, hbo_r, 0)
    hbr_r_thresholded = np.where(hbr_r >= hbr_r_mean, hbr_r, 0)
    # Thresholding r-matrices
    hbo_cross_mean = np.mean(hbo_cross)
    hbr_cross_mean = np.mean(hbr_cross)

    hbo_cross_thresholded = np.where(hbo_cross >= hbo_cross_mean, hbo_cross, 0)
    hbr_cross_thresholded = np.where(hbr_cross >= hbr_cross_mean, hbr_cross, 0)
      
    # Plot r-matrices and cross-correlation matrices
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 2, 1)
    sns.heatmap(hbo_r_thresholded, annot=False, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('HbO Correlation Matrix (r)')
    plt.xticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[N // 2:]], rotation=90)
    plt.yticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[N // 2:]], rotation=0)

    plt.subplot(2, 2, 2)
    sns.heatmap(hbr_r_thresholded, annot=False, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title('HbR Correlation Matrix (r)')
    plt.xticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[:N // 2]], rotation=90)
    plt.yticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[:N // 2]], rotation=0)

    plt.subplot(2, 2, 3)
    sns.heatmap(hbo_cross_thresholded, annot=False, cmap='viridis')
    plt.title('HbO Normalized Cross-Correlation Matrix (max(abs(Cxy)))')
    plt.xticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[N // 2:]], rotation=90)
    plt.yticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[N // 2:]], rotation=0)

    plt.subplot(2, 2, 4)
    sns.heatmap(hbr_cross_thresholded, annot=False, cmap='viridis')
    plt.title('HbR Normalized Cross-Correlation Matrix (max(abs(Cxy)))')
    plt.xticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[:N // 2]], rotation=90)
    plt.yticks(ticks=range(N // 2), labels=[name.split()[0] for name in channel_names[:N // 2]], rotation=0)

    plt.tight_layout()
    plt.show()

    return hbo_r, hbr_r
    
hbo_r, hbr_r = [functional_connectivity(f, True) for f in snirfs]

#[plot_r_matrices(hbo, hbr, snirfs[0].info["ch_names"]) for hbo, hbr in zip(hbo_r, hbr_r)]