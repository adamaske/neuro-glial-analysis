from wrappers.eeg import EEG
from datasets.eeg import find_eeg_files_in_folder
from preprocessing.eeg import trim, preprocess
from visualization.eeg import inspect_channels, inspect_channel_by_channel
from analysis.eeg import event_related_potentials, short_time_fourier_transform, multi_channel_stft, continuous_wavelet_transform, multi_channel_cwt

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, coherence, hilbert
from scipy.stats import pearsonr

from scipy.signal import butter, lfilter
from scipy.signal import hilbert
paths = ["data/Subject01/Trial 1 - Supination/heel2025.03.24_14.27.28.hdf5",
         "data/Subject01/Trial 2 - Pronation/heel2025.03.24_14.31.33.hdf5",
         "data/Subject01/Trial 3 - Supination/heel2025.03.24_14.36.01.hdf5",
         "data/Subject01/Trial 4 - Pronation/heel2025.03.24_14.40.18.hdf5",
         "data/Subject01/Trial 5 - Supination/heel2025.03.24_14.45.30.hdf5",
         "data/Subject01/Trial 6 - Pronation/heel2025.03.24_14.50.12.hdf5",
         "data/Subject02/Trial 1/HeelSubject22025.03.27_11.14.27.hdf5",
         "data/Subject02/Trial 2/HeelSubject22025.03.27_11.17.47.hdf5",
         "data/Subject02/Trial 3/HeelSubject22025.03.27_11.21.29.hdf5",
         "data/Subject02/Trial 4/HeelSubject22025.03.27_11.25.31.hdf5",
         "data/Subject02/Trial 5/HeelSubject22025.03.27_11.29.06.hdf5",
         "data/Subject02/Trial 6/HeelSubject22025.03.27_11.32.54.hdf5",
         ]

supination_paths = [#"data/Subject01/Trial 1 - Supination/heel2025.03.24_14.27.28.hdf5",
                    #"data/Subject01/Trial 3 - Supination/heel2025.03.24_14.36.01.hdf5",
                    #"data/Subject01/Trial 5 - Supination/heel2025.03.24_14.45.30.hdf5",
         "data/Subject02/Trial 1/HeelSubject22025.03.27_11.14.27.hdf5",
         "data/Subject02/Trial 3/HeelSubject22025.03.27_11.21.29.hdf5",
         "data/Subject02/Trial 5/HeelSubject22025.03.27_11.29.06.hdf5",
                    ]
pronation_paths = [#"data/Subject01/Trial 2 - Pronation/heel2025.03.24_14.31.33.hdf5",
                   #"data/Subject01/Trial 4 - Pronation/heel2025.03.24_14.40.18.hdf5",
                   #"data/Subject01/Trial 6 - Pronation/heel2025.03.24_14.50.12.hdf5",
         "data/Subject02/Trial 2/HeelSubject22025.03.27_11.17.47.hdf5",
         "data/Subject02/Trial 4/HeelSubject22025.03.27_11.25.31.hdf5",
         "data/Subject02/Trial 6/HeelSubject22025.03.27_11.32.54.hdf5",
                   ]

supination_trials = [EEG(path) for path in supination_paths]
pronation_trials = [EEG(path) for path in pronation_paths]

[eeg.trim_from_features(5, 10) for eeg in supination_trials]
[eeg.trim_from_features(5, 10) for eeg in pronation_trials]

supination_trials = [preprocess(e, ica=False, rerefernce=True) for e in supination_trials]
pronation_trials = [preprocess(e, ica=False, rerefernce=True) for e in pronation_trials]

def correlate_channels(channel_data, sampling_rate):
    
    num_channels = channel_data.shape[0]
    r = np.zeros((num_channels, num_channels))
    cc = np.zeros((num_channels, num_channels))
    for i, channel1 in enumerate(channel_data):
        for j, channel2 in enumerate(channel_data):
            r_, p = pearsonr(channel_data[i], channel_data[j])
            r[i, j] = r_

            Cxy = correlate(channel_data[i], channel_data[j], mode='full')
            normalization_factor = np.sqrt(np.sum(channel1**2) * np.sum(channel2**2))
            if normalization_factor == 0:
                normalized_Cxy =  np.zeros_like(Cxy) #Handle zero division case.
            else:
                normalized_Cxy = Cxy / normalization_factor
            cc[i, j] = np.max(np.abs(normalized_Cxy))
           
            #plt.subplot(1, 2, 1)
            #plt.plot(channel1)
            #plt.subplot(1, 2, 2)
            #plt.plot(channel2)
            #print("r: ", r_)
            #print("p:", p)
            #plt.show()
    return r, cc

band_ranges_spec = {
        "Broadband (0.5-12 Hz)": (0.5, 12),
        "Delta (0.5-4 Hz)": (0.5, 4),
        "Theta (4-8 Hz)": (4, 8),
        "Alpha (8-12 Hz)": (8, 12),
        "Beta (12-30 Hz)": (12, 30),
        "Gamma (30-100 Hz)": (30, 100)
    }
#band_ranges_spec = {
#    'Delta': (1, 4),
#    'Theta': (4, 7.5),
#    'Theta1': (4, 6),
#    'Theta2': (6, 7.5),
#    'Alpha': (7.5, 13),
#    'Alpha1': (7.5, 9.5),
#    'Alpha2': (9.5, 11),
#    'Alpha3': (11, 13),
#    'Beta': (13, 35),
#    'Beta1': (13, 20),
#    'Beta2': (20, 35)
#}
def coherence_channels(channel_data, sampling_rate):
    num_channels = channel_data.shape[0]
    
    mean_band_coherence = {}
    
    for i, channel1 in enumerate(channel_data):
        for j, channel2 in enumerate(channel_data):
            
            f, Cxy = coherence(channel_data[i], channel_data[j], fs=sampling_rate)

            for band, (low, high) in band_ranges_spec.items():
                
                if band not in mean_band_coherence:
                    mean_band_coherence[band] = np.zeros((num_channels, num_channels))
                
                band_indices = np.where((f >= low) & (f <= high))[0]
                if len(band_indices) > 0:
                    mean_coherence = np.mean(Cxy[band_indices])
                else:
                    mean_coherence = 0
                
                mean_band_coherence[band][i, j] = mean_coherence
            
            
            #plt.figure(figsize=(10, 6))
            #plt.subplot(1, 2, 1)
            #plt.plot(f, Cxy)
            #plt.title(f"Channel {i} - Channel {j}")
            #plt.xlabel("Frequencies (Hz)")
            #plt.ylabel("Coherence")
            #plt.subplot(1, 2, 2)
            #plt.bar(band_names, band_coherences)
            #plt.title(f"Coherence between Channels (Bands)")
            #plt.xlabel("Frequency Bands")
            #plt.ylabel("Mean Coherence")
            #plt.ylim(0, 1) #ensure y axis is between 0 and 1.
            #plt.show()
    
    
    return mean_band_coherence


def phase_lag_channels(channel_data, sampling_rate, band_ranges_spec):
    """
    Calculates the phase lag between all pairs of channels within specified frequency bands.

    Args:
        channel_data: NumPy array of shape (num_channels, num_samples).
        sampling_rate: Sampling rate of the EEG data (Hz).
        band_ranges_spec: Dictionary specifying frequency bands and their ranges.
                           Example: {'Delta': (1, 4), 'Theta': (4, 8), ...}

    Returns:
        Dictionary where keys are tuples (channel1_index, channel2_index, band_name)
        and values are the mean phase lag in radians.
    """

    num_channels = channel_data.shape[0]
    band_phase_lag = {}

    for i in range(num_channels):
        for j in range(num_channels):
            channel1 = channel_data[i]
            channel2 = channel_data[j]

            analytic_signal1 = hilbert(channel1)
            analytic_signal2 = hilbert(channel2)

            phase1 = np.angle(analytic_signal1)
            phase2 = np.angle(analytic_signal2)

            for band, (low, high) in band_ranges_spec.items():
                # Bandpass filter the phase differences (optional but recommended)
                # We can't directly filter the phase, so we filter the original signals, then recompute the phase.
                def butter_bandpass(lowcut, highcut, fs, order=3):
                    nyq = 0.5 * fs
                    low = lowcut / nyq
                    high = highcut / nyq
                    b, a = butter(order, [low, high], btype='band')
                    return b, a

                def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
                    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
                    y = lfilter(b, a, data)
                    return y

                filtered_channel1 = butter_bandpass_filter(channel1, low, high, sampling_rate)
                filtered_channel2 = butter_bandpass_filter(channel2, low, high, sampling_rate)

                filtered_analytic_signal1 = hilbert(filtered_channel1)
                filtered_analytic_signal2 = hilbert(filtered_channel2)

                filtered_phase1 = np.angle(filtered_analytic_signal1)
                filtered_phase2 = np.angle(filtered_analytic_signal2)

                phase_diff = filtered_phase1 - filtered_phase2
                mean_phase_lag = np.mean(np.unwrap(phase_diff))  # Unwrap to avoid jumps

                band_phase_lag[(i, j, band)] = mean_phase_lag 

    return band_phase_lag
supination_r = []
supination_cc = []
supination_coh = []
pronation_r = []
pronation_cc = []
pronation_coh = []
for i, trial in enumerate(supination_trials):
    
    
    channel_data = trial.channel_data
    num_channels = channel_data.shape[0]
    

    r, cc = correlate_channels(channel_data, trial.sampling_frequency)
    band_coherence = coherence_channels(channel_data, trial.sampling_frequency)
    
    supination_r.append(r)
    supination_cc.append(cc)
    supination_coh.append(band_coherence["Broadband (0.5-12 Hz)"])

for i, trial in enumerate(pronation_trials):
    
    
    channel_data = trial.channel_data
    num_channels = channel_data.shape[0]
    

    r, cc = correlate_channels(channel_data, trial.sampling_frequency)
    band_coherence = coherence_channels(channel_data, trial.sampling_frequency)
    
    pronation_r.append(r)
    pronation_cc.append(cc)
    pronation_coh.append(band_coherence["Broadband (0.5-12 Hz)"])  
       #phase_lags = phase_lag_channels(channel_data, trial.sampling_frequency, band_ranges_spec)
   #
# P#int results
   #for (channel1, channel2, band), lag in phase_lags.items():
   # 
   #    print(f"Phase lag between channel {channel1} and {channel2} in band {band}: {lag:.4f} radians")
    
    
    #for band in band_coherence:
    #    plt.title(f"Coherence : {band}")
    #    plt.imshow(band_coherence[band], cmap='RdBu_r', vmin=-1, vmax=1)
    #    plt.colorbar(label="Correlation")
    #    plt.xlabel("Channel")
    #    plt.ylabel("Channel")
    #    plt.show()
    #
    #
    #plt.figure(figsize=(8, 6))
    #
    #plt.subplot(1, 2, 1)
    #plt.imshow(r, cmap='RdBu_r', vmin=-1, vmax=1)
    #plt.colorbar(label="Correlation")
    #plt.title(f"Supination : Pearson Correlation Coefficients")
    #plt.xlabel("Channel")
    #plt.ylabel("Channel")
    #plt.xticks(range(num_channels))
    #plt.yticks(range(num_channels))
    #plt.tight_layout()
    #
    #plt.subplot(1, 2, 2)
    #plt.imshow(cc, cmap='RdBu_r', vmin=-1, vmax=1)
    #plt.colorbar(label="Correlation")
    #plt.title(f"Supination : Cross-Correlation")
    #plt.xlabel("Channel")
    #plt.ylabel("Channel")
    #plt.xticks(range(num_channels))
    #plt.yticks(range(num_channels))
    #plt.tight_layout()
    #plt.show()

#Group Level Analysis
sup_mean_r = np.mean(supination_r, axis=0)
sup_mean_cc = np.mean(supination_cc, axis=0)
sup_mean_coh = np.mean(supination_coh, axis=0)

pro_mean_r = np.mean(pronation_r, axis=0)
pro_mean_cc = np.mean(pronation_cc, axis=0)
pro_mean_coh = np.mean(pronation_coh, axis=0)

# Strong Threshold 0.7
threshold = 0.5
sup_thresholded_r = np.where(sup_mean_r > threshold, sup_mean_r, 0)
sup_thresholded_cc = np.where(sup_mean_cc > threshold, sup_mean_cc, 0)
sup_thresholded_coh = np.where(sup_mean_coh > threshold, sup_mean_coh, 0)
pro_thresholded_r = np.where(pro_mean_r > threshold, pro_mean_r, 0)
pro_thresholded_cc = np.where(pro_mean_cc > threshold, pro_mean_cc, 0)
pro_thresholded_coh = np.where(pro_mean_coh > threshold, pro_mean_coh, 0)
sup_mean_r =  sup_thresholded_r  
sup_mean_cc = sup_thresholded_cc 
sup_mean_coh = sup_thresholded_coh 
pro_mean_r =  pro_thresholded_r  
pro_mean_cc = pro_thresholded_cc 
pro_mean_coh = pro_thresholded_coh 
plt.figure(figsize=(8, 6))
plt.title(" Supination vs. Pronation : Pearson, Cross-Correlation, Coherence")
plt.subplot(2, 3, 1)
plt.imshow(sup_mean_r, cmap='RdBu_r', vmin=-1, vmax=1)
#plt.colorbar(label="Correlation")
plt.title(f"Supination : Pearson Correlation Coefficients")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.xticks(range(num_channels))
plt.yticks(range(num_channels))

plt.subplot(2, 3, 2)
plt.imshow(sup_mean_cc, cmap='RdBu_r', vmin=-1, vmax=1)
#plt.colorbar(label="Correlation")
plt.title(f"Supination : Cross-Correlation")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.xticks(range(num_channels))
plt.yticks(range(num_channels))
    
plt.subplot(2, 3, 3)
plt.imshow(sup_mean_coh, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.title(f"Supination : 0.5-12 Hz Coherence")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.xticks(range(num_channels))
plt.yticks(range(num_channels))

plt.subplot(2, 3, 4)
plt.imshow(pro_mean_r, cmap='RdBu_r', vmin=-1, vmax=1)
#plt.colorbar(label="Correlation")
plt.title(f"Pronation : Pearson Correlation Coefficients")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.xticks(range(num_channels))
plt.yticks(range(num_channels))

plt.subplot(2, 3, 5)
plt.imshow(pro_mean_cc, cmap='RdBu_r', vmin=-1, vmax=1)
#plt.colorbar(label="Correlation")
plt.title(f"Pronation : Cross-Correlation")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.xticks(range(num_channels))
plt.yticks(range(num_channels))
    
plt.subplot(2, 3, 6)
plt.imshow(pro_mean_coh, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.title(f"Pronation : 0.5-12 Hz Coherence")
plt.xlabel("Channel")
plt.ylabel("Channel")
plt.xticks(range(num_channels))
plt.yticks(range(num_channels))

plt.tight_layout()
plt.show()
exit()
eegs = [EEG(path) for path in paths]
eegs = [preprocess(e, normalization=True, bandpass=True, ica=False, rerefernce=True) for e in eegs]
#[inspect_channels(e.channel_data, e.sampling_frequency) for e in eegs]

markers = { 3 : "Pronation", 4 : "Supination"} # NOTE : This can be easily converted into a  P-Indent vs S-Indent, and P-Shear vs S-Shear 
epochs = {}


for eeg in eegs:
    eeg.print()
    
    fs = eeg.sampling_frequency
    onsets = eeg.feature_onsets
    descriptions = eeg.feature_descriptions
    assert(len(onsets) == len(descriptions))

    tmin = 0
    tmax = 2

    for i, desc in enumerate(descriptions):
        if not desc in markers:
            continue
        marker = markers[desc]
        onset = onsets[i]

        start_frame = int(onset + float(tmin*fs))
        end_frame = int(onset + float(tmax*fs))

        start_seconds = float(onset / fs) + tmin
        end_seconds = float(onset / fs) + tmax 

        epoch_data = eeg.channel_data[:, start_frame:end_frame]
        print(f"{desc} : {markers[desc]} @ {onset} / {float(onset / fs):.2f} s")
        print(f"Epoch {epoch_data.shape} @ {start_frame} - {end_frame} / {start_seconds} - {end_seconds}")
        if marker not in epochs:
            epochs[marker] = []
        epochs[marker].append(epoch_data)
   
matrices = [] 
for marker in epochs:
    print(f"{marker} : ", len(epochs[marker]))# : ", len(epoch_data[marker]))
    
    # Calculate FC for this marker
    marker_epochs  = epochs[marker]
    

    num_epochs = len(marker_epochs)
    num_channels = marker_epochs[0].shape[0]  # Assuming all epochs have the same number of channels

    # Initialize an array to store the correlation matrices for each epoch
    r_matrices = np.zeros((num_epochs, num_channels, num_channels))
    cc_matrices = np.zeros((num_epochs, num_channels, num_channels))
    for i, epoch in enumerate(marker_epochs):
        r_matrices[i] = np.corrcoef(epoch)  # Calculate correlation matrix for each epoch
        
        for j, channel1 in enumerate(epoch):
            for k, channel2 in enumerate(epoch):
                
                Cxy = correlate(channel1, channel2, mode='full')
                normalization_factor = np.sqrt(np.sum(channel1**2) * np.sum(channel2**2))
                if normalization_factor == 0:
                    normalized_Cxy =  np.zeros_like(Cxy) #Handle zero division case.
                else:
                    normalized_Cxy = Cxy / normalization_factor
                    
                cc_matrices[i, j, k] = np.max(np.abs(normalized_Cxy))
       
    r_mean = np.mean(r_matrices, axis=0)
    matrices.append(r_mean)

    cc_mean = np.mean(cc_matrices, axis=0)
    
    plt.figure(figsize=(8, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(r_mean, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(f"{marker} : Pearson product-moment correlation coefficients")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    plt.xticks(range(num_channels))
    plt.yticks(range(num_channels))
    plt.tight_layout()
    
    plt.subplot(1, 2, 2)
    plt.imshow(cc_mean, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(f"{marker} : Cross-correlation")
    plt.xlabel("Channel")
    plt.ylabel("Channel")
    plt.xticks(range(num_channels))
    plt.yticks(range(num_channels))
    plt.tight_layout()
    plt.show()
    

exit()

#eeg = EEG("data/omid_eegfnirs/TwoHandOmid2025.03.13_13.02.35.hdf5")
eeg = EEG("data/subject01/2025-03-24_001_PREPROCESSED.snirf")
eeg.print() # Inspect information retreived from hdf5 file. 

trim(eeg=eeg, cut_from_start=10, cut_from_end=10)
preprocess(eeg, bandpass=True, normalization=False, ica=False)

inspect_channels(eeg.channel_data, eeg.sampling_frequency)
inspect_channel_by_channel(eeg.channel_data, eeg.sampling_frequency)

eeg.print()
#new_eeg = eeg.write("data/FeaturesTesting_PROCESSED.hdf5")
#new_eeg.print()
exit()

hdf = read_hdf5("data/Adam_RestingState.hdf5")
samples, sampling_frequency, channel_num, features_onset, features_order, features_desc = parse_hdf5(hdf)



trimmed = trim(samples, sampling_frequency, cut_from_start=3, cut_from_end=3)                                       # NOTE : How does this trimming affect feature timings ? 
preprocessed = preprocess(trimmed, sampling_frequency)

#continuous_wavelet_transform(preprocessed[0], sampling_frequency)

short_time_fourier_transform(preprocessed[0], sampling_frequency)

exit()
epochs, order = epochs(preprocessed, sampling_frequency, -2, 5, features_onset, features_order, features_desc)

inspect_channels(preprocessed, sampling_frequency)
inspect_channel_by_channel(preprocessed, sampling_frequency)

write_hdf5_replace_data_keep_stats(data=preprocessed, original_hdf=hdf)



exit()    
# TODO : Preprocessing
# Channel selection : Remove non ROI channels
# Channel rejection : What algorithm?
# Movement correction : ? 
# Eyeblink removal : ICA -> how many components ?
# Feature / Markers parsing from XML 

# TODO : Analysis
# Time-Frequency domain analysis : STFT, Wavelet Transform
# Event-Related Potentials : 
# ROI -> pearson correlation
# Threshold hemodynamic response function, eeg erp amplitude -> pearson correlation
# Granger connectivity of EEG and fNIRS ROIs
# ROI comparison


