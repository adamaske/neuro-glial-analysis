from datasets.eeg import find_eeg_files_in_folder
from preprocessing.eeg import trim, preprocess
from visualization.eeg import inspect_channels, inspect_channel_by_channel
from analysis.eeg import epochs, event_related_potentials, short_time_fourier_transform, multi_channel_stft, continuous_wavelet_transform, multi_channel_cwt
from wrappers.eeg import EEG

eeg = EEG("data/omid_eegfnirs/TwoHandOmid2025.03.13_13.02.35.hdf5")
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


