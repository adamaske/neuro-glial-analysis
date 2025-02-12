from datasets.eeg import read_hdf5, find_hdf5_in_folder, parse_hdf5, write_hdf5_replace_data_keep_stats
from preprocessing.eeg import trim, preprocess
from visualization.eeg import inspect_channels, inspect_channel_by_channel
from analysis.eeg import epochs, event_related_potentials

hdf = read_hdf5("data/Adam_RestingState.hdf5")
samples, sampling_frequency, channel_num, features_onset, features_order, features_desc = parse_hdf5(hdf)


trimmed = trim(samples, sampling_frequency, cut_from_start=3, cut_from_end=3)
# NOTE : How does this trimming affect feature timings ? 
# We may need to onset -3 second for each feature
# trim offsets ? 
  
preprocessed = preprocess(trimmed, sampling_frequency)
#Problems :
# Normalization : Z-Normalization or Baseline correction ? 

epochs(preprocessed, sampling_frequency, -2, 10, features_onset, features_order, features_desc)

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

inspect_channels(preprocessed, sampling_frequency)
inspect_channel_by_channel(preprocessed, sampling_frequency)

inspect_channels(preprocessed, sampling_frequency)
inspect_channel_by_channel(preprocessed, sampling_frequency)

write_hdf5_replace_data_keep_stats(preprocessed, hdf, "data/Adam_RestingState_Preprocessed.hdf5")



    


