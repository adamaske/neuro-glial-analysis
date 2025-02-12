from datasets.eeg import read_hdf5, find_hdf5_in_folder, parse_hdf5, write_hdf5_replace_data_keep_stats
from preprocessing.eeg import trim, preprocess
from visualization.eeg import inspect_channels, inspect_channel_by_channel

hdf = read_hdf5("data/Adam_RestingState.hdf5")
samples, sampling_frequency, channel_num = parse_hdf5(hdf)

trimmed = trim(samples, sampling_frequency, cut_from_start=3, cut_from_end=3)
preprocessed = preprocess(trimmed, sampling_frequency)
# TODO : 
# Time-Frequency domain analysis : STFT, Wavelet Transform
# Event-Related Potentials : 
# Channel selection : Remove non ROI channels
# Channel rejection : What algorithm?
# Motorion correction : 
# Eyeblink removal : ICA -> how many components ?
# Feature / Markers parsing from XML 
inspect_channels(preprocessed, sampling_frequency)
inspect_channel_by_channel(preprocessed, sampling_frequency)

inspect_channels(preprocessed, sampling_frequency)
inspect_channel_by_channel(preprocessed, sampling_frequency)

write_hdf5_replace_data_keep_stats(preprocessed, hdf, "data/Adam_RestingState_Preprocessed.hdf5")



    


