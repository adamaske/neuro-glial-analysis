from wrappers.eeg import EEG, HDF5toEDFConverter

channel_names = []

paths = ["data/Subject02/Trial 1/HeelSubject22025.03.27_11.14.27.hdf5",
         "data/Subject02/Trial 2/HeelSubject22025.03.27_11.17.47.hdf5",
         "data/Subject02/Trial 3/HeelSubject22025.03.27_11.21.29.hdf5",
         "data/Subject02/Trial 4/HeelSubject22025.03.27_11.25.31.hdf5",
         "data/Subject02/Trial 5/HeelSubject22025.03.27_11.29.06.hdf5",
         "data/Subject02/Trial 6/HeelSubject22025.03.27_11.32.54.hdf5",
         ]
eegs = [EEG(path) for path in paths]

markers = { 3 : "Pronation", 4 : "Supination"}

epochs = {}
    
for eeg in eegs:
    eeg.print()
    
    fs = eeg.sampling_frequency
    onsets = eeg.feature_onsets
    descriptions = eeg.feature_descriptions
    assert(len(onsets) == len(descriptions))

    tmin = -2
    tmax = 10

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
   
     
for marker in epochs:
    print(f"{marker} : ", len(epochs[marker]))# : ", len(epoch_data[marker]))
    
exit()

from datasets.eeg import find_eeg_files_in_folder
from preprocessing.eeg import trim, preprocess
from visualization.eeg import inspect_channels, inspect_channel_by_channel
from analysis.eeg import epochs, event_related_potentials, short_time_fourier_transform, multi_channel_stft, continuous_wavelet_transform, multi_channel_cwt


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


