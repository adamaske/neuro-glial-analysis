import os
from neuropipeline.fnirs import fNIRS
from wrappers.eeg import EEG

from preprocessing.eeg import trim, preprocess

def get_fnirs_data(data_folder, load=True):
    
    data = {}
    for i, subject_folder in enumerate(os.scandir(data_folder)):
        if subject_folder.is_file():
            continue
        
        for j, trial_folder in enumerate(os.scandir(subject_folder.path)):
            if trial_folder.is_file():
                continue
            
            for k, entry in enumerate(os.scandir(trial_folder.path)):
                if entry.name.endswith('.snirf'):
                    # Read the .snirf file and return the fNIRS object
                    print(f"subject_{i+1} trial_{j+1} - fNIRS file found: {entry.name}")
                    
                    if f"subject_{i+1}" not in data.keys():
                        data[f"subject_{i+1}"] = {}
                    if not load:
                        data[f"subject_{i+1}"][f"trial_{j+1}"]  = entry.path
                        continue
                    # Load fNIRS
                    fnirs = fNIRS(entry.path)
                    fnirs.remove_features([5, 6])
                    fnirs.trim_from_features(cut_from_first_feature=5, cut_from_last_feature=16)
                    fnirs.remove_features([2])
                    fnirs.preprocess(optical_density=True, 
                                     hemoglobin_concentration=True, 
                                     temporal_filtering=True, 
                                     normalization=True,
                                     detrending=True)
                    data[f"subject_{i+1}"][f"trial_{j+1}"] = fnirs
    
    return data

def get_eeg_data(data_folder):
    
    data = {}
    for i, subject_folder in enumerate(os.scandir(data_folder)):
        if subject_folder.is_file():
            continue
        for j, trial_folder in enumerate(os.scandir(subject_folder.path)):
            if trial_folder.is_file():
                continue
            for k, entry in enumerate(os.scandir(trial_folder.path)):
                if entry.name.endswith('.hdf5'):
                    # Read the .snirf file and return the fNIRS object
                    print(f"subject_{i+1} trial_{j+1} - EEG file found: {entry.name}")
                    
                    if f"subject_{i+1}" not in data.keys():
                        data[f"subject_{i+1}"] = {}
                    # Load fNIRS
                    eeg = EEG(entry.path)
                    
                    eeg = trim(eeg, cut_from_first_feature=5, cut_from_last_feature=20)
                    eeg = preprocess(eeg, 
                                     bandpass=True,
                                     normalization=True,
                                     ica=False,
                                     rerefernce=True)
                    data[f"subject_{i+1}"][f"trial_{j+1}"] = eeg
    return data
