import os
import sys
import numpy as np
from neuropipeline.fnirs import fNIRS
from neuropipeline.eeg import EEG
from functional import composite_correlation, plot_r_matrix
import matplotlib.pyplot as plt

supination_trial_indices = [0, 2, 4]
pronation_trial_indices = [1, 3, 5]

def analyze_trial(trial_folder):
    
def analyze_subject(subject_folder):
    
    for i, trial in enumerate(os.listdir(subject_folder)):
        
        if j in supination_trial_indices:
    # Load the fNIRS data
    fnirs_data = fNIRS("data/Subject01/Trial 1 - Supination/2025-03-24_001.snirf")
    
    # Preprocess the data
    fnirs_data.remove_features([2, 5, 6])  # Remove markers
    fnirs_data.trim_from_features(cut_from_first_feature=0, cut_from_last_feature=16)
    fnirs_data.preprocess(optical_density=True, 
                 hemoglobin_concentration=True,
                 temporal_filtering=True, # Bandpass 0.01 to 0.1 Hz
                 normalization=True, # z-normalization
                 )
    
    # Plot the channels
    fnirs_data.plot_channels()
    
    # Split the data into HbO and HbR
    hbo_data, hbo_names, hbr_data, hbr_names = fnirs_data.split()
    
    # Perform analysis on the data (e.g., connectivity analysis)
    # ...
    # Block Averaging
    
    # Connectivity Analysis
    
    # Neurovascular Coupling
    # Power Envelope Correlation    
    
analyze_subject("C:/Users/Adam/Desktop/Heel Stimulation/Subject01")
    
base_folder = "C:/Users/Adam/Desktop/Heel Stimulation"
subjects = {}

for i, subject_dir in enumerate(os.listdir(base_folder)):
    subjects[f"Subject_{i+1}"] = { "Pronation" : [], "Supination" : []}
    
    trial_dirs = os.listdir(os.path.join(base_folder, subject_dir))
    # Only Keep Folders
    trial_dirs = [trial_dir for trial_dir in trial_dirs if os.path.isdir(os.path.join(base_folder, subject_dir, trial_dir))]
    for j, trial_dir in enumerate(trial_dirs):
        for k, file in enumerate(os.listdir(os.path.join(base_folder, subject_dir, trial_dir))):
            # Check if "file" is a folder or file
            if os.path.isdir(os.path.join(base_folder, subject_dir, trial_dir, file)):
                continue  # Skip if it's a directory
            if file.endswith(".snirf"):
                print(f"Subject {i+1} Trial {j+1} File {k+1}: {file}")
                
                if j in supination_trial_indices:
                    subjects[f"Subject_{i+1}"]["Supination"].append(os.path.join(base_folder, subject_dir, trial_dir, file))
                if j in pronation_trial_indices:
                    subjects[f"Subject_{i+1}"]["Pronation"].append(os.path.join(base_folder, subject_dir, trial_dir, file))

for i, subject in enumerate(subjects):
    print(f"{subject} -> Supination : {len(subjects[subject]['Supination'])}, Pronation : {len(subjects[subject]['Pronation'])}")


fnirs_data = {}
for i, subject in enumerate(subjects):
    
    fnirs_data[subject] = {}
    fnirs_data[subject]["Supination"] = []
    fnirs_data[subject]["Pronation"] = []
    
    for j, trial in enumerate(subjects[subject]["Supination"]):
        print(f"Loading {subject} Supination Trial {j+1} : {subjects[subject]['Supination'][j]}")
        fnirs_data[subject]["Supination"].append(fNIRS(subjects[subject]["Supination"][j]))
    
    for j, trial in enumerate(subjects[subject]["Pronation"]):
        print(f"Loading {subject} Pronation Trial {j+1} : {subjects[subject]['Pronation'][j]}")
        fnirs_data[subject]["Pronation"].append(fNIRS(subjects[subject]["Pronation"][j]))
        
# Preprocessing
for i, subject in enumerate(fnirs_data):
    print("Preprocessing Subject ", subject)
    for j, trial in enumerate(fnirs_data[subject]["Supination"]):
        print(f"Preprocessing {subject} Supination Trial {j+1}")
        fnirs_data[subject]["Supination"][j].remove_features([2, 5, 6]) # Remove markers
        fnirs_data[subject]["Supination"][j].trim_from_features(cut_from_first_feature=0, cut_from_last_feature=16)
        fnirs_data[subject]["Supination"][j].preprocess(optical_density=True, 
                     hemoglobin_concentration=True,
                     temporal_filtering=True, # Bandpass 0.01 to 0.1 Hz
                     normalization=True, # z-normalization
                     )
        
    for j, trial in enumerate(fnirs_data[subject]["Pronation"]):
        print(f"Preprocessing {subject} Pronation Trial {j+1}")
        fnirs_data[subject]["Pronation"][j].remove_features([2, 5, 6]) # Remove markers
        fnirs_data[subject]["Pronation"][j].trim_from_features(cut_from_first_feature=0, cut_from_last_feature=16)
        fnirs_data[subject]["Pronation"][j].preprocess(optical_density=True, 
                     hemoglobin_concentration=True,
                     temporal_filtering=True, # Bandpass 0.01 to 0.1 Hz
                     normalization=True, # z-normalization
                     )    
# GLM

# Block Averaging
for i, subject in enumerate(fnirs_data):
    print("Blck Averaging Subject ", subject)
    # Supination
    supination_trials = fnirs_data[subject]["Supination"]
    # Pronation
    pronation_trials = fnirs_data[subject]["Pronation"]
    # Inspect
    for s, trial in enumerate(supination_trials):
        trial.plot_channels()
    for s, trial in enumerate(pronation_trials):
        trial.plot_channels()
    
    
    
# Connectivity Analysis

# Neurovascular Coupling
# Power Envelope Correlation
# Phase-amplitude coupling

exit()
# Loading files -> fNIRS
sub_01_trial_1 = fNIRS("data/Subject01/Trial 1 - Supination/2025-03-24_001.snirf")
sub_01_trial_2 = fNIRS("data/Subject01/Trial 2 - Pronation/2025-03-24_002.snirf")
sub_01_trial_3 = fNIRS("data/Subject01/Trial 3 - Supination/2025-03-24_003.snirf")
sub_01_trial_4 = fNIRS("data/Subject01/Trial 4 - Pronation/2025-03-24_004.snirf")
sub_01_trial_5 = fNIRS("data/Subject01/Trial 5 - Supination/2025-03-24_005.snirf")
sub_01_trial_6 = fNIRS("data/Subject01/Trial 6 - Pronation/2025-03-24_006.snirf")

sub_02_trial_1 = fNIRS("data/Subject02/Trial 1/2025-03-27_002.snirf")
sub_02_trial_2 = fNIRS("data/Subject02/Trial 2/2025-03-27_003.snirf")
sub_02_trial_3 = fNIRS("data/Subject02/Trial 3/2025-03-27_004.snirf")
sub_02_trial_4 = fNIRS("data/Subject02/Trial 4/2025-03-27_005.snirf")
sub_02_trial_5 = fNIRS("data/Subject02/Trial 5/2025-03-27_006.snirf")
sub_02_trial_6 = fNIRS("data/Subject02/Trial 6/2025-03-27_007.snirf")

sub_03_trial_1 = fNIRS("data/Subject03/Trial 1/2025-04-01_002.snirf")
sub_03_trial_2 = fNIRS("data/Subject03/Trial 2/2025-04-01_003.snirf")
sub_03_trial_3 = fNIRS("data/Subject03/Trial 3/2025-04-01_005.snirf")
sub_03_trial_4 = fNIRS("data/Subject03/Trial 4/2025-04-01_007.snirf")
sub_03_trial_5 = fNIRS("data/Subject03/Trial 5/2025-04-01_008.snirf")
sub_03_trial_6 = fNIRS("data/Subject03/Trial 6/2025-04-01_009.snirf")

cvit_sub_02_trial_1 = fNIRS("data/BeforeCVit_Subject02/Lower Body/Trial 1 - Supination/2025-04-10_001.snirf")
cvit_sub_02_trial_2 = fNIRS("data/BeforeCVit_Subject02/Lower Body/Trial 2 - Pronation/2025-04-10_002.snirf")    
cvit_sub_02_trial_3 = fNIRS("data/BeforeCVit_Subject02/Lower Body/Trial 3 - Supination/2025-04-10_003.snirf")
cvit_sub_02_trial_4 = fNIRS("data/BeforeCVit_Subject02/Lower Body/Trial 4 - Pronation/2025-04-10_004.snirf")
cvit_sub_02_trial_5 = fNIRS("data/BeforeCVit_Subject02/Lower Body/Trial 5 - Supination/2025-04-10_005.snirf")
cvit_sub_02_trial_6 = fNIRS("data/BeforeCVit_Subject02/Lower Body/Trial 6 - Pronation/2025-04-10_006.snirf")

# Loading files -> EEG

subject_01_supination = [sub_01_trial_1, sub_01_trial_3, sub_01_trial_5]
subject_01_pronation = [sub_01_trial_2, sub_01_trial_4, sub_01_trial_6]

subject_02_supination = [sub_02_trial_1, sub_02_trial_3, sub_02_trial_5]
subject_02_pronation = [sub_02_trial_2, sub_02_trial_4, sub_02_trial_6]

subject_03_supination = [sub_03_trial_1, sub_03_trial_3, sub_03_trial_5]
subject_03_pronation = [sub_03_trial_2, sub_03_trial_4, sub_03_trial_6]

subject_cv_supination = [cvit_sub_02_trial_1, cvit_sub_02_trial_3, cvit_sub_02_trial_5]
subject_cv_pronation = [cvit_sub_02_trial_2, cvit_sub_02_trial_4, cvit_sub_02_trial_6]

supination = [subject_01_supination, subject_02_supination, subject_03_supination, subject_cv_supination]  
pronation = [subject_01_pronation, subject_02_pronation, subject_03_pronation, subject_cv_pronation]

for i, subject in enumerate(supination + pronation):
    for j, trial in enumerate(subject):
        trial.remove_features([2, 5, 6]) # Remove markers 
        
        trial.trim_from_features(cut_from_first_feature=1, cut_from_last_feature=16) 
        
        trial.preprocess(optical_density=True, 
                     hemoglobin_concentration=True,
                     temporal_filtering=True, # Bandpass 0.01 to 0.1 Hz
                     normalization=True, # z-normalization
                     )
        trial.print()
       
# All files are now preprocessed and ready for analysis

for i, subject in enumerate(supination):
    for j, trial in enumerate(subject):
        
        hbo_data, hbo_names, hbr_data, hbr_names = trial.split()
        
        # Epoching
        hbo_epochs = []
        hbr_epochs = []
        
        
        markers = {3 : "Pronation", 
                   4 : "Supination"
                  }
        tmin = 0
        tmax = 20
        
        for k, desc in enumerate(trial.feature_descriptions):
            onset = trial.feature_onsets[k]
            
            marker = markers[desc] 
            
            start = int((onset + tmin) * trial.sampling_frequency)
            end = int((onset + tmax) * trial.sampling_frequency)
            print("start : ", start, "end : ", end)
            hbo_epoch = hbo_data[:, start:end]
            hbr_epoch = hbr_data[:, start:end]
            print("hbo epoch : ", hbo_epoch.shape)
            print("hbr epoch : ", hbr_epoch.shape)
            hbo_epochs.append(hbo_epoch)
            hbr_epochs.append(hbr_epoch)
        
        plt.figure(figsize=(8, 12))
        plt.title(f"Subject {i+1} Supination Trial {j+1}/{len(subject)} : HbO / HbR")
        
        for k, epoch in enumerate(zip(hbo_epochs, hbr_epochs)):
            hbo_epoch, hbr_epoch = epoch
            
            # Calculate the composite correlation for each epoch
            hbo_r = composite_correlation(hbo_epoch, hbo_names, trial.sampling_frequency, None, 0.01, 0.1)
            hbr_r = composite_correlation(hbr_epoch, hbr_names, trial.sampling_frequency, None, 0.01, 0.1)

            combined_r = np.zeros((hbo_r.shape[0], hbo_r.shape[0]))
            # Fill the lower triangle with hbo_r
            combined_r = np.tril(hbo_r)
            # Fill the upper triangle with hbr_r
            combined_r += np.triu(hbr_r, k=1)
            
            # Plot the combined matrix
            plt.subplot(1, len(subject), k + 1)
            plot_r_matrix(combined_r, hbo_names, f"Epoch {k + 1}")
        
        plt.show()
        
exit()