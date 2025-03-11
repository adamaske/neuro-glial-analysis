import numpy as np
import matplotlib.pyplot as plt
from datasets.fnirs import find_snirf_in_folder, read_snirf
from preprocessing.fnirs import preprocess_snirf        
from analysis.fnirs import epochs_snirf
from hrf import double_gamma_chrf
from beta_values import glm, sliding_window_glm

# Montage Information
rois = {"S1"    : ["S1_D1", "S1_D2"],
        "S2"    : ["S4_D4", "S4_D10"],
        "M1"    : ['S14_D12', 'S14_D15'],
        "SMA"   : ['S5_D6', 'S5_D7'],
        "PMA"   : ['S11_D11', 'S12_D2', 'S12_D11'],
        "BROCA" : ['S7_D6', 'S7_D7'],
        }

#Experiment Information
markers = {0: "Rest", 
           1: "Right Foot", 
           2: "Left Foot"
           }

paths, snirfs = find_snirf_in_folder("data/OMID-13-12-024")
snirfs = [preprocess_snirf(f) for f in snirfs]

s_freq = snirfs[0].info["sfreq"]



tmin = 0
tmax = 20

subject_epoch_dict = {}
for i, snirf in enumerate(snirfs):
    print(f"{i} : {paths[i]}")
    
    epoch_dict = epochs_snirf(snirf, tmin, tmax)
   
    for i, channel_name in enumerate(epoch_dict):

        if channel_name not in subject_epoch_dict:
            subject_epoch_dict[channel_name] = epoch_dict[channel_name]
            print(f"Added {channel_name} to subject_epoch_dict")
            continue # We copied the epoch dict, thus we can continue to next channel
        
        for j, event in enumerate(epoch_dict[channel_name]):
            if event in subject_epoch_dict[channel_name]:
      
                for block in epoch_dict[channel_name][event]:
                    subject_epoch_dict[channel_name][event].append(block)


for i, channel_name in enumerate(subject_epoch_dict):
    events = []
    for j, event in enumerate(subject_epoch_dict[channel_name]):
        events.append(event)

    print(f"{channel_name} : ")
    for event in subject_epoch_dict[channel_name]:
        blocks = subject_epoch_dict[channel_name][event]    
        print(f"    {event} : ", len(blocks))

# SUCCESS we have epoched all trials for a subject

# NEXT -> Block averaging
epoch_dict = subject_epoch_dict

subject_channel_beta_values = {}
for i, channel_name in enumerate(epoch_dict):
    
    parts = channel_name.split()
    source_detector = parts[0]
    wavelength = parts[1]
    
    if wavelength == "760" or wavelength == "HbR".lower(): 
        channel_type = "HbR" 
    elif wavelength == "850" or wavelength == "HbO".lower():
        channel_type = "HbO" 
    else:
        raise ValueError(f"Unexpected wavelength: {wavelength}")
    
    if source_detector not in subject_channel_beta_values:
        subject_channel_beta_values[source_detector] = {1 : {"HbO" : 0, "HbR" : 0}, 
                                                        2 : {"HbO" : 0, "HbR" : 0}} 
    
    for j, event_description in enumerate(epoch_dict[channel_name]):
        event_desc = int(event_description)
        if event_desc not in subject_channel_beta_values[source_detector]: # We dont care about the events not in the event dict
            continue
        
        marker = markers[event_desc] # What marker are we looking at 
        
        event_blocks = epoch_dict[channel_name][event_description]
        
        # Pre-normalization ? 
        # Method 1 : Average all the blocks and then calculate beta

        block_avg = np.mean(event_blocks, axis=0)
        block_avg_normalized = (block_avg - np.min(block_avg)) / (np.max(block_avg) - np.min(block_avg))
        
        block_time = np.linspace(tmin, tmax, len(block_avg))

        hrf = double_gamma_chrf(block_time, 6, 16, 1, 1, 1/6)
        beta_value = glm(block_avg_normalized, hrf)

        subject_channel_beta_values[source_detector][event_desc][channel_type] = float(beta_value)
        
        # Method 2 : Calculate beta for all blocks, then average them
        #beta_values = []
        #normalized_blocks = []
        #for k, block in enumerate(event_blocks):
        #    block_normalized = (block - np.min(block)) / (np.max(block) - np.min(block))
        #    hrf = double_gamma_chrf(np.linspace(tmin, tmax, len(block_normalized)), 
        #                            6, 16, 1, 1, 1/6)
        #    beta_values.append(glm(block_normalized, hrf))
        #    normalized_blocks.append(block_normalized)
        #    
        #avg_beta_value = np.mean(beta_values, axis=0)

        # Plotting
        if False:
            plt.figure(figsize=(12, 8))
            for k, block in enumerate(event_blocks):
                block_time = np.linspace(tmin, tmax, len(block))
                block_normalized = (block - np.min(block)) / (np.max(block) - np.min(block))
                plt.plot(block_time, block_normalized, label=f"Block {k+1}")
            plt.plot(block_time, block_avg_normalized, label="Average Block", linewidth=3, color='black')
            plt.plot(block_time, hrf, label="cHRF", linewidth=3, color='red')
            plt.xlabel("Time", fontsize=15)
            plt.ylabel("Signal Amplitude", fontsize=15)
            plt.title(f"Averaged Block: Channel {source_detector} {channel_type}\n{marker} : {beta_value}", fontsize=15)
            plt.legend(fontsize=15)
            plt.grid(True)
            plt.show()    

rest_hbo = []
right_foot_hbo = []
for i, channel_name in enumerate(subject_channel_beta_values):
    
    events = []
    hbos = []
    hbrs = []

    for k, event in enumerate(subject_channel_beta_values[channel_name]):
        events.append(markers[event])

        hbos.append(subject_channel_beta_values[channel_name][event]["HbO"])
        hbrs.append(subject_channel_beta_values[channel_name][event]["HbR"])

    rest_hbo.append(hbos[0])
    right_foot_hbo.append(hbos[1])
    print(f"{channel_name} : {events}")
    print(f"HbO : {hbos}")
    print(f"HbR : {hbrs}")

print(f"Average Right Foot HbO : {np.mean(rest_hbo)}")
print(f"Average Left Foot HbO : {np.mean(right_foot_hbo)}")

exit()

def extract_fnirs_channels(data, channel_names):
    """
    Extracts fNIRS channels and organizes them by source-detector pair and wavelength.

    Args:
        data (numpy.ndarray): A (102 x N) numpy array representing the fNIRS data,
                              where N is the length of the time series.
        channel_names (list of str): A list of channel names like 
                                     ["S1-D1 760", "S2-D1 760", ..., "S1-D1 850", "S2-D1 850", ...].

    Returns:
        dict: A dictionary where keys are source-detector pairs (e.g., "S1-D1") and
              values are dictionaries containing "760" (deoxy) and "850" (oxy) time series.
    """

    if data.shape[0] != len(channel_names):
        raise ValueError(f"Number of channels in data({data.shape[0]}) and channel_names({len(channel_names)}) must match.")
    
    channel_dict = {}

    for i, channel_name in enumerate(channel_names):
        parts = channel_name.split()
        source_detector = parts[0]
        wavelength = parts[1]

        if source_detector not in channel_dict:
            channel_dict[source_detector] = {"HbO": None,"HbR": None}

        # Consider changing the value being the i-index instead of the actual data to save performance
        if wavelength == "760" or wavelength == "HbR".lower():
            channel_dict[source_detector]["HbR"] = data[i, :]
        elif wavelength == "850" or wavelength == "HbO".lower():
            channel_dict[source_detector]["HbO"] = data[i, :]
        else:
            raise ValueError(f"Unexpected wavelength: {wavelength}")

    return channel_dict

from datasets.fnirs import find_snirf_in_folder
from preprocessing.fnirs import preprocess_snirf        
from analysis.fnirs import epochs_snirf
import matplotlib.pyplot as plt

paths, snirfs = find_snirf_in_folder("data/OMID-13-12-024")
preprocessed = [preprocess_snirf(f) for f in snirfs]

# Channel dictornary -> Consider changing the acutl
channel_dicts = [extract_fnirs_channels(f.get_data(), f.info["ch_names"]) for f in preprocessed]

# For every subject ->
# Per channel block averaging for event_descriptor 1 and 2 
# Both hemisphere ROI block averging
# Per hemisphere ROI block averged + LI


# For each ROI we want to extract all the channels from every snirf file. 
for i, roi in enumerate(rois):
    print(f"{roi} : {rois[roi]}")

    for i, snirf in enumerate(preprocessed):
        print(f"{i} : {paths[i]}")

        channel_dict = extract_fnirs_channels(snirf.get_data(), snirf.info["ch_names"])

        # epoch my ROI channels
    roi_channel_data_hbo = []
    roi_channel_data_hbr = []
    
    for j, ch in enumerate(rois[roi]):

        for k, ch_dict in enumerate(channel_dicts):
            if ch in ch_dict:
                roi_channel_data_hbo.append(ch_dict[ch]["HbO"])
                roi_channel_data_hbr.append(ch_dict[ch]["HbR"])
            else:
                print(f"Could not resolve {ch} in ch_dict[{k}]")

    print("HbO : ", len(roi_channel_data_hbo))
    print("HbR : ", len(roi_channel_data_hbr))

