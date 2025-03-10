import numpy as np



import numpy as np
import matplotlib.pyplot as plt
from datasets.fnirs import find_snirf_in_folder, read_snirf
from preprocessing.fnirs import preprocess_snirf        
from analysis.fnirs import epochs_snirf
from hrf import double_gamma_chrf
from beta_values import glm, sliding_window_glm

paths, snirfs = find_snirf_in_folder("data/OMID-13-12-024")
preprocessed = [preprocess_snirf(f) for f in snirfs]

# 
rois = {"S1"    : ["S1_D1", "S1_D2"],
        "S2"    : ["S4_D4", "S4_D10"],
        "M1"    : ['S14_D12', 'S14_D15'],
        "SMA"   : ['S5_D6', 'S5_D7'],
        "PMA"   : ['S11_D11', 'S12_D2', 'S12_D11'],
        "BROCA" : ['S7_D6', 'S7_D7'],
        }

markers = {0: "Rest", 
           1: "Right Foot", 
           2: "Left Foot"
           }
subjects = []
for subject in subjects:
    trials = []
    
    
    # block_avg = { S1_D1  : { "HbO" : [ time series, time series, time series ], "HbR" : "time series, time series, time series"}, ...}
    # channel_dict = { "S1_D1" : { "HbO" : [beta value, beta value, beta value], "HbR" : [beta, beta, beta...]}}

    for trial in trials:
        
        # load snirf
        snirf = read_snirf("test")
        tmin = 0
        tmax = 20
        epoch_dict = epochs_snirf(snirf, tmin, tmax)
        #
        
        pass
    
for i, snirf in enumerate(snirfs):
    print(f"{i} : {paths[i]}")
    
    s_freq = snirf.info["sfreq"]
    tmin = 0
    tmax = 20
    epoch_dict = epochs_snirf(snirf, tmin, tmax)
    
    #for i, roi in enumerate(rois):
    # get the block average for each channel ->
    # average those block averages
    # calculate beta value
    # display in same manner as below
    
    for i, channel_name in enumerate(epoch_dict):
        print(f"{i}. {channel_name}")
        #example S15_D15 850 -> I.e HbO source 15 - detector 15 channel
        
        parts = channel_name.split()
        source_detector = parts[0]
        wavelength = parts[1]
        
        if wavelength == "760" or wavelength == "HbR".lower(): 
            channel_type = "HbR" 
        elif wavelength == "850" or wavelength == "HbO".lower():
            channel_type = "HbO" 
        else:
            raise ValueError(f"Unexpected wavelength: {wavelength}")
        
        event_dict = {1 : {"HbO" : [], "HbR" : []}, 2 : {"HbO" : [], "HbR" : []}} # What events are we interested in ?
        # Example : 1 and 2 where 1 is pronation and 2 is supination
        
        for j, event_description in enumerate(epoch_dict[channel_name]):
            desc = int(event_description)
            #example event "0" : 8, meaning the event marked was conducted 8 times which are stored here
            print(f'    "{event_description}" : {len(epoch_dict[channel_name][event_description])}')

            marker = markers[desc]
            
            if int(event_description) in event_dict: # Only care about the predefined events
                event_blocks = epoch_dict[channel_name][event_description]

                # Plot individual blocks
                
                #plt.figure(figsize=(12, 8))
                for k, block in enumerate(event_blocks):
                    
                    block_time = np.linspace(tmin, tmax, len(block))
                    avergaed_block = (block - np.min(block)) / (np.max(block) - np.min(block))
                    #plt.plot(block_time, avergaed_block, label=f"Block {k+1}")

                # Calculate and plot the average block
                block_avg = np.mean(event_blocks, axis=0)
                block_time = np.linspace(tmin, tmax, len(block_avg))
                
                 # Normalize block_avg to the range [0, 1]
                block_avg_normalized = (block_avg - np.min(block_avg)) / (np.max(block_avg) - np.min(block_avg))

                hrf = double_gamma_chrf(block_time, 6, 16, 1, 1, 1/6)
                beta_value = glm(block_avg_normalized, hrf)
                
                event_dict[desc][channel_type].append(beta_value)
                
                #plt.plot(block_time, block_avg_normalized, label="Average Block", linewidth=3, color='black')
                #plt.plot(block_time, hrf, label="cHRF", linewidth=3, color='red')
                #plt.xlabel("Time", fontsize=15)
                #plt.ylabel("Signal Amplitude", fontsize=15)
                #plt.title(f"Averaged Block: Channel {source_detector} {channel_type}\n{marker} @ {beta_value}", fontsize=15)
                #plt.legend(fontsize=15)
                #plt.grid(True)
                #plt.show()
        
        
        for i, event in enumerate(event_dict):
            
            hbo = event_dict[event]["HbO"]
            hbr = event_dict[event]["HbR"]
            
            avg_hbo = np.mean(hbo)
            avg_hbr = np.mean(hbr)
            
            print(f"Event {event} : {markers[event]}")
            print(f"avg_hbo : ", avg_hbo)
            print(f"avg_hbr : ", avg_hbr)  
    exit()
                
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

