import numpy as np


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
preprocessed = [preprocess_snirf(f, z_norm=True) for f in snirfs]

# Channel dictornary -> Consider changing the acutl
channel_dicts = [extract_fnirs_channels(f.get_data(), f.info["ch_names"]) for f in preprocessed]

epoch_dict = [epochs_snirf(f, tmin=0, tmax=10) for f in snirfs]
# 
rois = {"S1"    : ["S1_D1", "S1_D2"],
        "S2"    : ["S4_D4", "S4_D10"],
        "M1"    : ['S14_D12', 'S14_D15'],
        "SMA"   : ['S5_D6', 'S5_D7'],
        "PMA"   : ['S11_D11', 'S12_D2', 'S12_D11'],
        "BROCA" : ['S7_D6', 'S7_D7'],
        }

for i, snirf in enumerate(snirfs):
    print(f"{i} : {paths[i]}")
    



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

