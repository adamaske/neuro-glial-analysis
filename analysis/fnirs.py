import numpy as np
import matplotlib.pyplot as plt
from mne.io.snirf._snirf import RawSNIRF
from mne import Annotations


def epochs_snirf(snirf:RawSNIRF, tmin, tmax):
    """
    
    Returns:
        
    """
    data = snirf.get_data()
    s_freq = snirf.info["sfreq"]
    ch_names = snirf.info["ch_names"]

    annotations = snirf._annotations
    event_order = annotations.description
    event_onsets = annotations.onset
    # example = {"S1_D1 hbo" : { 0: [np.array, np.array],
    #                            1: [np.array, np.array],
    #                            2: ....
    #                           }, 
    #           "S1_D2 hbo" : { 0: ...,
    #                           1: ...}
    #      
    epoch_dict = {}
    for ch_idx, ch_name in enumerate(ch_names): # Fill channel names
        epoch_dict[ch_name] = {}

    tmin_sample = int(tmin * s_freq)
    tmax_sample = int(tmax * s_freq)

    for event_idx, event_onset in enumerate(event_onsets):
        event_desc = event_order[event_idx]

        onset_sample = int(event_onset * s_freq) # Onset frame

        start_sample = onset_sample + tmin_sample
        end_sample = onset_sample + tmax_sample

        if start_sample < 0 or end_sample > data.shape[1]:
            continue  # Skip epochs that go out of bounds

        epoch_data = data[:, start_sample:end_sample]

        for ch_idx, ch_name in enumerate(ch_names):

            if event_desc not in epoch_dict[ch_name]: # 
                epoch_dict[ch_name][event_desc] = []
                
            epoch_dict[ch_name][event_desc].append(epoch_data[ch_idx, :])


    return epoch_dict

def block_averaging_snirf(snirf: RawSNIRF, tmin=2, tmax=15, event_descriptions=None):
    """
    Performs block averaging on epoched SNIRF data, using event descriptions as blocks.

    Args:
        snirf (RawSNIRF): The RawSNIRF object.
        tmin (float): Start time before event onset (in seconds).
        tmax (float): End time after event onset (in seconds).
        event_descriptions (list, optional): A list of event descriptions to average. If None, averages all event descriptions.

    Returns:
        dict: A dictionary containing block-averaged data, organized by channel and event description.
              The dictionary has the following structure:
              {
                  "channel_name": {
                      "event_description_1": np.array(averaged_epoch),
                      "event_description_2": np.array(averaged_epoch),
                      ...
                  },
                  ...
              }
    """
    epoch_dict = epochs_snirf(snirf, tmin, tmax)
    averaged_data = {}

    for ch_name, event_data in epoch_dict.items():
        averaged_data[ch_name] = {}
        if event_descriptions is None:
            event_descriptions_to_use = list(event_data.keys())
        else:
            event_descriptions_to_use = event_descriptions

        for event_desc in event_descriptions_to_use:
            if event_desc in event_data:
                averaged_data[ch_name][event_desc] = np.mean(event_data[event_desc], axis=0)

    return averaged_data