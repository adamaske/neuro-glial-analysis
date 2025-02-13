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
    
    annotations = snirf._annotations
    order = annotations.description
    onsets = annotations.onset
    
    epochs = []
    for i, onset in enumerate(onsets):
        desc = order[i]
        
        start = int((onset + tmin) * s_freq)
        end = int((onset + tmax) * s_freq)
        
        if end >= len(data[0]):
            continue

        epoch = data[:, start:end]        
        epoch_avg = np.mean(epoch, axis=0)
        
        epochs.append(epoch, epoch_avg)
        
    return epochs, order


def block_averaging_snirf(snirf:RawSNIRF, tmin=2, tmax=15):
    epochs, order = epochs_snirf(snirf, tmin, tmax)
    
    avg_epochs = []
    for i, epoch in enumerate(epochs):
        avg_epoch = np.mean(epoch, axis=0)
        avg_epochs.append(avg_epoch)
    
    return avg_epochs, order
