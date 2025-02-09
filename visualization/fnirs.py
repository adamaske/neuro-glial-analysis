import numpy as np
import matplotlib.pyplot as plt
from mne.io.snirf._snirf import RawSNIRF
from analysis.fnirs.frequencies import compute_psd
def visualize_snirf(snirf):

    pass

def plot_snirf(snirf:RawSNIRF):
    """
    Display fNIRS channels
    Args:
        snirf (RawSNIRF) : snirf object.
        channels (int or list) : Channel index or list of indices
    """
    fig, axs = plt.subplots(1, 2)
    channels = np.array(snirf.get_data())
    
    
    hbo = channels[0:int(len(channels)/2),:]
    hbr = channels[int(len(channels)/2):len(channels),:]
    for ch in hbo:
        axs[0].plot(ch)
    for ch in hbr:
        axs[1].plot(ch)
        
    plt.show()
    return


def plot_psd_snirf(snirf:RawSNIRF):
    """
    Display Power Spectral Density of all channels
    Args:
        snirf (RawSNIRF) : snirf object.
    """
    fig, axs = plt.subplots(1, 2)
    channels = np.array(snirf.get_data())
    fs = snirf.info["sfreq"]
    
    hbo = channels[0:int(len(channels)/2),:]
    hbr = channels[int(len(channels)/2):len(channels),:]
    for ch in hbo:
        freqs, spectra = compute_psd(ch, fs, int(fs/2))
        axs[0].plot(freqs, spectra )
    for ch in hbr:
        freqs, spectra = compute_psd(ch, fs, int(fs/2))
        axs[1].plot(freqs, spectra )
        
    plt.show()
    return
    
    