import numpy as np
import matplotlib.pyplot as plt
from mne.io.snirf._snirf import RawSNIRF
from analysis.frequencies import compute_psd

def plot_snirf(snirf:RawSNIRF, show=True):
    """
    Display fNIRS channels
    Args:
        snirf (RawSNIRF) : snirf object.
        channels (int or list) : Channel index or list of indices
    """
    fig, axs = plt.subplots(1, 2)
    channels = np.array(snirf.get_data())
    
    axs[0].set_title("HbO")
    axs[1].set_title("HbR")
    axs[0].set_ylabel("Concentration")
    axs[1].set_ylabel("Concentration")
    axs[0].set_xlabel("Time")
    axs[1].set_xlabel("Time")
    hbo = channels[0:int(len(channels)/2),:]
    hbr = channels[int(len(channels)/2):len(channels),:]
    for ch in hbo:
        axs[0].plot(ch)
    for ch in hbr:
        axs[1].plot(ch)
    
    if show:
        plt.show()
    return


def plot_psd_snirf(snirf:RawSNIRF, show=True):
    """
    Display Power Spectral Density of all channels
    Args:
        snirf (RawSNIRF) : snirf object.
    """
    fig, axs = plt.subplots(1, 2)
    channels = np.array(snirf.get_data())
    fs = snirf.info["sfreq"]
    
    axs[0].set_title("HbO")
    axs[1].set_title("HbR")
    axs[0].set_ylabel("Magnitude")
    axs[1].set_ylabel("Magnitude")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[1].set_xlabel("Frequency (Hz)")

    hbo = channels[0:int(len(channels)/2),:]
    hbr = channels[int(len(channels)/2):len(channels),:]
    for ch in hbo:
        freqs, spectra = compute_psd(ch, fs, int(fs/2))
        axs[0].plot(freqs, spectra )
    for ch in hbr:
        freqs, spectra = compute_psd(ch, fs, int(fs/2))
        axs[1].plot(freqs, spectra )

    if show:
        plt.show()
    return
    
    