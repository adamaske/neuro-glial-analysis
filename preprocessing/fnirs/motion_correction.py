from mne.preprocessing.nirs import temporal_derivative_distribution_repair
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
from mne.io.snirf._snirf import RawSNIRF
import numpy as np
def motion_correction(snirf:RawSNIRF) -> "RawSNIRF":
    """
    This removes baseline shift and spike artifacts.\n
    It is recommended to apply this to optical density data, however hemoglobin will also work. 
   
    Args:
        snirf (): mne.raw snrif object or filepath to ".snirf" with either optical density of hemoglobin_concentration
    Returns:
        raw (): mne.raw snirf object with baseline
    
    """

    corrected = temporal_derivative_distribution_repair(snirf.copy())
    return corrected

def channel_rejection(snirf:RawSNIRF, threshold=0.1) -> "RawSNIRF":
    """Remove channels with coefficient of variation (CoV) above threshold."""
    data = snirf.get_data()  # Extract fNIRS data
    mean_vals = np.mean(data, axis=1)  # Compute mean per channel
    std_vals = np.std(data, axis=1)  # Compute std per channel

    cov = std_vals / mean_vals  # Compute CoV
    valid_channels = cov <= threshold  # Keep channels with CoV <= 10%
    ch_names = snirf.info['channels'] 
    snirf.drop_channels()
    # Apply mask to remove bad channels
    snirf._data = data[valid_channels]
    snirf.info['channels'] = [ch for i, ch in enumerate(snirf.info['channels']) if valid_channels[i]]

    return snirf