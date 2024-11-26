from mne.io.snirf._snirf import RawSNIRF
from mne.filter import filter_data
import numpy as np

def z_normalize():
    import numpy as np

    # Example time series
    time_series = np.array([10, 20, 15, 25, 30])

    # Calculate mean and standard deviation
    mean = np.mean(time_series)
    std = np.std(time_series)

    # Z-normalize the time series
    z_normalized_series = (time_series - mean) / std

    print("Original time series:", time_series)
    print("Z-normalized time series:", z_normalized_series)
    pass

def filter_snirf(snirf:RawSNIRF) -> RawSNIRF:
    """
    Frequency filtering of fNIRS data. Filter design based on Pinti et al. 2019. 
    Bandpass : F_low = 0.01,  F_stim < F_high < F_mayer (> 0.06).
    PS: This means stimulation frequency must be minimum 16.5 seconds.  
    This method removes heartbeat, breathing, mayer waves and instrumental noise (and low frequency vasomotor oscilations).
    Args:
        snirf (mne.raw or str) : Either mne.raw snirf object or filepath to ".snirf" file. 
        artefacts (list) : What artefacts to filter out "heartbeat", "mayer", "breathing"
    Returns:
        filtered (mne.raw or str) : Either mne.raw snirf object or filepath to filtered ".snirf" file. 
    """
    #sampling_frequency = snirf.info["sfreq"]
    ##Covariance 10%
    #
    ##Bandpass FIR 0.01 < F_stim < 0.07
    #
    ##Mayer wave removal
    #mw_l = 0.7
    #mw_h = 1.4
#
    #data = np.array(snirf.get_data())
    #filtered_data = filter_data(data=data, 
    #                            sfreq=sampling_frequency,
    #                            l_freq=0.01,
    #                            h_freq=0.07,
    #                            picks=None,
    #                            filter_length=)
    
    filtered = snirf.copy().filter(l_freq=0.01,
                            h_freq=0.07, 
                            picks='all',
                            method='fir',
                            filter_length=200,
                            )


    return filtered