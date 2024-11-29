import numpy as np
from scipy.signal import butter, lfilter
from mne.io.snirf._snirf import RawSNIRF

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

def butter_bandpass(lowcut, highcut, fs, order):
    return butter(order, [lowcut, highcut], fs=fs, btype='bandpass')

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def digital_bandpass_filter(snirf:str|RawSNIRF, lowcut=0.01, highcut=0.1, order=5) -> RawSNIRF|str:
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
    
    if not isinstance(snirf, str) or not isinstance(snirf, RawSNIRF):
        raise TypeError("Object must be of type str or RawSNIRF...")
     
    if isinstance(snirf, str):#load filepath
        pass
        
    sampling_frequency = snirf.info["sfreq"]
    
    #Z-Normalization
    
    #Digital Filter
    b, a = butter_bandpass(lowcut, highcut, sampling_frequency, order)
    
    #data = np.array(snirf.get_data())
    ##filter = filter_data(data=data, 
    ##                     sfreq=sampling_frequency,
    ##                     l_freq=0.01,
    ##                     h_freq=0.07,
    ##                     picks=None,
    ##                     filter_length=101)
    
    filtered = snirf.copy().filter(l_freq=0.01,
                            h_freq=0.07, 
                            picks='all',
                            method='fir',
                            filter_length=200,
                            )


    return filtered