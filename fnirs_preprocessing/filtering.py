from mne.io.snirf._snirf import RawSNIRF

def filter(snirf:RawSNIRF) -> RawSNIRF:
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

    filtered = snirf.copy().filter(l_freq=0.01,
                            h_freq=0.09, 
                            picks='all',
                            method='fir',

                            )


    return filtered