from mne.preprocessing.nirs import temporal_derivative_distribution_repair



def motion_correction(snirf):
    """
    This removes baseline shift and spike artifacts.\n
    It is recommended to apply this to optical density data, however hemoglobin will also work. 
   
    Args:
        snirf (): mne.raw snrif object with either optical density of hemoglobin_concentration
    Returns:
        raw (): mne.raw snirf object with baseline
    
    """
    corrected = temporal_derivative_distribution_repair(snirf)
    return corrected