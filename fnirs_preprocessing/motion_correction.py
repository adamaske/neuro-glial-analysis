from mne.preprocessing.nirs import temporal_derivative_distribution_repair
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf

def motion_correction(snirf):
    """
    This removes baseline shift and spike artifacts.\n
    It is recommended to apply this to optical density data, however hemoglobin will also work. 
   
    Args:
        snirf (): mne.raw snrif object or filepath to ".snirf" with either optical density of hemoglobin_concentration
    Returns:
        raw (): mne.raw snirf object with baseline
    
    """
    if isinstance(snirf, str): 
        print("CAREFUL MOTION CORRECTION WITH FILEPATH OVERWRITES THE PREVIOUS FILE")
        #load the file
        data = read_raw_snirf(snirf).load_data()
        #motion correct
        corrected_data = temporal_derivative_distribution_repair(data)

        write_raw_snirf(data, snirf)
        #write new file
        


    corrected = temporal_derivative_distribution_repair(snirf.copy())
    return corrected