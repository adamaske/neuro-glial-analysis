

def crop_by_first_and_last_annotation(snirf):
    
    """
    Crop a snirf file to start when the first annoation begins and ends at the end of the last annotation. 
    Returns a cropped copy of the SnirfRAW object. 
    Args:
        snirf (mne.SnirfRAW) : 
    Returns:
        cropped (mne.SnirfRAW) : Cropped snrif object. 
    """
    
    start = snirf.annotations[0]["onset"]
    last = snirf.annotations[len(snirf.annotations)-1]
    end = last["onset"] + last["duration"]
    cropped = snirf.copy()
    cropped = cropped.crop(tmin=start, tmax=end)
    return cropped
