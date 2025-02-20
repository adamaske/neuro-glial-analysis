import pathlib
import os 
from mne.io import read_raw_snirf
from mne.io.snirf._snirf import RawSNIRF
from mne_nirs.io import write_raw_snirf
from snirf import validateSnirf
import logging

dir_path = os.path.dirname(os.path.realpath(__file__)) #path of this file

def is_snirf_file(filepath): #Check for snirf file
   return pathlib.Path(filepath).suffix == '.snirf'

def validate_snirf(snirf:RawSNIRF):
    result = validateSnirf(snirf)
    result.display()
    return result.is_valid()

def validate_snirf_files(files):
    
    
    return 0


def read_snirf(filepath:str) -> "RawSNIRF":
    """
    Load a .snirf file into a RawSNIRF object and validates it. 
    
    Args:
        filepath : str
        
    Returns:
        snirf : RawSNIRF object
    
    """
    snirf = read_raw_snirf(filepath)
    
    #valid = validate_snirf(snirf)
    #if not valid:
    #    print("read_snirf : Invalid snirf object @ ", __file__)
    #    return None
    
    return snirf


def find_snirf_in_folder(folder_path):
    """
    Locate all .snirf files in directory and subdirectories. 
    Args:
        folder_path (str) : Path to directory.
    Returns:
        paths (array:str) : All filepaths
        
        snirfs (array:RawSNIRF) : Loaded snirf objects
    """
    paths = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)

        if os.path.isdir(entry_path):# is this a folder?
            p, f = find_snirf_in_folder(entry_path)
            for path in p:
                paths.append(path)
            continue

        if is_snirf_file(entry_path): #is this a snirf file?
            paths.append(entry_path)
            
    snirfs = []
    for path in paths:
        snirfs.append(read_snirf(path))
        
    return paths, snirfs

def write_snirf(snirf:RawSNIRF, filepath:str) -> str:
    write_raw_snirf(snirf, filepath)
    logging.info(f"Wrote .sNIRF : {filepath}")
    
def write_snirfs(paths, snirfs, append_to_path):
    assert(len(paths) == len(snirfs))

    logging.info(f"Writing {len(snirfs)} .sNIRF files")
    for path, snirf in zip(paths, snirfs):

        dir, filename = os.path.split(path)
        name, ext = os.path.splitext(filename)

        # Construct new filename
        new_filename = f"{name}_{append_to_path}.snirf"
        new_path = os.path.join(dir, new_filename)

        write_snirf(snirf, new_path)
        
def snirf_channel_names_to_indices(snirf:RawSNIRF):
    indicies = []
    
    return indicies

def get_roi_by_indicies(snirf:RawSNIRF, indicies):
    """
    Extract the ROI channels from snirf file. 
    Returns the channels, and channel names where channels[i] = channel_names[i]
    Args:
        snirf(RawSNIRF): snirf object
        indices(array:int): list of channel indices in roi
    Returns:
        channels(np.array) : each channel
        channel_names(array:str) : List of channel names where channels[i] = channel_names[i]
    """
    data = snirf.get_data()
    ch_names = snirf.info["ch_names"]
    
    roi_channels = []
    roi_channel_names = []
    for i in indicies:
        roi_channels.append(data[i])
        roi_channel_names.append(ch_names[i])
        
    return roi_channels, roi_channel_names

    
    