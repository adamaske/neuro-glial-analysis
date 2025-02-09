import pathlib
import os 
import snirf

from mne.io import read_raw_eeglab

from preprocessing.fnirs.validation import validate_snirf
from mne.io.snirf._snirf import RawSNIRF

dir_path = os.path.dirname(os.path.realpath(__file__)) #path of this file

def is_snirf_file(filepath): #Check for snirf file
   return pathlib.Path(filepath).suffix == '.snirf'


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
    
    valid = validate_snirf(snirf)
    if not valid:
        print("read_snirf : Invalid snirf object @ ", __file__)
        return None
    
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
            snirfs = find_snirf_in_folder(entry_path)
            for file in snirfs:
                paths.append(file)
            continue

        if is_snirf_file(entry_path): #is this a snirf file?
            paths.append(entry_path)
            
    snirfs = []
    for path in paths:
        snirfs.append(read_snirf(path))
        
    return paths, snirfs
