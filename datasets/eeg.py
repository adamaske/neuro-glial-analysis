import h5py
import os
import shutil
import pathlib
import numpy as np
import xml.etree.ElementTree as et
from wrappers.eeg import EEG

def is_hdf5_file(filepath): #Check for snirf file
   return pathlib.Path(filepath).suffix == '.snirf'

def find_eeg_files_in_folder(folder_path:str, recursive:bool=True):
    """
    Locate all .hdf5 files in directory and subdirectories. 
    Args:
        folder_path (str) : Path to directory.
    Returns:
        paths (array:str) : Filepaths of located hdf5 files
        eegs (array:eeg.EEG) : Loaded EEG objects
    """
    paths = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)

        if os.path.isdir(entry_path) and recursive:# is this a folder?
            p, f = find_eeg_files_in_folder(entry_path, recursive)
            for path in p:
                paths.append(path)
            continue

        if is_hdf5_file(entry_path): 
            paths.append(entry_path)
            
    eegs = []
    for path in paths:
        eegs.append(EEG(path))
        
    return paths, eegs