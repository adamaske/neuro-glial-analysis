#find all snirf files in a given directory and subdirectories

from datasets.fnirs_data import find_snirf_in_folder

from mne.io import read_raw_snirf

from preprocessing.fnirs.conversion import light_intensity_to_hemoglobin_concentration

from os.path import split, splitext, join
import numpy as np

folder_path = "data/OMID-13-12-024"
snirf_filepaths = find_snirf_in_folder(folder_path)

for path in snirf_filepaths:
    dir, name = split(path)
    snirf = read_raw_snirf(path).load_data()
    print(f"Loaded SNIRF : {name}")
    print(snirf.info)
        
    #hb = light_intensity_to_hemoglobin_concentration(hb)
    
    data = np.array(snirf.get_data())
    
    print("Data : ", data.shape)
    

def preprocess_snirf_files():
    
    
    
    pass
    
    

