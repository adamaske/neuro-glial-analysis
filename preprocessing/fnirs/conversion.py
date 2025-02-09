import pandas as pd
import numpy as np
from mne.io import read_raw_snirf
from mne_nirs.io import write_raw_snirf
from mne.preprocessing.nirs import optical_density, beer_lambert_law
from os.path import split, splitext, join
from preprocessing.fnirs.validation import validate_snirf

from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from mne.io.snirf._snirf import RawSNIRF

import matplotlib.pyplot as plt
def csv_to_snirf(csv_filepath, channel_names, channel_types, sampling_frequency):
    """
    Convert CSV file to .snirf file.
    Args:
        csv_filepath (str) : filepath to a CSV 1 channel per row, 1 colum per time point
        channel_names (array) : 
        channel_types (str) : "HbO", "HbR", "HbT" etc..
        sampling_frequency (float) : Acqusition hardware sampling frequency
        
    Returns:
        snirf (RawSNIRF) :   
    
    """
    pd.DataFrame(np.random.normal(size=(16, 100))).to_csv("temp/fnirs.csv")
    
    data = pd.read_csv("temp/fnirs.csv")    
    ch_names = [ "S1_D1 hbo", "S1_D1 hbr", "S2_D1 hbo", "S2_D1 hbr", "S3_D1 hbo", "S3_D1 hbr", "S4_D1 hbo", "S4_D1 hbr", "S5_D2 hbo", "S5_D2 hbr", "S6_D2 hbo", "S6_D2 hbr", "S7_D2 hbo", "S7_D2 hbr", "S8_D2 hbo", "S8_D2 hbr"]
    ch_types = [ "hbo", "hbr", "hbo", "hbr", "hbo", "hbr", "hbo", "hbr", "hbo", "hbr", "hbo", "hbr", "hbo", "hbr", "hbo", "hbr"]
    
    fs = 10.0
    
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=fs)
    raw = RawArray(data, info, verbose=True)
    
    #Montage
    montage = make_standard_montage("artinis-octamon")
    raw.set_montage(montage)
    
    raw.plot()
    plt.show()
    return

    

def light_intensity_to_optical_density(snirf) -> "RawSNIRF":
    """
    Convert raw light intensity data to optical density. Handles both mne raw snirf objects and filepath to snirf file.

    Args:
        snirf (mne.raw or str) : Either mne.raw snirf object or filepath to snirf file. 

    Returns:
        od (mne.raw or str) : Either mne.raw snirf object or filepath to new snirf file.
    """
    if isinstance(snirf, str): #handle as a filepath
        directory, filename = split(snirf)
        name, extension = splitext(filename)

        new_filename = f"{name}_optical_density{extension}"
        new_path = join(directory, new_filename)

        light_intensity = read_raw_snirf(snirf)
        #is_valid = validate_snirf(light_intensity)
        #if not is_valid:
        #    print("Invalid snirf object, returning original filepath : light_intensity_to_optical_density, ", __file__)
        #    return snirf
        
        od = optical_density(light_intensity)
        #od_valid = validate_snirf(od_valid)
        #if not od_valid:
        #    print("An error occured in optical density conversion, returning original filepath : light_intensity_to_optical_density, ", __file__)
        #    return snirf
        
        write_raw_snirf(od, new_path)
        print(f"Optical density conversion succeeded, new filepath : {new_path}")
        return new_path
        
    #is_valid = validate_snirf(snirf)
    #if not is_valid:
    #    print("Invalid snirf object, returning original snirf object : light_intensity_to_optical_density, ", __file__)
    #    return snirf

    od = optical_density(snirf)
    #od_valid = validate_snirf(od)
    #if not od_valid:
    #        print("An error occured in optical density conversion, returning original snrif object : light_intensity_to_optical_density, ", __file__)
    #        return snirf
    
    print(f"Optical density conversion succeeded, new snirf object : ")
    print(od.info)
    return od


def optical_density_to_hemoglobin_concentration(od) -> "RawSNIRF":
    """
    Convert optical density data to hemoglobin concentration. Handles both mne raw snirf objects and filepath to snirf file.

    Args:
        od (mne.raw or str) : Either mne.raw snirf object or filepath to snirf file. 

    Returns:
        hb (mne.raw or str) : Either mne.raw snirf object or filepath to new snirf file.
    """
    if isinstance(od, str): #handle as a filepath
        directory, filename = split(od)
        name, extension = splitext(filename)

        new_filename = f"{name}_hemoglobin{extension}"
        new_path = join(directory, new_filename)

        od = read_raw_snirf(od)
        #is_valid = validate_snirf(od)
        #if not is_valid:
        #    print("Invalid optical density snirf object, returning original filepath : optical_density_to_hemoglobin_concentration, ", __file__)
        #    return od
        
        hb = beer_lambert_law(od)
        #hb_is_valid = validate_snirf(hb)
        #if not hb_is_valid:
        #    print("An error occured in hemoglobin conversion, returning original filepath : optical_density_to_hemoglobin_concentration, ", __file__)
        #    return od
        
        write_raw_snirf(hb, new_path)
        print(f"Hemoglobin conversion succeeded, new filepath : {new_path}")
        return new_path
        
    #is_valid = validate_snirf(od)
    #if not is_valid:
    #    print("Invalid snirf object, returning original snirf object : optical_density_to_hemoglobin_concentration, ", __file__)
    #    return od

    hb = beer_lambert_law(od)
    #hb_is_valid = validate_snirf(hb)
    #if not hb_is_valid:
    #        print("An error occured in hemoglobin conversion, returning original snrif object : optical_density_to_hemoglobin_concentration, ", __file__)
    #        return od
    
    print(f"Optical density conversion succeeded, new snirf object : ")
    print(hb.info)
    return hb

def light_intensity_to_hemoglobin_concentration(snirf):
     od = light_intensity_to_optical_density(snirf)
     hb = optical_density_to_hemoglobin_concentration(od)
     return hb