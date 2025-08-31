import h5py
import os
import shutil
import pathlib
import numpy as np
import xml.etree.ElementTree as et
import datetime


def read_snirf(filepath):
        print("Reading SNIRF (HDF5) file : ", filepath)
        
        filepath = filepath
        hdf = h5py.File(filepath, mode="r+")
        print(hdf.keys())

        # formatVersion
        format_version = hdf.get("formatVersion")
        print("formatVersion:", format_version[()])

        nirs = hdf.get("nirs")
        print("nirs:", nirs.keys())
        
        # data1
        data = nirs.get("data1")
        print("data1:", data)
        # metaDataTags
        metaData_tags = nirs.get("metaDataTags")
        print("metaDataTags:", metaData_tags)
        #probe
        probe = nirs.get("probe")
        print("probe:", probe.keys())
        wavelengths = probe.get("wavelengths")
        print("wavelengths:", wavelengths[()])
        #stim1
        stim1 = nirs.get("stim1")
        print("stim1:", stim1.keys())
        #stim2
        stim2 = nirs.get("stim2")
        print("stim2:", stim2.keys())
        # Read all the data

read_snirf("data/ProcessedData/Filtered_/Active_C3_Trial_1_FILTERED_2.snirf")