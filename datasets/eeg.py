import h5py
import os
import shutil
import pathlib
import numpy as np
import xml.etree.ElementTree as et

def is_hdf5_file(filepath): #Check for snirf file
   return pathlib.Path(filepath).suffix == '.snirf'

def read_hdf5(filepath:str) -> h5py.File:
    hdf = h5py.File(filepath, mode="r")
    return hdf

def find_hdf5_in_folder(folder_path:str):
    """
    Locate all .hdf5 files in directory and subdirectories. 
    Args:
        folder_path (str) : Path to directory.
    Returns:
        paths (array:str) : Filepaths of located hdf5 files
        hdfs (array:h5py.File) : Loaded hdf5 file objects
    """
    paths = []
    for entry in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry)

        if os.path.isdir(entry_path):# is this a folder?
            p, f = find_hdf5_in_folder(entry_path)
            for path in p:
                paths.append(path)
            continue

        if is_hdf5_file(entry_path): 
            paths.append(entry_path)
            
    hdfs = []
    for path in paths:
        hdfs.append(read_hdf5(path))
        
    return paths, hdfs

def format_comment(comment: str) -> str:
    lines = comment.strip().split("\n")
    formatted = ", ".join(f"{lines[i]}:{lines[i+1]}" for i in range(0, len(lines), 2) if i+1 < len(lines))
    return formatted

def parse_xml(xml_str):
    root = et.fromstring(xml_str)  # Parse XML
    parsed_data = {}
    for child in root:
        parsed_data[child.tag] = child.text.strip() if child.text else None
    return parsed_data

def parse_hdf5(hdf:h5py.File):
    """
    Parses an HDF5 file created by g.Recorder, extracting EEG data and metadata.

    Args:
        hdf (h5py.File): An open HDF5 file object containing EEG recordings.

    Returns:
        tuple: A tuple containing:
            - samples (numpy.ndarray): The EEG time-series data.
            - sampling_frequency (float): The sampling frequency of the recording.
            - channel_num (int): The number of channels in the EEG data.
    """
    
    # 'AsynchronData'
    async_data_group = hdf["AsynchronData"] 
    async_data_keys = list(async_data_group.keys())
    async_signal_types = parse_xml(async_data_group["AsynchronSignalTypes"].asstr()[0])

    # 'SavedFeatures' # NOTE : TODO this needs to be implemetned further 
    saved_features_group = hdf["SavedFeatues"]
    saved_features_keys = saved_features_group.keys()
    features_num = saved_features_group["NumberOfFeatures"].astype('int32')[0]
    print("features_num : ", features_num)

    # 'Version'
    version_group = hdf["Version"]
    version_keys = version_group.keys()
    version = version_group["Version"].asstr()[0]
    print("version : ", version)

    # Handle 'RawData'
    rawdata_group = hdf["RawData"] # this is a group

    #   'Samples' NOTE : This is the actual channel data
    print(rawdata_group["Samples"])
    samples = np.array(rawdata_group["Samples"]).transpose()
    print("samples : ", samples.shape)

    #   'AcquisitionTaskDescription' 
    acquisition_task_desc = parse_xml(rawdata_group["AcquisitionTaskDescription"].asstr()[0])
    channel_properties = acquisition_task_desc.get("ChannelProperties")
    channel_num = int(acquisition_task_desc.get("NumberOfAcquiredChannels"))
    sampling_frequency = float(acquisition_task_desc.get("SamplingFrequency"))
    print("channel_properties : ", channel_properties)
    print("channel_num : ", channel_num)
    print("sampling_frequency : ", sampling_frequency)

    #   'DAQDeviceCapabilities' # NOTE : No important information here as far as I can tell
    daw_capabilities = parse_xml(rawdata_group["DAQDeviceCapabilities"].asstr()[0])

    #   'DAQDeviceDescription
    daq_desc = parse_xml(rawdata_group["DAQDeviceDescription"].asstr()[0])
    acquisition_unit = daq_desc.get("Unit") # micro Volts
    acquisition_device = daq_desc.get("Name")
    print("acquisition_unit : ", acquisition_unit)
    print("acquisition_device : ", acquisition_device)

    #   'SessionDescription'
    session_desc = parse_xml(rawdata_group["SessionDescription"].asstr()[0])
    session_run = session_desc.get("Run")
    session_comment = session_desc.get("Comment")
    print("session_run :", session_run)
    print("session_comment :", session_comment)

    #   'SubjectDescription'
    subject_desc = parse_xml(rawdata_group["SubjectDescription"].asstr()[0])
    comment = format_comment(subject_desc.get("Comment"))
    birthday = subject_desc.get("DayOfBirth")
    first_name = subject_desc.get("FirstName")
    last_name = subject_desc.get("LastName")
    print("comment : ", comment)
    print("birthday : ", birthday)
    print("first_name : ", first_name)
    print("last_name : ", last_name)
    
    #Features / Markers
    features_onset = [13.4, 25.4, 34.5, 45.3]
    features_order = [0, 1, 0, 2,]
    features_duration = [ 15, 10, 10]
    features_desc = [ "Rest", "Left", "Right"]
    return samples, sampling_frequency, channel_num, features_onset, features_order, features_desc

def write_hdf5_replace_data_keep_stats(data, original_hdf:h5py.File):
    
    # First Create a copy of the orignial file with a temporary name, then move that file into the new filepath
    old_filepath = original_hdf.filename #
    filename, suffix = old_filepath.split(".")
    temp_filepath = filename + "_copy." + suffix
    new_filepath = filename + "_PROCESSED." + suffix
    original_hdf.close()
    shutil.copy(old_filepath, temp_filepath)
    os.rename(temp_filepath, new_filepath)
    
    print(f"Write HDF5 : {old_filepath} copied and renamed to {new_filepath}")
    
    # Load the newly created file, and replace the data 
    hdf = h5py.File(new_filepath, mode="r+")
    print("Write HDF5 : Loaded {hdf} in read+ mode.")

    rawdata_group = hdf["RawData"]
    del rawdata_group["Samples"]
    rawdata_group.create_dataset("Samples", data=data.T, dtype="f4")
    
    hdf.close()
    return
    
    