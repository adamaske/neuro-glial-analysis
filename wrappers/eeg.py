import h5py
import os
import shutil
import pathlib
import numpy as np
import xml.etree.ElementTree as et

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

class EEG:
    def __init__(self, filepath:str=""):
        if filepath != "":
            self.read(filepath)
        
        pass
    
    def print_features(self):
        assert(len(self.onsets) == len(self.descs)) # 
        for idx, onset in enumerate(self.onsets):
            print(f"Feature {idx} @ {self.onsets[idx]}:{self.descs[idx]}")
        
    def print(self):
        print("EEG : ", self.filepath)
        print("sampling_frequency : ", self.sampling_frequency)  
        print("channel_num : ", self.channel_num )
        print("channel_data : ", self.channel_data.shape)
        print("feature_onsets : ", self.feature_onsets)
        print("feature_descriptions : ", self.feature_descriptions) 
        print("session_run :", self.session_run)
        print("session_comment :", self.session_comment)
        print("acquisition_device : ", self.acquisition_device)
        print("acquisition_unit : ", self.acquisition_unit)
        
    def read(self, filepath):
        print("Reading HDF5 file : ", filepath)
        
        self.filepath = filepath
        self.hdf = h5py.File(filepath, mode="r+")
        
        # Handle 'RawData' containing the acutal channeld ata
        rawdata_group = self.hdf["RawData"]
        self.channel_data = np.array(rawdata_group["Samples"]).transpose()
        print("Samples : ", self.channel_data.shape)
        
        # 'AsynchronData' : This contains Features / Markers
        async_data_group = self.hdf ["AsynchronData"] 
        self.feature_onsets = np.array(async_data_group["Time"]).T[0]
        self.feature_descriptions = np.array(async_data_group["TypeID"]).T[0]
        print("feature_onsets : ", self.feature_onsets)
        print("feature_descriptions : ", self.feature_descriptions) 
        
        # 'AcquisitionTaskDescription' 
        acquisition_task_desc = parse_xml(rawdata_group["AcquisitionTaskDescription"].asstr()[0])
        channel_properties = acquisition_task_desc.get("ChannelProperties")
        self.channel_num = int(acquisition_task_desc.get("NumberOfAcquiredChannels"))
        self.sampling_frequency = float(acquisition_task_desc.get("SamplingFrequency"))
        print("channel_properties : ", channel_properties)
        print("channel_num : ", self.channel_num )
        print("sampling_frequency : ", self.sampling_frequency)  
        
        #   'DAQDeviceDescription
        daq_desc = parse_xml(rawdata_group["DAQDeviceDescription"].asstr()[0])
        self.acquisition_unit = daq_desc.get("Unit") # micro Volts
        self.acquisition_device = daq_desc.get("Name")
        print("acquisition_unit : ", self.acquisition_unit)
        print("acquisition_device : ", self.acquisition_device)
        
        #   'SessionDescription'
        session_desc = parse_xml(rawdata_group["SessionDescription"].asstr()[0])
        print(session_desc)
        self.session_run = session_desc.get("Run")
        self.session_comment = session_desc.get("Comment")
        print("session_run :", self.session_run)
        print("session_comment :", self.session_comment)
        
        #   'SubjectDescription'
        subject_desc = parse_xml(rawdata_group["SubjectDescription"].asstr()[0])
        comment = format_comment(subject_desc.get("Comment"))
        first_name = subject_desc.get("FirstName")
        last_name = subject_desc.get("LastName")
        print("comment : ", comment)
        print("name : ", first_name, " ", last_name)
        
        self.hdf.close()
        # 'SavedFeatures' # NOTE : This is not important
        #saved_features_group = self.hdf["SavedFeatues"]
        #saved_features_keys = saved_features_group.keys()
        #features_num = saved_features_group["NumberOfFeatures"].astype('int32')[0]
        #print("features_num : ", features_num)

        # 'Version' # NOTE : This is not important
        #version_group = self.hdf["Version"]
        #version_keys = version_group.keys()
        #version = version_group["Version"].asstr()[0]
        #print("version : ", version)

        #   'DAQDeviceCapabilities' # NOTE : No important information here as far as I can tell
        #daw_capabilities = parse_xml(rawdata_group["DAQDeviceCapabilities"].asstr()[0])
        
        return self
    
    def write(self, new_filepath:str) -> "EEG":
        
        old_filepath = self.filepath #  What file to copy
        
        filename, suffix = old_filepath.split(".") 
        temp_filepath = filename + "_temporary." + suffix # Where to create a copy
        
        shutil.copy(old_filepath, temp_filepath) # Copy file
        print(f"Write HDF5 : {old_filepath} copied, new filepath : {new_filepath}")

        # Loads the temporary file, alters it
        hdf = h5py.File(temp_filepath, mode="r+")
        print("Write HDF5 : Loaded {hdf} in read+ mode.")

        #Replace features
        async_group = hdf["AsynchronData"]
        del async_group["Time"]
        del async_group["TypeID"] 
        async_group.create_dataset("Time", data=np.array(self.feature_onsets).reshape(-1, 1))
        async_group.create_dataset("TypeID", data=np.array(self.feature_descriptions).reshape(-1, 1))
        
        # Overwrite channel data
        rawdata_group = hdf["RawData"]
        del rawdata_group["Samples"]
        rawdata_group.create_dataset("Samples", data=self.channel_data.T, dtype="f4")
        
        hdf.close()
       
        # Then relocates and renames the file
        if os.path.exists(new_filepath):
            ans = input(f"{new_filepath} already exists. Do you want to overwrite it? [Y / N] : ")
            if ans.capitalize() == "Y":
                shutil.move(temp_filepath, new_filepath)
                print(f"Wrote HDF5 to {new_filepath}")
            else:
                print("Write canceled....")
                os.remove(temp_filepath)
                return
        else:
            os.rename(temp_filepath, new_filepath) # Change file name & path
            print(f"Wrote HDF5 to {new_filepath}")

        return EEG(new_filepath)
    
    