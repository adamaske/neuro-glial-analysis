import h5py
import os
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

# This is a wrapper for hdf5 eeg file from g.Recorder
class EEG_: 
    def __init__(self):
        pass
    
    def parse_hdf(self, hdf):
        print(hdf.keys())
        # 'AsynchronData' : This contains markers
        async_data_group = hdf["AsynchronData"] 
        print(async_data_group["AsynchronSignalTypes"])
        root = et.fromstring(async_data_group["AsynchronSignalTypes"].asstr()[0])

        signals = []
        for signal in root.findall('AsynchronSignalDescription'):
            signal_data = {
                'IsCombinedSignal': signal.find('IsCombinedSignal').text,
                'Name': signal.find('Name').text,
                'IsToggleSignal': signal.find('IsToggleSignal').text,
                'IsTrigger': signal.find('IsTrigger').text,
                'IsEnabled': signal.find('IsEnabled').text,
                'ID': signal.find('ID').text,
                'Edge': signal.find('Edge').text,
                'Description': signal.find('Description').text,
                'Direction': signal.find('Direction').text,
                'Color': signal.find('Color').text,
                'ChannelNumber': signal.find('ChannelNumber').text,
                'DeviceSerial': signal.find('SourceDevice/DeviceSerial').text,
                'SourceType': signal.find('SourceDevice/SourceType').text,
            }
            print("\n")
            print(signal_data)

        onsets = np.array(async_data_group["Time"]).T
        descs = np.array(async_data_group["TypeID"]).T
        values = np.array(async_data_group["Value"]).T

        
        print("onsets : ", onsets)
        print("descs : ", descs)
        print("values : ", values)
        return

        # 'SavedFeatures' # NOTE : TODO this needs to be implemetned further 
        saved_features_group = hdf["SavedFeatues"]
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

    def open(self, filepath:str):

        hdf = h5py.File(filepath, mode="r")

        self.parse_hdf(hdf)

        return 
    
    def write(self, filepath:str):

        #prompt do you want to overwrite it?


        return

    def close(self, filepath:str):


        return
    
if __name__ == "__main__":

    eeg = EEG_()
    eeg.open("data/FeaturesTesting.hdf5")