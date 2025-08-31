
import h5py
import os
import shutil
import pathlib
import numpy as np
import xml.etree.ElementTree as et
import datetime
import matplotlib.pyplot as plt

import scipy.stats as stats

from neuropipeline.fnirs import fNIRS

fnirs = fNIRS("data/beetroot/Trials/Post_C4_Trial_1.snirf")


def handle_metadata_tags(metadata_tags):
    """
    Handles the metadata tags from a SNIRF file.

    Args:
        metadata_tags (h5py.Group): The metadata tags group from the SNIRF file.
    """
    aurora_version = metadata_tags.get("AuroraVersion", None)[0].decode('utf-8') 
    channel_mask = metadata_tags.get("ChannelMask", None)[0].decode('utf-8') 
    ERABaselineCorrection = int(metadata_tags.get("ERABaselineCorrection", None)[0])
    ERABaselineCorrectionPost =int( metadata_tags.get("ERABaselineCorrectionPost", None))
    ERABaselineCorrectionPre = int(metadata_tags.get("ERABaselineCorrectionPre", None))
    #print(aurora_version)
    #print(channel_mask)
    #print(ERABaselineCorrection)
    #print(ERABaselineCorrectionPost)
    #print(ERABaselineCorrectionPre) 
    exit()

def get_stimulus_data(stim_group):

    name = stim_group["name"][0].decode('utf-8')
    stim_data = np.array(stim_group["data"])

    onsets = stim_data[:, 0]
    durations = stim_data[:, 1]

    #print(f"stim name: {name}")
    #print(f"onsets: {onsets}")
    #print(f"durations: {durations}")

    return name, onsets, durations # in seconds

def read_snirf(filepath):
    snirf = h5py.File(filepath, 'r')

    nirs = snirf["nirs"]

    data1 = nirs["data1"]
    metaDataTags = nirs["metaDataTags"]
    #handle_metadata_tags(metaDataTags)
    probe = nirs["probe"]
    stim1 = nirs["stim1"]
    stim2 = nirs["stim2"]


    #print("nirs:", nirs.keys())
    #print("data1:", data1.keys())
    #print("metaDataTags:", metaDataTags.keys())
#
    #print("probe:", probe.keys())
    #print("stim1:", stim1.keys())
    #print("stim2:", stim2.keys())


    time_points = np.array(data1["time"])
    #print("time points shape:", time_points.shape)

    dataTimeSeries = np.array(data1["dataTimeSeries"]).T
    #print("dataTimeSeries shape:", dataTimeSeries.shape)

    data1_list = list(data1.keys())
    #print("data1 list:", data1_list)

    # Handle measurementList 
    #skip the first and last of :
    hbo_indices = []
    hbr_indices = []

    for i, mlist in enumerate(data1_list[1:-1], start=0):
        if mlist.startswith("measurementList"):
            
            measurement_list = data1[mlist]
            # HbO or HbR ? 
            dataTypeLabelDataset = measurement_list["dataTypeLabel"]
            label = dataTypeLabelDataset[:][0].decode('utf-8')

            sourceIndexDataset = measurement_list["sourceIndex"]
            source_index = sourceIndexDataset[:][0]

            detectorIndexDataset = measurement_list["detectorIndex"]
            detector_index = detectorIndexDataset[:][0]
            print(f"Measurement List {i+1} : S{source_index}-D{detector_index} : {label}")

            if label == "HbO":
                hbo_indices.append(i)
            if label == "HbR":
                hbr_indices.append(i)

    # Get all HbO data
    hbo_data = dataTimeSeries[0:38, :]
    #print("HbO data shape:", hbo_data.shape)
    # Get all HbR data
    hbr_data = dataTimeSeries[38:76, :]
    #print("HbR data shape:", hbr_data.shape)
    
    # Handle probes 
    stim1 = get_stimulus_data(stim1)
    stim2 = get_stimulus_data(stim2)

    
    if False:
        return
        print("data1:", data1.keys())
        dataTimeSeries = np.array(data1["dataTimeSeries"]).T
        print("dataTimeSeries:", dataTimeSeries.shape)
        time = data1["time"]
        print("time:", time[:10])

        for mlist in data1.keys():
            print("list:", mlist)   


            label = mlist["dataTypeLabel"][:][0].decode('utf-8')
            source_index = mlist["sourceIndex"][:][0]
            detector_index = mlist["detectorIndex"][:][0]

            print(f" S{source_index}-D{detector_index} : {label}")

        return
        list1 = data1["measurementList1"]
        print("measurementList1:", list1.keys())
        dataTypeDataset = list1["dataType"]
        print("dataType:", dataTypeDataset[:])

        # HbO or HbR ? 
        dataTypeLabelDataset = list1["dataTypeLabel"]
        label = dataTypeLabelDataset[:][0].decode('utf-8')

        sourceIndexDataset = list1["sourceIndex"]
        source_index = sourceIndexDataset[:][0]

        detectorIndexDataset = list1["detectorIndex"]
        detector_index = detectorIndexDataset[:][0]

        print(f" S{source_index}-D{detector_index} : {label}")
        #
        dataUnitDataset = list1["dataUnit"]
        #print("dataUnit:", dataUnitDataset[:])
        detectorIndexDataset = list1["detectorIndex"]
        #print("detectorIndex:", detectorIndexDataset[:])
        sourceIndexDataset = list1["sourceIndex"]
        #print("sourceIndex:", sourceIndexDataset[:])
        wavelengthIndexDataset = list1["wavelengthIndex"]
        #print("wavelengthIndex:", wavelengthIndexDataset[:])

    return hbo_data, hbr_data, time_points, stim1, stim2

paths = [
"data/beetroot/ProcessedData/Pre_C4_Trial_1_Filtered.snirf",
"data/beetroot/ProcessedData/Pre_C4_Trial_2_Filtered.snirf",
"data/beetroot/ProcessedData/Pre_C5_Trial_3_Filtered.snirf",
"data/beetroot/ProcessedData/Pre_C3_Trial_1_Filtered.snirf",
"data/beetroot/ProcessedData/Pre_C3_Trial_2_Filtered.snirf",
"data/beetroot/ProcessedData/Pre_C3_Trial_3_Filtered.snirf",
"data/beetroot/ProcessedData/Post_C4_Trial_1_Filtered.snirf",
"data/beetroot/ProcessedData/Post_C4_Trial_2_Filtered.snirf",
"data/beetroot/ProcessedData/Post_C4_Trial_3_Filtered.snirf",
"data/beetroot/ProcessedData/Post_C3_Trial_1_Filtered.snirf",
"data/beetroot/ProcessedData/Post_C3_Trial_2_Filtered.snirf",
"data/beetroot/ProcessedData/Post_C3_Trial_3_Filtered.snirf"
]

# Channel Layout:
after_channel_names = [ "S1-D1",
                        "S1-D2",
                        "S1-D3",
                        "S1-D4",
                        "S1-D5",
                        "S1-D6",
                        "S1-D7",
                        "S1-D8",
                        "S1-D9",
                        "S1-D10",
                        "S1-D11",
                        "S1-D12",
                        "S1-D13",
                        "S1-D14",
                        "S1-D15",
                        "S2-D1",
                        "S2-D4",
                        "S2-D5",
                        "S2-D9",
                        "S2-D12",
                        "S2-D14",
                        "S3-D1",
                        "S3-D2",
                        "S3-D6",
                        "S3-D9",
                        "S3-D10",
                        "S3-D15",
                        "S4-D2",
                        "S4-D3",
                        "S4-D7",
                        "S4-D11",
                        "S4-D16",
                        "S5-D3",
                        "S5-D4",
                        "S5-D8",
                        "S5-D11",
                        "S5-D12",
                        "S5-D13"
                        ]

before_channel_names = [    "S1-D1",
                            "S1-D2",
                            "S1-D3",
                            "S1-D4",
                            "S1-D5",
                            "S1-D6",
                            "S1-D7",
                            "S1-D8",
                            "S1-D9",
                            "S1-D10",
                            "S1-D11",
                            "S1-D12",
                            "S1-D13",
                            "S1-D15",
                            "S1-D16",
                            "S2-D1",
                            "S2-D4",
                            "S2-D5",
                            "S2-D9",
                            "S2-D10",
                            "S2-D13",
                            "S3-D1",
                            "S3-D2",
                            "S3-D6",
                            "S3-D10",
                            "S3-D14",
                            "S4-D2",
                            "S4-D3",
                            "S4-D7",
                            "S4-D11",
                            "S4-D12",
                            "S4-D15",
                            "S5-D3",
                            "S5-D4",
                            "S5-D8",
                            "S5-D9",
                            "S5-D12",
                            "S5-D16"
                            ]


# Left Hemisphere - Before and After
pre_c3_trials = [paths[3], paths[4], paths[5]]
post_c3_trials = [paths[9], paths[10], paths[11]]

# Right Hemisphere - Before and After
pre_c4_trials = [paths[0], paths[1], paths[2]]
post_c4_trials = [paths[6], paths[7], paths[8]]

#
# Plot All HbO - Before versus After

plt.subplot(2, 2, 1)
plt.title("HbO - Before")

plt.subplot(2, 2, 2)
plt.title("HbO - After")

plt.subplot(2, 2, 3)
plt.title("HbR - Before")

plt.subplot(2, 2, 4)
plt.title("HbR - After")

time = np.linspace(0, )

for i in range(len(pre_c3_trials)):
    before = read_snirf(pre_c4_trials[i])
    after = read_snirf(post_c4_trials[i])
    
    hbo_data = before[0]
    hbr_data = before[1]

    plt.subplot(2, 2, 1)
    for ch in hbo_data:
        plt.plot(ch)

    plt.subplot(2, 2, 1)
    plt.title("Pre-HbO")
    plt.plot(before[0].T)
    plt.axhline(y=100, color='black', linestyle='--', label="Reference")
    plt.axhline(y=-100, color='black', linestyle='--')
    plt.ylim(-200, 200)

    plt.subplot(2, 2, 2)
    plt.title("Pre-HbR")
    plt.plot(before[1].T)
    plt.axhline(y=20, color='black', linestyle='--', label="Reference")
    plt.axhline(y=-20, color='black', linestyle='--')
    plt.ylim(-40, 40)

    plt.subplot(2, 2, 3)
    plt.title("Post-HbO")
    plt.plot(after[0].T)
    plt.axhline(y=100, color='black', linestyle='--', label="Reference")
    plt.axhline(y=-100, color='black', linestyle='--')
    plt.ylim(-200, 200)

    plt.subplot(2, 2, 4)
    plt.title("Post-HbR")
    plt.plot(after[1].T)
    plt.axhline(y=20, color='black', linestyle='--', label="Reference")
    plt.axhline(y=-20, color='black', linestyle='--')
    plt.ylim(-40, 40)

    plt.tight_layout()
    plt.suptitle("Beetroot Analysis - Pre and Post Trials")
    plt.show()

